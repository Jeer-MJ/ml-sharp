"""Equirectangular Projection Module for 360° Gaussian Splatting.

This module provides the mathematical foundation for converting equirectangular
(360° panoramic) images and their associated depth maps into 3D Gaussian Splat
representations using spherical projection.

Mathematical Background
-----------------------

An equirectangular projection maps a sphere onto a 2D rectangular image:

    +------------------------------------------+
    | North Pole (θ = +π/2)                    |  v=0
    |                                          |
    | ← φ=-π        Center (φ=0)      φ=+π →  |  v=H/2
    |                                          |
    | South Pole (θ = -π/2)                    |  v=H
    +------------------------------------------+
      u=0                                  u=W

Where:
    - u: horizontal pixel coordinate [0, W)
    - v: vertical pixel coordinate [0, H)
    - φ (phi): longitude angle [-π, +π]
    - θ (theta): latitude angle [-π/2, +π/2]

Conversion formulas:
    φ = (u / W) × 2π - π
    θ = (v / H) × π - π/2

Spherical to Cartesian (Y-up, OpenCV convention):
    x = depth × cos(θ) × sin(φ)
    y = depth × sin(θ)
    z = depth × cos(θ) × cos(φ)

This places the viewer at the origin looking along +Z, with +Y up.

Scale Compensation
------------------

In equirectangular projection, pixels near the poles represent smaller solid
angles than pixels at the equator. To maintain uniform visual density, we
scale Gaussian splats based on their latitude:

    scale_adjusted = scale_predicted × depth × k × cos(θ)

Where k is a density factor that depends on the desired output quality.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch

from sharp.utils.gaussians import (
    Gaussians3D,
    compose_covariance_matrices,
    decompose_covariance_matrices,
)
from sharp.utils import linalg

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


class PanoQuality(Enum):
    """Quality presets for 360° Gaussian Splatting output.
    
    Each level defines:
        - resolution_factor: Downsampling applied to input before processing
        - scale_factor: Multiplier for Gaussian scales (larger = less detail)
        - approx_splats: Approximate number of output Gaussian splats
    
    Usage Notes
    -----------
    - LOW: Fast preview, suitable for quick iterations
    - MEDIUM: Balanced quality/performance, good for web viewers
    - HIGH: High quality, suitable for desktop VR applications
    - ULTRA: Maximum quality, may require high-end GPU for real-time rendering
    """
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class QualityConfig(NamedTuple):
    """Configuration parameters for a quality preset.
    
    Attributes
    ----------
    resolution_factor : float
        Factor by which to scale the input resolution. 
        1.0 = full resolution, 0.5 = half resolution.
    
    scale_multiplier : float
        Multiplier applied to Gaussian scales to compensate for 
        reduced density at lower quality levels.
    
    depth_scale_factor : float
        Factor for depth-based scale adjustment.
        Controls how much depth affects Gaussian size.
    
    latitude_compensation : bool
        Whether to apply cosine latitude compensation.
        Helps maintain uniform density across the sphere.
    """
    resolution_factor: float
    scale_multiplier: float
    depth_scale_factor: float
    latitude_compensation: bool


# Quality preset configurations
# These values are tuned for typical 4K (4096x2048) equirectangular inputs
QUALITY_CONFIGS: dict[PanoQuality, QualityConfig] = {
    PanoQuality.LOW: QualityConfig(
        resolution_factor=0.25,
        scale_multiplier=4.0,
        depth_scale_factor=0.004,
        latitude_compensation=True,
    ),
    PanoQuality.MEDIUM: QualityConfig(
        resolution_factor=0.5,
        scale_multiplier=2.0,
        depth_scale_factor=0.002,
        latitude_compensation=True,
    ),
    PanoQuality.HIGH: QualityConfig(
        resolution_factor=0.75,
        scale_multiplier=1.33,
        depth_scale_factor=0.0015,
        latitude_compensation=True,
    ),
    PanoQuality.ULTRA: QualityConfig(
        resolution_factor=1.0,
        scale_multiplier=1.0,
        depth_scale_factor=0.001,
        latitude_compensation=True,
    ),
}


# ==============================================================================
# VALIDATION
# ==============================================================================


def validate_equirectangular_aspect_ratio(
    width: int,
    height: int,
    tolerance: float = 0.1,
) -> bool:
    """Validate that an image has the correct aspect ratio for equirectangular format.
    
    An equirectangular panorama should have a 2:1 aspect ratio (width:height),
    as it maps 360° horizontally and 180° vertically.
    
    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    tolerance : float, optional
        Allowed deviation from the ideal 2:1 ratio. Default is 0.1 (10%).
    
    Returns
    -------
    bool
        True if the aspect ratio is within the acceptable range.
    
    Raises
    ------
    ValueError
        If the aspect ratio deviates too far from 2:1.
    
    Examples
    --------
    >>> validate_equirectangular_aspect_ratio(4096, 2048)  # Perfect 2:1
    True
    >>> validate_equirectangular_aspect_ratio(3840, 1920)  # Also 2:1
    True
    >>> validate_equirectangular_aspect_ratio(1920, 1080)  # 16:9, not equirectangular
    ValueError: Image aspect ratio 1.78 is not suitable for equirectangular...
    """
    aspect_ratio = width / height
    ideal_ratio = 2.0
    
    if abs(aspect_ratio - ideal_ratio) / ideal_ratio > tolerance:
        raise ValueError(
            f"Image aspect ratio {aspect_ratio:.2f} is not suitable for "
            f"equirectangular projection. Expected ratio close to 2:1 "
            f"(360° × 180°). Got {width}×{height}."
        )
    
    LOGGER.debug(
        "Validated equirectangular aspect ratio: %d×%d (ratio: %.3f)",
        width, height, aspect_ratio
    )
    return True


# ==============================================================================
# CORE PROJECTION FUNCTIONS
# ==============================================================================


def create_spherical_coordinates(
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create longitude (phi) and latitude (theta) grids for equirectangular projection.
    
    This function generates the angular coordinate grids used to map 2D pixel
    positions to 3D spherical directions.
    
    Parameters
    ----------
    width : int
        Image width in pixels (corresponds to 360° longitude).
    height : int
        Image height in pixels (corresponds to 180° latitude).
    device : torch.device
        Device to create tensors on (CPU, CUDA, MPS).
    
    Returns
    -------
    phi : torch.Tensor
        Longitude angles in radians, shape (H, W), range [-π, +π].
        Positive values point right/east, negative values point left/west.
    theta : torch.Tensor
        Latitude angles in radians, shape (H, W), range [-π/2, +π/2].
        Positive values point up (north pole), negative values point down.
    
    Notes
    -----
    The coordinate system follows the convention:
    - Phi = 0 corresponds to the center of the image (looking forward, +Z)
    - Phi = ±π corresponds to looking backward (-Z)
    - Theta = 0 corresponds to the equator (horizon)
    - Theta = +π/2 corresponds to looking straight up (+Y, north pole)
    - Theta = -π/2 corresponds to looking straight down (-Y, south pole)
    
    Implementation Detail
    ---------------------
    We use pixel centers, so pixel (0, 0) maps to the top-left corner
    of the leftmost-topmost pixel, offset by half a pixel to the center.
    """
    # Normalized coordinates [0, 1) for pixel centers
    u_normalized = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    v_normalized = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    
    # Create 2D grids
    uu, vv = torch.meshgrid(u_normalized, v_normalized, indexing='xy')
    
    # Convert to spherical angles
    # Longitude (phi): 0->2π, then shift to -π->+π
    phi = uu * 2.0 * np.pi - np.pi
    
    # Latitude (theta): 0->π, then shift to +π/2->-π/2
    # Note: v=0 is top of image = north pole = +π/2
    theta = (0.5 - vv) * np.pi
    
    return phi, theta


def spherical_to_cartesian(
    phi: torch.Tensor,
    theta: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Convert spherical coordinates to Cartesian coordinates.
    
    Uses the Y-up convention compatible with OpenCV and most 3D engines:
        - X: Right
        - Y: Up  
        - Z: Forward (into the scene)
    
    Parameters
    ----------
    phi : torch.Tensor
        Longitude angles in radians, range [-π, +π].
    theta : torch.Tensor
        Latitude angles in radians, range [-π/2, +π/2].
    radius : torch.Tensor
        Distance from origin (depth values).
    
    Returns
    -------
    xyz : torch.Tensor
        Cartesian coordinates, shape (*input_shape, 3).
    
    Mathematical Derivation
    -----------------------
    Starting from spherical coordinates (r, θ, φ) where:
        - r = radius (depth)
        - θ = latitude (elevation from equator)
        - φ = longitude (azimuth from forward direction)
    
    The conversion to Cartesian is:
        x = r × cos(θ) × sin(φ)
        y = r × sin(θ)
        z = r × cos(θ) × cos(φ)
    
    This places:
        - φ=0, θ=0: Point along +Z axis (forward)
        - φ=π/2, θ=0: Point along +X axis (right)
        - φ=0, θ=π/2: Point along +Y axis (up)
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    x = radius * cos_theta * sin_phi
    y = radius * sin_theta
    z = radius * cos_theta * cos_phi
    
    return torch.stack((x, y, z), dim=-1)


def equirectangular_to_xyz(
    depth_map: torch.Tensor,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor:
    """Convert an equirectangular depth map to 3D Cartesian point cloud.
    
    This is the core function that transforms a 2D depth map from a 360°
    panoramic image into a 3D point cloud positioned on a sphere.
    
    Parameters
    ----------
    depth_map : torch.Tensor
        Depth values for each pixel. Can be:
            - Shape (H, W): Single depth map
            - Shape (B, H, W): Batch of depth maps
            - Shape (B, 1, H, W): Batch with channel dimension
    width : int, optional
        Image width. If None, inferred from depth_map shape.
    height : int, optional
        Image height. If None, inferred from depth_map shape.
    
    Returns
    -------
    xyz : torch.Tensor
        3D coordinates for each pixel, shape (*input_shape, 3).
    
    Example
    -------
    >>> depth = torch.rand(2048, 4096)  # 360° panorama depth
    >>> points = equirectangular_to_xyz(depth)
    >>> points.shape
    torch.Size([2048, 4096, 3])
    
    Notes
    -----
    The resulting point cloud forms a complete sphere around the origin.
    Points with depth=0 will be at the origin; consider filtering these
    or replacing with a minimum depth value.
    """
    # Handle various input shapes
    original_shape = depth_map.shape
    if depth_map.dim() == 4:  # (B, C, H, W)
        depth_map = depth_map.squeeze(1)  # Remove channel dim
    
    if depth_map.dim() == 2:  # (H, W)
        depth_map = depth_map.unsqueeze(0)  # Add batch dim
    
    batch_size, h, w = depth_map.shape
    
    # Use provided dimensions or infer from tensor
    width = width or w
    height = height or h
    
    # Create spherical coordinate grids
    phi, theta = create_spherical_coordinates(width, height, depth_map.device)
    
    # Expand grids for batch processing
    phi = phi.unsqueeze(0).expand(batch_size, -1, -1)
    theta = theta.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Convert to Cartesian
    xyz = spherical_to_cartesian(phi, theta, depth_map)
    
    # Restore original batch dimensions if needed
    if len(original_shape) == 2:
        xyz = xyz.squeeze(0)
    
    return xyz


# ==============================================================================
# GAUSSIAN SCALE ADJUSTMENT
# ==============================================================================


def compute_latitude_compensation(
    theta: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute scale compensation factor based on latitude.
    
    In equirectangular projection, pixels near the poles represent smaller
    areas on the sphere than pixels at the equator. This function computes
    a compensation factor to maintain uniform visual density.
    
    Parameters
    ----------
    theta : torch.Tensor
        Latitude angles in radians, range [-π/2, +π/2].
    epsilon : float, optional
        Small value to prevent division by zero at poles.
    
    Returns
    -------
    compensation : torch.Tensor
        Scale multipliers, larger near poles, 1.0 at equator.
    
    Mathematical Background
    -----------------------
    The solid angle per pixel in equirectangular projection is proportional
    to cos(θ). To compensate, we scale by 1/cos(θ), clamped to prevent
    extreme values at the poles.
    
    In practice, we use a softer compensation to avoid overly large splats
    at the poles:
        compensation = 1.0 / max(cos(θ), epsilon)
    
    Clamped to a maximum of 3.0 to prevent explosion at exact poles.
    """
    cos_theta = torch.cos(theta).clamp(min=epsilon)
    compensation = (1.0 / cos_theta).clamp(max=3.0)
    return compensation


def adjust_gaussian_scales_for_sphere(
    singular_values: torch.Tensor,
    depth: torch.Tensor,
    theta: torch.Tensor,
    config: QualityConfig,
) -> torch.Tensor:
    """Adjust Gaussian scales for spherical projection.
    
    Applies three adjustments to predicted Gaussian scales:
    1. Depth scaling: Larger scales for distant objects
    2. Latitude compensation: Larger scales near poles (if enabled)
    3. Quality multiplier: Overall scale adjustment for output density
    
    Parameters
    ----------
    singular_values : torch.Tensor
        Predicted Gaussian scales (singular values of covariance),
        shape (N, 3) where N is number of Gaussians.
    depth : torch.Tensor
        Depth values for each Gaussian, shape (N,).
    theta : torch.Tensor
        Latitude angles for each Gaussian, shape (N,).
    config : QualityConfig
        Quality configuration with scale parameters.
    
    Returns
    -------
    adjusted_scales : torch.Tensor
        Adjusted singular values, same shape as input.
    
    Notes
    -----
    The formula applied is:
        scale_final = scale_pred × depth × depth_scale_factor 
                    × latitude_comp × quality_multiplier
    
    This ensures that:
    - Distant splats are larger (maintains coverage)
    - Polar splats are larger (compensates for projection distortion)
    - Overall density matches the selected quality level
    """
    # Base depth scaling
    depth_factor = depth.unsqueeze(-1) * config.depth_scale_factor
    
    # Latitude compensation (if enabled)
    if config.latitude_compensation:
        lat_comp = compute_latitude_compensation(theta).unsqueeze(-1)
    else:
        lat_comp = 1.0
    
    # Apply all scale adjustments
    adjusted_scales = (
        singular_values 
        * depth_factor 
        * lat_comp 
        * config.scale_multiplier
    )
    
    return adjusted_scales


# ==============================================================================
# MAIN UNPROJECTION FUNCTION
# ==============================================================================


def unproject_gaussians_equirectangular(
    gaussians_ndc: Gaussians3D,
    image_shape: tuple[int, int],
    quality: PanoQuality = PanoQuality.MEDIUM,
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates using spherical projection.
    
    This is the main entry point for converting ml-sharp's predicted Gaussians
    (in normalized device coordinates) to a 360° spherical representation.
    
    Unlike the standard `unproject_gaussians()` which uses pinhole camera intrinsics,
    this function applies equirectangular (spherical) projection mathematics.
    
    Parameters
    ----------
    gaussians_ndc : Gaussians3D
        Predicted Gaussians in normalized device coordinates.
        Expected fields:
            - mean_vectors: (B, N, 3) where channels are (x_ndc, y_ndc, inv_depth)
            - singular_values: (B, N, 3) predicted scales
            - quaternions: (B, N, 4) predicted rotations
            - colors: (B, N, 3) predicted RGB values
            - opacities: (B, N) predicted opacity values
    image_shape : tuple[int, int]
        Original image dimensions as (width, height).
    quality : PanoQuality, optional
        Quality preset controlling output density. Default is MEDIUM.
    
    Returns
    -------
    Gaussians3D
        Transformed Gaussians in world coordinates (spherical distribution).
    
    Algorithm
    ---------
    1. Extract NDC coordinates (x_ndc, y_ndc, inverse_depth)
    2. Convert NDC x,y to pixel coordinates (u, v)
    3. Convert pixel coordinates to spherical angles (phi, theta)
    4. Convert depth from inverse depth to actual depth
    5. Apply spherical-to-Cartesian transformation for mean positions
    6. Adjust Gaussian scales based on depth and latitude
    7. Transform covariance matrices to world space
    
    Example
    -------
    >>> from sharp.models import create_predictor, PredictorParams
    >>> from sharp.pano360gs import unproject_gaussians_equirectangular, PanoQuality
    >>> 
    >>> # Run standard prediction
    >>> predictor = create_predictor(PredictorParams())
    >>> gaussians_ndc = predictor(panorama_image, disparity_factor)
    >>> 
    >>> # Convert to 360° representation
    >>> gaussians_3d = unproject_gaussians_equirectangular(
    ...     gaussians_ndc,
    ...     image_shape=(4096, 2048),
    ...     quality=PanoQuality.HIGH
    ... )
    
    Notes
    -----
    - The output Gaussians form a complete sphere around the origin
    - The "forward" direction (camera look-at) corresponds to the center
      of the equirectangular image
    - Consider applying additional filtering to remove low-opacity Gaussians
      for performance optimization
    
    See Also
    --------
    equirectangular_to_xyz : Core projection function
    adjust_gaussian_scales_for_sphere : Scale adjustment logic
    sharp.utils.gaussians.unproject_gaussians : Standard pinhole unprojection
    """
    width, height = image_shape
    device = gaussians_ndc.mean_vectors.device
    config = QUALITY_CONFIGS[quality]
    
    LOGGER.info(
        "Unprojecting Gaussians with equirectangular projection "
        "(quality=%s, image=%dx%d)",
        quality.value, width, height
    )
    
    # Validate input dimensions
    validate_equirectangular_aspect_ratio(width, height)
    
    # Extract mean vectors: (B, N, 3) -> channels are (x_ndc, y_ndc, inverse_depth)
    mean_ndc = gaussians_ndc.mean_vectors
    batch_size, num_gaussians, _ = mean_ndc.shape
    
    # NDC coordinates: x_ndc, y_ndc are in [-1, 1] range
    x_ndc = mean_ndc[..., 0]  # (B, N)
    y_ndc = mean_ndc[..., 1]  # (B, N)
    inv_depth = mean_ndc[..., 2]  # (B, N)
    
    # Convert inverse depth to depth (clamp to avoid division issues)
    depth = 1.0 / inv_depth.clamp(min=1e-4, max=1e4)
    
    # Convert NDC [-1, 1] to normalized image coordinates [0, 1]
    u_normalized = (x_ndc + 1.0) / 2.0
    v_normalized = (y_ndc + 1.0) / 2.0
    
    # Convert to spherical angles
    # Longitude (phi): maps [0, 1] to [-π, +π]
    phi = u_normalized * 2.0 * np.pi - np.pi
    
    # Latitude (theta): maps [0, 1] to [+π/2, -π/2] (top=north pole)
    theta = (0.5 - v_normalized) * np.pi
    
    # Convert spherical to Cartesian coordinates
    mean_world = spherical_to_cartesian(phi, theta, depth)
    
    # Adjust scales for spherical projection
    # Flatten for processing, then reshape back
    singular_values_flat = gaussians_ndc.singular_values.flatten(0, 1)  # (B*N, 3)
    depth_flat = depth.flatten()  # (B*N,)
    theta_flat = theta.flatten()  # (B*N,)
    
    adjusted_scales = adjust_gaussian_scales_for_sphere(
        singular_values_flat,
        depth_flat,
        theta_flat,
        config,
    )
    
    adjusted_scales = adjusted_scales.view(batch_size, num_gaussians, 3)
    
    # Transform quaternions to align with spherical coordinates
    # The Gaussians need to be oriented tangent to the sphere
    adjusted_quaternions = _adjust_quaternions_for_sphere(
        gaussians_ndc.quaternions,
        phi,
        theta,
    )
    
    LOGGER.info(
        "Unprojection complete: %d Gaussians, depth range [%.2f, %.2f]",
        num_gaussians,
        depth.min().item(),
        depth.max().item()
    )
    
    return Gaussians3D(
        mean_vectors=mean_world,
        singular_values=adjusted_scales,
        quaternions=adjusted_quaternions,
        colors=gaussians_ndc.colors,
        opacities=gaussians_ndc.opacities,
    )


def _adjust_quaternions_for_sphere(
    quaternions: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Adjust Gaussian orientations to be tangent to the sphere surface.
    
    The predicted quaternions assume a planar image. For spherical projection,
    we need to rotate each Gaussian so that its "flat" side is tangent to
    the sphere at its position.
    
    Parameters
    ----------
    quaternions : torch.Tensor
        Predicted quaternions, shape (B, N, 4) in (w, x, y, z) format.
    phi : torch.Tensor
        Longitude angles, shape (B, N).
    theta : torch.Tensor
        Latitude angles, shape (B, N).
    
    Returns
    -------
    adjusted_quaternions : torch.Tensor
        Rotated quaternions, same shape as input.
    
    Implementation Notes
    --------------------
    We construct a rotation that:
    1. Rotates around Y-axis by phi (longitude rotation)
    2. Rotates around the local X-axis by theta (latitude tilt)
    
    This orients the Gaussian's local Z-axis to point toward the sphere center,
    with the local XY plane tangent to the sphere surface.
    """
    device = quaternions.device
    batch_size, num_gaussians, _ = quaternions.shape
    
    # Flatten for processing
    phi_flat = phi.flatten()
    theta_flat = theta.flatten()
    
    # Create rotation quaternions for longitude (Y-axis rotation)
    half_phi = phi_flat / 2.0
    q_longitude = torch.stack([
        torch.cos(half_phi),             # w
        torch.zeros_like(half_phi),      # x
        torch.sin(half_phi),             # y
        torch.zeros_like(half_phi),      # z
    ], dim=-1)
    
    # Create rotation quaternions for latitude (X-axis rotation, after longitude)
    # Note: negative theta because we're rotating "down" from north pole
    half_theta = -theta_flat / 2.0
    q_latitude = torch.stack([
        torch.cos(half_theta),           # w
        torch.sin(half_theta),           # x
        torch.zeros_like(half_theta),    # y
        torch.zeros_like(half_theta),    # z
    ], dim=-1)
    
    # Combine rotations: first longitude, then latitude
    # Quaternion multiplication: q_combined = q_latitude * q_longitude
    q_sphere = _quaternion_multiply(q_latitude, q_longitude)
    
    # Apply sphere rotation to predicted quaternions
    # Final = q_sphere * q_predicted
    q_flat = quaternions.flatten(0, 1)  # (B*N, 4)
    q_adjusted = _quaternion_multiply(q_sphere, q_flat)
    
    # Normalize to ensure unit quaternions
    q_adjusted = q_adjusted / q_adjusted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    return q_adjusted.view(batch_size, num_gaussians, 4)


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions using Hamilton product.
    
    Parameters
    ----------
    q1, q2 : torch.Tensor
        Quaternions in (w, x, y, z) format, shape (..., 4).
    
    Returns
    -------
    q_product : torch.Tensor
        Product quaternion q1 * q2, same shape as inputs.
    
    Notes
    -----
    Hamilton product formula:
        (a1 + b1*i + c1*j + d1*k) × (a2 + b2*i + c2*j + d2*k) =
        (a1*a2 - b1*b2 - c1*c2 - d1*d2) +
        (a1*b2 + b1*a2 + c1*d2 - d1*c2)*i +
        (a1*c2 - b1*d2 + c1*a2 + d1*b2)*j +
        (a1*d2 + b1*c2 - c1*b2 + d1*a2)*k
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)
