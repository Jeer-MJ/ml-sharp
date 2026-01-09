"""Spherical wrapping utilities for panoramic gaussians.

The key insight: SHARP generates gaussians from a flat image.
An equirectangular panorama IS a flat image representing a sphere.
We just need to "wrap" those flat gaussians onto a sphere.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.linalg import quaternion_product

LOGGER = logging.getLogger(__name__)


def wrap_gaussians_to_sphere(
    gaussians: Gaussians3D,
    original_aspect_ratio: float = 2.0,
) -> Gaussians3D:
    """Wrap flat gaussians onto a sphere using equirectangular projection.
    
    SHARP produces gaussians where X,Y are image-space positions and Z is depth.
    We use X,Y to determine direction (azimuth/elevation) and Z for radius.
    
    Since the panorama (2:1) was stretched to square (1:1), we need to
    compensate the Y coordinate by the aspect ratio.
    
    Args:
        gaussians: Flat gaussians from SHARP (from square 1536x1536 input)
        original_aspect_ratio: Original panorama width/height (2.0 for equirectangular)
        
    Returns:
        Gaussians wrapped onto a sphere
    """
    device = gaussians.mean_vectors.device
    positions = gaussians.mean_vectors.squeeze(0)  # (N, 3)
    quaternions = gaussians.quaternions.squeeze(0)  # (N, 4)
    
    # Get X, Y, Z from flat gaussians
    x = positions[:, 0]  # Horizontal position (image space)
    y = positions[:, 1]  # Vertical position (image space) 
    z = positions[:, 2]  # Depth
    
    # The gaussians are centered around (0, 0) after unprojection
    # X ranges roughly from -max to +max horizontally
    # Y ranges roughly from -max to +max vertically (but was stretched from 2:1)
    
    # Find the range of X and Y to normalize
    # X should map to [-π, π] (full 360°)
    # Y should map to [-π/2, π/2] (180°)
    
    # Since image was stretched from 2:1 to 1:1, Y is compressed
    # We need to expand Y back to original proportions
    y_corrected = y * original_aspect_ratio
    
    # Calculate angles from position
    # Use atan2 for proper angle calculation
    # Azimuth from X position relative to Z (or just X if we treat as image plane)
    # Elevation from Y position
    
    # For a pinhole camera with f=768 (half of 1536), the FOV is ~90°
    # But we want to map the full image to full sphere
    # So we normalize X, Y to [-1, 1] and then to angles
    
    # Get the bounds
    x_range = x.abs().max()
    y_range = y_corrected.abs().max()
    
    # Normalize to [-1, 1]
    x_norm = x / x_range if x_range > 0 else x
    y_norm = y_corrected / y_range if y_range > 0 else y_corrected
    
    # Convert to angles
    # X: -1 to 1 -> -π to π (360° azimuth)
    # Y: -1 to 1 -> π/2 to -π/2 (180° elevation, top to bottom)
    azimuth = x_norm * math.pi
    elevation = -y_norm * (math.pi / 2)  # Negative because Y increases downward
    
    # Get radius from depth (use absolute value, ensure minimum)
    radius = z.abs().clamp(min=0.1)
    
    # Normalize radius for better visualization
    # Center around 1.0 with reasonable range
    radius_median = radius.median()
    radius = radius / radius_median
    
    LOGGER.info(f"Position ranges: X=[{x.min():.2f}, {x.max():.2f}], "
                f"Y=[{y.min():.2f}, {y.max():.2f}], Z=[{z.min():.2f}, {z.max():.2f}]")
    LOGGER.info(f"Angle ranges: azimuth=[{azimuth.min():.2f}, {azimuth.max():.2f}], "
                f"elevation=[{elevation.min():.2f}, {elevation.max():.2f}]")
    
    # Convert spherical to Cartesian (Y-up convention)
    cos_elev = torch.cos(elevation)
    sin_elev = torch.sin(elevation)
    cos_azim = torch.cos(azimuth)
    sin_azim = torch.sin(azimuth)
    
    # Spherical to Cartesian
    new_x = radius * cos_elev * sin_azim
    new_y = radius * sin_elev
    new_z = radius * cos_elev * cos_azim
    
    new_positions = torch.stack([new_x, new_y, new_z], dim=-1)
    
    # Rotate quaternions to face outward from sphere center
    new_quaternions = rotate_quaternions_to_sphere_normal(
        quaternions, azimuth, elevation, device
    )
    
    # Scale singular values - gaussians further away should be larger
    depth_scale = radius.clamp(0.5, 2.0)
    new_singular_values = gaussians.singular_values * depth_scale.unsqueeze(0).unsqueeze(-1)
    
    n_gaussians = positions.shape[0]
    LOGGER.info(f"Wrapped {n_gaussians} gaussians to sphere")
    LOGGER.info(f"  Radius range: {radius.min():.2f} - {radius.max():.2f}")
    
    return Gaussians3D(
        mean_vectors=new_positions.unsqueeze(0),
        singular_values=new_singular_values,
        quaternions=new_quaternions.unsqueeze(0),
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def rotate_quaternions_to_sphere_normal(
    quaternions: torch.Tensor,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Rotate gaussian orientations to align with sphere surface normal.
    
    The original gaussians face +Z (camera forward).
    We need to rotate them to face outward from the sphere center.
    
    Args:
        quaternions: Original quaternions (N, 4) in [w, x, y, z] format
        azimuth: Azimuth angles for each gaussian
        elevation: Elevation angles for each gaussian
        device: Torch device
        
    Returns:
        Rotated quaternions
    """
    n = quaternions.shape[0]
    
    # Build rotation quaternion for each gaussian
    # First rotate around Y by azimuth, then around X by -elevation
    
    # Quaternion for Y rotation (azimuth)
    half_azim = azimuth / 2
    qy_w = torch.cos(half_azim)
    qy_x = torch.zeros_like(half_azim)
    qy_y = torch.sin(half_azim)
    qy_z = torch.zeros_like(half_azim)
    q_azimuth = torch.stack([qy_w, qy_x, qy_y, qy_z], dim=-1)
    
    # Quaternion for X rotation (-elevation to look outward)
    half_elev = -elevation / 2
    qx_w = torch.cos(half_elev)
    qx_x = torch.sin(half_elev)
    qx_y = torch.zeros_like(half_elev)
    qx_z = torch.zeros_like(half_elev)
    q_elevation = torch.stack([qx_w, qx_x, qx_y, qx_z], dim=-1)
    
    # Combined rotation: first azimuth, then elevation
    q_rotation = quaternion_product(q_azimuth, q_elevation)
    
    # Apply to original quaternions
    new_quaternions = quaternion_product(q_rotation, quaternions)
    
    # Normalize
    new_quaternions = new_quaternions / new_quaternions.norm(dim=-1, keepdim=True)
    
    return new_quaternions


def filter_by_depth_percentile(
    gaussians: Gaussians3D,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Gaussians3D:
    """Filter gaussians by depth percentile to remove outliers.
    
    Args:
        gaussians: Input gaussians
        low_percentile: Remove gaussians below this depth percentile
        high_percentile: Remove gaussians above this depth percentile
        
    Returns:
        Filtered gaussians
    """
    positions = gaussians.mean_vectors.squeeze(0)
    depths = positions[:, 2]
    
    low_thresh = torch.quantile(depths, low_percentile / 100.0)
    high_thresh = torch.quantile(depths, high_percentile / 100.0)
    
    mask = (depths >= low_thresh) & (depths <= high_thresh)
    
    n_removed = (~mask).sum().item()
    if n_removed > 0:
        LOGGER.info(f"Removed {n_removed} depth outliers")
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )
