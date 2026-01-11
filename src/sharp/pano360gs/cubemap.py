"""Cubemap projection utilities for equirectangular panoramas.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class CubeFace(IntEnum):
    """Cubemap face indices following standard convention."""
    
    FRONT = 0   # +Z
    RIGHT = 1   # +X
    BACK = 2    # -Z
    LEFT = 3    # -X
    TOP = 4     # +Y (Zenith)
    BOTTOM = 5  # -Y (Nadir)


class CubemapConfig(NamedTuple):
    """Configuration for cubemap extraction."""
    
    face_size: int = 1024
    padding_ratio: float = 0.20  # 20% padding on each side for seamless blending


def get_face_rotation_matrix(face: CubeFace) -> np.ndarray:
    """Get the 3x3 rotation matrix for a cubemap face.
    
    Returns matrix that rotates from face-local coordinates to world coordinates.
    Convention: +Z forward, +X right, +Y up in world space.
    
    Args:
        face: The cubemap face index.
        
    Returns:
        3x3 rotation matrix.
    """
    # Rotation matrices for each face (world from local)
    # Each row represents where the local axis ends up in world space
    rotations = {
        CubeFace.FRONT: np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32),  # Identity - looking at +Z
        
        CubeFace.RIGHT: np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=np.float32),  # 90° rotation around Y (looking at +X)
        
        CubeFace.BACK: np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=np.float32),  # 180° rotation around Y (looking at -Z)
        
        CubeFace.LEFT: np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=np.float32),  # -90° rotation around Y (looking at -X)
        
        CubeFace.TOP: np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32),  # -90° rotation around X (looking at +Y)
        
        CubeFace.BOTTOM: np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32),  # 90° rotation around X (looking at -Y)
    }
    return rotations[face]


def direction_to_equirectangular_uv(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D direction vectors to equirectangular UV coordinates.
    
    Args:
        direction: Array of shape (..., 3) with normalized direction vectors.
        
    Returns:
        Tuple of (u, v) arrays with values in [0, 1].
    """
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    
    # Spherical coordinates
    # phi: azimuth angle (-pi to pi), measured from +Z axis around Y
    # theta: elevation angle (-pi/2 to pi/2), measured from XZ plane
    phi = np.arctan2(x, z)  # Note: atan2(x, z) gives angle from +Z
    theta = np.arcsin(np.clip(y, -1.0, 1.0))
    
    # Convert to UV coordinates [0, 1]
    # Standard equirectangular: V=0 is top (north pole), V=1 is bottom (south pole)
    u = (phi / (2 * np.pi)) + 0.5
    v = 0.5 - (theta / np.pi)  # Inverted: +Y (up) -> V=0, -Y (down) -> V=1
    
    return u, v


def extract_cubemap_face(
    equirect_image: np.ndarray,
    face: CubeFace,
    config: CubemapConfig,
) -> np.ndarray:
    """Extract a single cubemap face from an equirectangular image.
    
    Uses bilinear interpolation and includes padding for seamless blending.
    
    Args:
        equirect_image: Equirectangular image of shape (H, W, C) where W = 2*H.
        face: Which cubemap face to extract.
        config: Cubemap configuration with face size and padding.
        
    Returns:
        Extracted face image of shape (face_size, face_size, C).
    """
    height, width = equirect_image.shape[:2]
    face_size = config.face_size
    padding = int(face_size * config.padding_ratio)
    padded_size = face_size + 2 * padding
    
    # Get rotation matrix for this face
    rotation = get_face_rotation_matrix(face)
    
    # Create pixel grid for the face (including padding)
    # FOV = 90° means tan(45°) = 1, so at z=1, x and y range from -1 to 1
    half_size = padded_size / 2
    
    # Pixel coordinates (centered)
    y_coords = np.arange(padded_size) - half_size + 0.5
    x_coords = np.arange(padded_size) - half_size + 0.5
    
    # Scale to match FOV (for 90° FOV with padding)
    # The actual FOV needs to be slightly larger to account for padding
    fov_scale = (padded_size / face_size)  # Scale factor for padded FOV
    
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    x_grid = x_grid / half_size * fov_scale
    y_grid = -y_grid / half_size * fov_scale  # Flip Y to match image coordinates
    z_grid = np.ones_like(x_grid)
    
    # Stack into direction vectors (local face coordinates)
    directions_local = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # Normalize directions
    directions_local = directions_local / np.linalg.norm(
        directions_local, axis=-1, keepdims=True
    )
    
    # Rotate to world coordinates
    directions_world = directions_local @ rotation.T
    
    # Convert to equirectangular UV
    u, v = direction_to_equirectangular_uv(directions_world)
    
    # Convert UV to pixel coordinates
    px = u * width
    py = v * height
    
    # Handle wrapping for U coordinate (panorama wraps horizontally)
    px = px % width
    
    # Bilinear interpolation
    px0 = np.floor(px).astype(np.int32)
    py0 = np.floor(py).astype(np.int32)
    px1 = (px0 + 1) % width  # Wrap horizontally
    py1 = np.clip(py0 + 1, 0, height - 1)  # Clamp vertically
    
    # Clamp py0 as well
    py0 = np.clip(py0, 0, height - 1)
    
    # Interpolation weights
    wx = px - px0
    wy = py - py0
    
    # Sample four corners
    c00 = equirect_image[py0, px0]
    c01 = equirect_image[py0, px1]
    c10 = equirect_image[py1, px0]
    c11 = equirect_image[py1, px1]
    
    # Bilinear interpolation
    wx = wx[..., np.newaxis]
    wy = wy[..., np.newaxis]
    
    face_image = (
        c00 * (1 - wx) * (1 - wy) +
        c01 * wx * (1 - wy) +
        c10 * (1 - wx) * wy +
        c11 * wx * wy
    )
    
    return face_image.astype(equirect_image.dtype)


def extract_all_cubemap_faces(
    equirect_image: np.ndarray,
    config: CubemapConfig | None = None,
) -> dict[CubeFace, np.ndarray]:
    """Extract all 6 cubemap faces from an equirectangular image.
    
    Args:
        equirect_image: Equirectangular image of shape (H, W, C) where W = 2*H.
        config: Optional cubemap configuration.
        
    Returns:
        Dictionary mapping face indices to extracted face images.
    """
    if config is None:
        config = CubemapConfig()
    
    height, width = equirect_image.shape[:2]
    expected_ratio = width / height
    
    if abs(expected_ratio - 2.0) > 0.01:
        LOGGER.warning(
            f"Image aspect ratio {expected_ratio:.2f} differs from expected 2:1. "
            "Results may be distorted."
        )
    
    if width < 4096:
        LOGGER.warning(
            f"Image width {width} is less than recommended 4096px. "
            "Quality may be reduced."
        )
    
    LOGGER.info(f"Extracting 6 cubemap faces of size {config.face_size}x{config.face_size}")
    
    faces = {}
    for face in CubeFace:
        LOGGER.debug(f"Extracting face {face.name}")
        faces[face] = extract_cubemap_face(equirect_image, face, config)
    
    return faces


def get_face_focal_length(face_size: int) -> float:
    """Calculate focal length for a cubemap face with 90° FOV.
    
    For a 90° FOV: tan(45°) = 1 = (face_size/2) / focal_length
    Therefore: focal_length = face_size / 2
    
    Args:
        face_size: Size of the cubemap face in pixels.
        
    Returns:
        Focal length in pixels.
    """
    return face_size / 2.0


def get_padding_mask(
    face_size: int,
    padding_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a mask for the valid (non-padded) region of a face.
    
    Args:
        face_size: Original face size without padding.
        padding_ratio: Ratio of padding added on each side.
        device: Torch device for the mask.
        
    Returns:
        Boolean mask of shape (face_size + 2*padding, face_size + 2*padding).
    """
    padding = int(face_size * padding_ratio)
    padded_size = face_size + 2 * padding
    
    mask = torch.zeros(padded_size, padded_size, dtype=torch.bool, device=device)
    mask[padding:padding + face_size, padding:padding + face_size] = True
    
    return mask


def get_gaussian_center_mask(
    gaussians_xy: torch.Tensor,
    face_size: int,
    padding_ratio: float,
) -> torch.Tensor:
    """Create a mask for gaussians whose centers are in the valid region.
    
    After prediction, gaussians near the padded border should be discarded
    to avoid duplicates when merging faces.
    
    Args:
        gaussians_xy: XY positions of gaussians in image space, shape (N, 2).
        face_size: Original face size without padding.
        padding_ratio: Ratio of padding added on each side.
        
    Returns:
        Boolean mask of shape (N,) indicating valid gaussians.
    """
    padding = int(face_size * padding_ratio)
    padded_size = face_size + 2 * padding
    
    # Convert from normalized coordinates to pixel coordinates
    # Assuming gaussians_xy is in range [-1, 1] (NDC)
    pixels_x = (gaussians_xy[:, 0] + 1) * 0.5 * padded_size
    pixels_y = (gaussians_xy[:, 1] + 1) * 0.5 * padded_size
    
    # Check if within valid region
    valid_x = (pixels_x >= padding) & (pixels_x < padding + face_size)
    valid_y = (pixels_y >= padding) & (pixels_y < padding + face_size)
    
    return valid_x & valid_y
