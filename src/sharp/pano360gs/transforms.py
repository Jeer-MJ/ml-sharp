"""Gaussian transformation utilities for panoramic reconstruction.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sharp.utils.gaussians import Gaussians3D, apply_transform
from sharp.utils.linalg import quaternion_product

from .cubemap import CubeFace, get_face_rotation_matrix

LOGGER = logging.getLogger(__name__)


def rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> torch.Tensor:
    """Convert rotation matrix to quaternion in SHARP convention [w, x, y, z].
    
    Args:
        rotation_matrix: 3x3 rotation matrix.
        
    Returns:
        Quaternion tensor of shape (4,) in [w, x, y, z] format.
    """
    # scipy returns [x, y, z, w]
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    # Convert to [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return torch.tensor(quat_wxyz, dtype=torch.float32)


def get_face_transform_matrix(face: CubeFace) -> torch.Tensor:
    """Get the 4x4 transformation matrix for a cubemap face.
    
    This matrix transforms points from face-local space to world space.
    
    Args:
        face: The cubemap face.
        
    Returns:
        4x4 homogeneous transformation matrix.
    """
    rotation = get_face_rotation_matrix(face)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    return torch.tensor(transform, dtype=torch.float32)


def transform_face_gaussians(
    gaussians: Gaussians3D,
    face: CubeFace,
) -> Gaussians3D:
    """Transform gaussians from face-local space to world space.
    
    Uses the existing apply_transform function which properly handles
    both positions and covariance matrices (via quaternions).
    
    Args:
        gaussians: Gaussians in face-local coordinates.
        face: The cubemap face these gaussians came from.
        
    Returns:
        Gaussians transformed to world coordinates.
    """
    device = gaussians.mean_vectors.device
    transform = get_face_transform_matrix(face).to(device)
    
    # apply_transform expects a 3x4 matrix
    transform_3x4 = transform[:3]
    
    return apply_transform(gaussians, transform_3x4)


def filter_gaussians_by_depth(
    gaussians: Gaussians3D,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
) -> Gaussians3D:
    """Filter gaussians by depth range.
    
    Args:
        gaussians: Input gaussians.
        min_depth: Minimum depth threshold.
        max_depth: Maximum depth threshold.
        
    Returns:
        Filtered gaussians.
    """
    # Compute depth as distance from origin
    depths = torch.norm(gaussians.mean_vectors, dim=-1)
    
    # Create mask
    mask = (depths > min_depth) & (depths < max_depth)
    mask = mask.squeeze(0)  # Remove batch dimension
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )


def filter_gaussians_in_face_frustum(
    gaussians: Gaussians3D,
    padding_ratio: float = 0.0,
    edge_fade_ratio: float = 0.15,
) -> Gaussians3D:
    """Filter gaussians to keep only those within the face frustum.
    
    For a 90° FOV face looking at +Z, valid gaussians have:
    - z > 0 (in front of camera)
    - |x/z| < 1 (within horizontal FOV)
    - |y/z| < 1 (within vertical FOV)
    
    Also reduces opacity for gaussians near the edges to help blending.
    
    Args:
        gaussians: Input gaussians in face-local coordinates.
        padding_ratio: Allow gaussians slightly outside frustum.
        edge_fade_ratio: Ratio of frustum edge where opacity fades.
        
    Returns:
        Filtered gaussians within the frustum.
    """
    positions = gaussians.mean_vectors.squeeze(0)  # (N, 3)
    opacities = gaussians.opacities.clone()
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    # Must be in front of camera
    valid_depth = z > 0.001
    
    # Compute normalized frustum coordinates
    z_safe = z.clamp(min=0.001)
    frustum_x = torch.abs(x / z_safe)
    frustum_y = torch.abs(y / z_safe)
    
    # Within FOV (with optional padding allowance)
    fov_limit = 1.0 - padding_ratio  # Stricter limit to avoid edge artifacts
    valid_x = frustum_x < fov_limit
    valid_y = frustum_y < fov_limit
    
    mask = valid_depth & valid_x & valid_y
    
    # Apply edge fade to reduce seams
    # Gaussians near the edge get reduced opacity
    fade_start = 1.0 - padding_ratio - edge_fade_ratio
    max_frustum = torch.max(frustum_x, frustum_y)
    
    # Linear fade from fade_start to fov_limit
    edge_factor = torch.clamp(
        (fov_limit - max_frustum) / (fov_limit - fade_start),
        0.0, 1.0
    )
    
    # Apply fade to opacities
    opacities_faded = opacities.squeeze(0) * edge_factor
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=opacities_faded[mask].unsqueeze(0),
    )


def reposition_gaussians_for_panorama(
    gaussians: Gaussians3D,
    face: CubeFace,
    sphere_radius: float = 1.0,
) -> Gaussians3D:
    """Reposition gaussians to form a coherent panoramic sphere.
    
    Places gaussians on a sphere while preserving their relative depths.
    
    Args:
        gaussians: Input gaussians in face-local coordinates (z = depth).
        face: The cubemap face these gaussians came from.
        sphere_radius: Base radius for the panoramic sphere.
        
    Returns:
        Gaussians repositioned in world coordinates.
    """
    device = gaussians.mean_vectors.device
    rotation_matrix = get_face_rotation_matrix(face)
    rotation = torch.tensor(rotation_matrix, dtype=torch.float32, device=device)
    
    positions = gaussians.mean_vectors.squeeze(0)  # (N, 3)
    
    # The z coordinate represents depth from the face
    # x, y represent position on the face plane
    # Convert to spherical direction and scale by depth
    
    # Get depth (z in local coords is depth)
    depths = positions[:, 2:3].clamp(min=0.01)
    
    # Normalize x, y by depth to get direction on unit sphere
    directions_local = positions / depths
    
    # Now scale by actual depth
    positions_local = directions_local * depths
    
    # Rotate to world coordinates
    positions_world = positions_local @ rotation.T
    
    # Update quaternions by the face rotation
    face_quat = rotation_matrix_to_quaternion_wxyz(rotation_matrix).to(device)
    
    # For each gaussian, rotate its orientation
    original_quats = gaussians.quaternions.squeeze(0)  # (N, 4)
    
    # Broadcast face quaternion and multiply
    face_quat_expanded = face_quat.unsqueeze(0).expand(original_quats.shape[0], -1)
    rotated_quats = quaternion_product(face_quat_expanded, original_quats)
    
    return Gaussians3D(
        mean_vectors=positions_world.unsqueeze(0),
        singular_values=gaussians.singular_values,
        quaternions=rotated_quats.unsqueeze(0),
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
