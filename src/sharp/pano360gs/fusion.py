"""Gaussian fusion utilities for combining cubemap faces.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D, save_ply

LOGGER = logging.getLogger(__name__)


def merge_gaussians(gaussians_list: list[Gaussians3D]) -> Gaussians3D:
    """Merge multiple Gaussians3D objects into one.
    
    Args:
        gaussians_list: List of Gaussians3D to merge.
        
    Returns:
        Single merged Gaussians3D object.
    """
    if not gaussians_list:
        raise ValueError("Cannot merge empty list of gaussians")
    
    device = gaussians_list[0].mean_vectors.device
    
    # Concatenate all tensors along the gaussian dimension (dim=1)
    mean_vectors = torch.cat([g.mean_vectors for g in gaussians_list], dim=1)
    singular_values = torch.cat([g.singular_values for g in gaussians_list], dim=1)
    quaternions = torch.cat([g.quaternions for g in gaussians_list], dim=1)
    colors = torch.cat([g.colors for g in gaussians_list], dim=1)
    opacities = torch.cat([g.opacities for g in gaussians_list], dim=1)
    
    LOGGER.info(f"Merged {len(gaussians_list)} face(s) into {mean_vectors.shape[1]} gaussians")
    
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def remove_duplicate_gaussians(
    gaussians: Gaussians3D,
    distance_threshold: float = 0.01,
) -> Gaussians3D:
    """Remove duplicate gaussians that are too close together.
    
    This helps clean up overlapping regions from cubemap faces.
    Uses a simple grid-based approach for efficiency.
    
    Args:
        gaussians: Input gaussians.
        distance_threshold: Minimum distance between gaussians.
        
    Returns:
        Gaussians with duplicates removed.
    """
    positions = gaussians.mean_vectors.squeeze(0)  # (N, 3)
    n_gaussians = positions.shape[0]
    
    if n_gaussians < 2:
        return gaussians
    
    # Simple voxel-based deduplication
    # Round positions to grid and keep first occurrence
    grid_positions = torch.round(positions / distance_threshold).long()
    
    # Create unique hash for each voxel
    # Using a simple approach: hash = x * P1 + y * P2 + z * P3
    primes = torch.tensor([73856093, 19349663, 83492791], device=positions.device)
    hashes = (grid_positions * primes).sum(dim=-1)
    
    # Get unique hashes and their indices
    _, unique_indices = torch.unique(hashes, return_inverse=True)
    
    # Keep first occurrence of each unique hash
    mask = torch.zeros(n_gaussians, dtype=torch.bool, device=positions.device)
    seen = set()
    for i in range(n_gaussians):
        h = unique_indices[i].item()
        if h not in seen:
            seen.add(h)
            mask[i] = True
    
    n_removed = n_gaussians - mask.sum().item()
    if n_removed > 0:
        LOGGER.info(f"Removed {n_removed} duplicate gaussians")
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )


def filter_by_opacity(
    gaussians: Gaussians3D,
    min_opacity: float = 0.01,
) -> Gaussians3D:
    """Filter out gaussians with very low opacity.
    
    Args:
        gaussians: Input gaussians.
        min_opacity: Minimum opacity threshold.
        
    Returns:
        Filtered gaussians.
    """
    opacities = gaussians.opacities.squeeze(0)  # (N,)
    mask = opacities > min_opacity
    
    n_removed = (~mask).sum().item()
    if n_removed > 0:
        LOGGER.info(f"Removed {n_removed} low-opacity gaussians")
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )


def normalize_gaussian_scales(
    gaussians: Gaussians3D,
    target_scale: float = 1.0,
) -> Gaussians3D:
    """Normalize gaussian positions to fit within a target scale.
    
    Centers gaussians at origin and scales to target radius.
    
    Args:
        gaussians: Input gaussians.
        target_scale: Target radius for the scene.
        
    Returns:
        Normalized gaussians.
    """
    positions = gaussians.mean_vectors.squeeze(0)
    
    # Compute scene bounds
    center = positions.mean(dim=0)
    positions_centered = positions - center
    
    # Scale to target
    max_dist = positions_centered.norm(dim=-1).max()
    if max_dist > 0:
        scale_factor = target_scale / max_dist
    else:
        scale_factor = 1.0
    
    positions_normalized = positions_centered * scale_factor
    scales_normalized = gaussians.singular_values * scale_factor
    
    LOGGER.info(f"Normalized scene: center offset {center.tolist()}, scale {scale_factor:.4f}")
    
    return Gaussians3D(
        mean_vectors=positions_normalized.unsqueeze(0),
        singular_values=scales_normalized,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def save_panorama_ply(
    gaussians: Gaussians3D,
    output_path: Path,
    image_shape: tuple[int, int] = (2048, 4096),
) -> None:
    """Save panoramic gaussians to a PLY file.
    
    Uses a synthetic focal length and resolution for the metadata.
    
    Args:
        gaussians: The merged panoramic gaussians.
        output_path: Path to save the PLY file.
        image_shape: Virtual image shape (height, width) for metadata.
    """
    # For panoramic output, we use a nominal focal length
    # This is mainly for metadata compatibility
    height, width = image_shape
    f_px = width / (2 * np.pi) * 2  # Approximate for 360° horizontal FOV
    
    save_ply(gaussians, f_px, image_shape, output_path)
    LOGGER.info(f"Saved panoramic gaussians to {output_path}")
