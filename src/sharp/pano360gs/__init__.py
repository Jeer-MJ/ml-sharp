"""Panorama to 3D Gaussian Splatting module.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from .cubemap import (
    CubeFace,
    CubemapConfig,
    extract_all_cubemap_faces,
    extract_cubemap_face,
    get_face_focal_length,
    get_face_rotation_matrix,
)
from .fusion import (
    filter_by_opacity,
    merge_gaussians,
    normalize_gaussian_scales,
    remove_duplicate_gaussians,
    save_panorama_ply,
)
from .transforms import (
    filter_gaussians_by_depth,
    filter_gaussians_in_face_frustum,
    reposition_gaussians_for_panorama,
    transform_face_gaussians,
)
from .spherical import (
    filter_by_depth_percentile,
    wrap_gaussians_to_sphere,
)

__all__ = [
    # Cubemap
    "CubeFace",
    "CubemapConfig",
    "extract_all_cubemap_faces",
    "extract_cubemap_face",
    "get_face_focal_length",
    "get_face_rotation_matrix",
    # Transforms
    "transform_face_gaussians",
    "filter_gaussians_by_depth",
    "filter_gaussians_in_face_frustum",
    "reposition_gaussians_for_panorama",
    # Fusion
    "merge_gaussians",
    "remove_duplicate_gaussians",
    "filter_by_opacity",
    "normalize_gaussian_scales",
    "save_panorama_ply",
    # Spherical (single-pass approach)
    "wrap_gaussians_to_sphere",
    "filter_by_depth_percentile",
]
