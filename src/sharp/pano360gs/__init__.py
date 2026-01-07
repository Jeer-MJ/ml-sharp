"""Pano360GS: 360° Panoramic Gaussian Splatting Module.

This module extends Apple's ml-sharp to support equirectangular (360°) panoramic
images, enabling the generation of fully immersive 3D Gaussian Splat scenes.

Key Concepts
------------
- **Equirectangular Projection**: Maps a full sphere onto a 2D rectangle where:
    - Width spans longitude: -π to +π (360°)
    - Height spans latitude: -π/2 to +π/2 (180°)

- **Spherical to Cartesian**: Instead of using camera intrinsics (pinhole model),
  we convert each pixel's angular position directly to 3D Cartesian coordinates.

Components
----------
- `equirectangular`: Core spherical projection mathematics
- `unproject_gaussians_equirectangular`: Main unprojection function for 360° images

Example
-------
>>> from sharp.pano360gs import unproject_gaussians_equirectangular
>>> from sharp.pano360gs import PanoQuality
>>>
>>> # After running standard ml-sharp prediction:
>>> gaussians_3d = unproject_gaussians_equirectangular(
...     gaussians_ndc=predicted_gaussians,
...     image_shape=(width, height),
...     quality=PanoQuality.HIGH
... )

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from .equirectangular import (
    PanoQuality,
    equirectangular_to_xyz,
    unproject_gaussians_equirectangular,
    adjust_gaussian_scales_for_sphere,
    validate_equirectangular_aspect_ratio,
)

__all__ = [
    "PanoQuality",
    "equirectangular_to_xyz",
    "unproject_gaussians_equirectangular",
    "adjust_gaussian_scales_for_sphere",
    "validate_equirectangular_aspect_ratio",
]
