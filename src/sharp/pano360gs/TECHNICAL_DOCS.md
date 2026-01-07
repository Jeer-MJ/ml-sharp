# Pano360GS Technical Documentation

## Module Overview

**Pano360GS** extends Apple's ml-sharp model to support **equirectangular (360°) panoramic images**, enabling the generation of fully immersive 3D Gaussian Splat scenes from a single panoramic photograph.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PANO360GS PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │ Equirect.    │     │  ml-sharp    │     │ Spherical Unprojection   │   │
│   │ Panorama     │────▶│  Predictor   │────▶│ (pano360gs module)       │   │
│   │ (2:1 ratio)  │     │  (standard)  │     │                          │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│         │                     │                         │                   │
│         │                     ▼                         ▼                   │
│         │              Gaussians (NDC)           Gaussians (3D World)       │
│         │              - positions               - spherical positions     │
│         │              - scales                  - adjusted scales          │
│         │              - rotations               - sphere-aligned rots      │
│         │              - colors                  - colors (unchanged)       │
│         │              - opacities               - opacities (unchanged)    │
│         │                                                                   │
│         └─────────────────────────────────────────────────────────────────  │
│                              Output: .PLY file (3DGS format)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### 1. Equirectangular Projection

An equirectangular image is a 2D representation of a complete sphere, commonly used for 360° photography and VR content.

```
    Image Coordinates                    Spherical Coordinates
    ┌────────────────────┐              
    │ u=0            u=W │               φ = -π ←─────────→ φ = +π
    │ ┌────────────────┐ │                     (longitude)
    │ │                │ │ v=0    ═══▶   θ = +π/2 (north pole)
    │ │                │ │ 
    │ │     Center     │ │ v=H/2  ═══▶   θ = 0 (equator)
    │ │                │ │
    │ │                │ │ v=H    ═══▶   θ = -π/2 (south pole)
    │ └────────────────┘ │                     (latitude)
    └────────────────────┘
```

**Conversion formulas:**

$$\phi = \frac{u}{W} \times 2\pi - \pi \quad \text{(longitude)}$$

$$\theta = \left(\frac{1}{2} - \frac{v}{H}\right) \times \pi \quad \text{(latitude)}$$

### 2. Spherical to Cartesian Transformation

We use the **Y-up convention** (compatible with OpenCV, Unity, Unreal):

$$x = d \times \cos(\theta) \times \sin(\phi)$$

$$y = d \times \sin(\theta)$$

$$z = d \times \cos(\theta) \times \cos(\phi)$$

where $d$ is the depth at each pixel.

```
                 +Y (up)
                  │
                  │    θ (latitude)
                  │   ╱
                  │  ╱
                  │ ╱
                  │╱───────────── +X (right)
                 ╱│
                ╱ │
               ╱  │
              ╱   │
           +Z (forward)
             
        φ = 0 → looking along +Z
        φ = π/2 → looking along +X
        θ = 0 → equator (horizon)
        θ = π/2 → looking up (+Y)
```

### 3. Latitude Compensation

In equirectangular projection, pixels near the poles cover **less solid angle** than pixels at the equator. Without compensation, polar regions would appear sparse.

**Solid angle per pixel:**
$$\Omega \propto \cos(\theta)$$

**Compensation factor:**
$$k_{lat} = \frac{1}{\max(\cos(\theta), \epsilon)}$$

Clamped to a maximum of 3.0 to prevent extreme scaling at exact poles.

### 4. Depth-Based Scale Adjustment

To maintain visual coverage, Gaussian splat sizes are scaled proportionally to depth:

$$\text{scale}_{final} = \text{scale}_{predicted} \times d \times k_{depth} \times k_{lat} \times k_{quality}$$

where:
- $k_{depth}$: depth scale factor (quality-dependent)
- $k_{lat}$: latitude compensation
- $k_{quality}$: overall quality multiplier

---

## Quality Presets

| Preset | Resolution | Scale Mult. | Depth Factor | Approx. Splats* |
|--------|------------|-------------|--------------|-----------------|
| LOW    | 25%        | 4.0         | 0.004        | ~500K           |
| MEDIUM | 50%        | 2.0         | 0.002        | ~2M             |
| HIGH   | 75%        | 1.33        | 0.0015       | ~5M             |
| ULTRA  | 100%       | 1.0         | 0.001        | ~8.4M           |

*Based on 4K (4096×2048) input image

---

## API Reference

### `unproject_gaussians_equirectangular()`

Main entry point for converting ml-sharp predictions to 360° representation.

```python
from sharp.pano360gs import unproject_gaussians_equirectangular, PanoQuality

gaussians_3d = unproject_gaussians_equirectangular(
    gaussians_ndc=predicted_gaussians,  # From ml-sharp predictor
    image_shape=(4096, 2048),           # (width, height)
    quality=PanoQuality.HIGH            # Quality preset
)
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `gaussians_ndc` | `Gaussians3D` | Predicted Gaussians in NDC space |
| `image_shape` | `tuple[int, int]` | Image dimensions (width, height) |
| `quality` | `PanoQuality` | Quality preset (LOW, MEDIUM, HIGH, ULTRA) |

**Returns:** `Gaussians3D` with spherically-projected positions and adjusted scales.

---

### `equirectangular_to_xyz()`

Core function for spherical projection of depth maps.

```python
from sharp.pano360gs import equirectangular_to_xyz

# depth_map: torch.Tensor shape (H, W) or (B, H, W)
point_cloud = equirectangular_to_xyz(depth_map)
# Returns: torch.Tensor shape (*input_shape, 3)
```

---

### `validate_equirectangular_aspect_ratio()`

Validates that input image has correct 2:1 aspect ratio.

```python
from sharp.pano360gs import validate_equirectangular_aspect_ratio

# Raises ValueError if aspect ratio is not approximately 2:1
validate_equirectangular_aspect_ratio(width=4096, height=2048)
```

---

## Quaternion Adjustment for Spherical Surfaces

Each Gaussian must be **rotated to be tangent to the sphere** at its position. The predicted quaternions assume a flat image plane; we apply additional rotations:

1. **Longitude rotation** (around Y-axis by φ)
2. **Latitude rotation** (around local X-axis by θ)

This ensures the Gaussian's local XY-plane is tangent to the sphere, with local Z pointing toward the center.

```
        Before                              After
    (flat projection)                 (sphere tangent)
    
         Z                                   ╲
         │                                    ╲ Z (toward center)
         │                                     ╲
    ─────┼───── X                     ─────────●─────────
         │                                    ╱ XY plane
         │                                   ╱  (tangent)
         Y
```

---

## Integration with ml-sharp

### Replacement for `unproject_gaussians()`

The standard ml-sharp pipeline uses:
```python
from sharp.utils.gaussians import unproject_gaussians

gaussians = unproject_gaussians(
    gaussians_ndc, extrinsics, intrinsics, image_shape
)
```

For 360° images, replace with:
```python
from sharp.pano360gs import unproject_gaussians_equirectangular

gaussians = unproject_gaussians_equirectangular(
    gaussians_ndc, image_shape, quality=PanoQuality.HIGH
)
```

Note: Extrinsics and intrinsics are not needed for equirectangular projection, as the geometry is defined purely by the spherical coordinate system.

---

## File Structure

```
src/sharp/pano360gs/
├── __init__.py           # Package exports
└── equirectangular.py    # Core projection implementation
    ├── PanoQuality       # Quality preset enum
    ├── QualityConfig     # Quality configuration tuple
    ├── QUALITY_CONFIGS   # Preset configurations
    ├── validate_equirectangular_aspect_ratio()
    ├── create_spherical_coordinates()
    ├── spherical_to_cartesian()
    ├── equirectangular_to_xyz()
    ├── compute_latitude_compensation()
    ├── adjust_gaussian_scales_for_sphere()
    ├── unproject_gaussians_equirectangular()
    ├── _adjust_quaternions_for_sphere()
    └── _quaternion_multiply()
```

---

## Usage Example

```python
import torch
from sharp.models import create_predictor, PredictorParams
from sharp.pano360gs import unproject_gaussians_equirectangular, PanoQuality
from sharp.utils.gaussians import save_ply
from sharp.utils.io import load_rgb

# 1. Load equirectangular panorama
image, _, f_px = load_rgb("panorama_360.jpg")
height, width = image.shape[:2]

# 2. Validate aspect ratio (should be 2:1)
from sharp.pano360gs import validate_equirectangular_aspect_ratio
validate_equirectangular_aspect_ratio(width, height)

# 3. Create predictor and run inference
predictor = create_predictor(PredictorParams())
predictor.load_state_dict(torch.load("checkpoint.pt"))
predictor.eval()

# 4. Preprocess and predict (standard ml-sharp)
image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
disparity_factor = torch.tensor([f_px / width])
gaussians_ndc = predictor(image_tensor.unsqueeze(0), disparity_factor)

# 5. Apply spherical unprojection (PANO360GS)
gaussians_3d = unproject_gaussians_equirectangular(
    gaussians_ndc,
    image_shape=(width, height),
    quality=PanoQuality.HIGH
)

# 6. Save to PLY
save_ply(gaussians_3d, f_px, (height, width), "output_360.ply")
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Aspect ratio not suitable" error | Input is not 2:1 | Use proper equirectangular image |
| Holes at poles | Insufficient scale compensation | Increase `depth_scale_factor` or use higher quality |
| Inverted scene | Y-axis convention mismatch | Check if renderer uses Y-up or Z-up |
| Splats too large/small | Quality preset mismatch | Adjust `scale_multiplier` in config |

### Debugging Tips

```python
import logging
logging.getLogger("sharp.pano360gs").setLevel(logging.DEBUG)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0   | 2025-01 | Initial implementation |
