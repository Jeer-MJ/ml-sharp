"""Contains `sharp predict-pano-sphere` CLI - Single pass panorama to gaussians.

The key insight: SHARP generates gaussians from any flat image.
An equirectangular panorama IS a flat image that represents a sphere.
We just pass the whole panorama through SHARP and wrap the result onto a sphere.

NO CUBEMAP. NO CUTTING. NO SEAMS. ONE PASS.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.pano360gs.spherical import (
    filter_by_depth_percentile,
    wrap_gaussians_to_sphere,
)
from sharp.pano360gs.fusion import filter_by_opacity, save_panorama_ply
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an equirectangular panorama image (2:1 ratio, min 4096x2048).",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted panoramic Gaussians.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_pano_sphere_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    device: str,
    verbose: bool,
):
    """Predict 3D Gaussians from panorama in ONE PASS - no seams!
    
    This command:
    1. Takes the full equirectangular panorama
    2. Runs SHARP prediction in a single pass
    3. Wraps the resulting gaussians onto a sphere
    
    No cubemap cutting, no seams, no merging needed.
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    
    # Validate input file
    extensions = io.get_supported_image_extensions()
    if input_path.suffix.lower() not in extensions:
        LOGGER.error(f"Unsupported image format: {input_path.suffix}")
        return
    
    # Setup device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info(f"Using device: {device}")
    
    # Load panorama image
    LOGGER.info(f"Loading panorama: {input_path}")
    image, _, _ = io.load_rgb(input_path, auto_rotate=False)
    height, width = image.shape[:2]
    
    # Validate aspect ratio
    aspect_ratio = width / height
    if abs(aspect_ratio - 2.0) > 0.05:
        LOGGER.error(
            f"Invalid aspect ratio {aspect_ratio:.2f}. "
            "Equirectangular panoramas must be 2:1 (e.g., 4096x2048)."
        )
        return
    
    LOGGER.info(f"Panorama size: {width}x{height}")
    
    # Load model
    if checkpoint_path is None:
        LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True)
    
    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)
    
    # Single pass prediction
    LOGGER.info("Running SHARP prediction (single pass)...")
    gaussians_flat = predict_panorama_flat(
        gaussian_predictor,
        image,
        torch.device(device),
    )
    
    n_flat = gaussians_flat.mean_vectors.shape[1]
    LOGGER.info(f"Generated {n_flat} flat gaussians")
    
    # Filter depth outliers before wrapping
    LOGGER.info("Filtering depth outliers...")
    gaussians_flat = filter_by_depth_percentile(gaussians_flat, 1.0, 99.0)
    
    # Wrap onto sphere
    # Note: internal shape is always 1536x1536 (square), but we pass
    # the original panorama aspect ratio for correct UV mapping
    LOGGER.info("Wrapping gaussians onto sphere...")
    internal_shape = get_internal_shape()
    gaussians_sphere = wrap_gaussians_to_sphere(
        gaussians_flat,
        original_aspect_ratio=width / height,  # Should be ~2.0 for equirectangular
    )
    
    # Filter low opacity
    LOGGER.info("Post-processing...")
    gaussians_sphere = filter_by_opacity(gaussians_sphere, min_opacity=0.01)
    
    # Save output
    output_path.mkdir(exist_ok=True, parents=True)
    ply_path = output_path / f"{input_path.stem}_sphere360.ply"
    
    save_panorama_ply(gaussians_sphere, ply_path, image_shape=(height, width))
    
    n_final = gaussians_sphere.mean_vectors.shape[1]
    LOGGER.info(f"Done! Saved {n_final} gaussians to {ply_path}")


def get_internal_shape() -> tuple[int, int]:
    """Get SHARP's required internal processing shape.
    
    SHARP REQUIRES square 1536x1536 input. The panorama will be resized
    to fit this, and the UV mapping will compensate for the distortion.
    """
    return (1536, 1536)


@torch.no_grad()
def predict_panorama_flat(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    device: torch.device,
) -> Gaussians3D:
    """Predict flat gaussians from panorama image.
    
    SHARP requires 1536x1536 square input. We resize the panorama to fit,
    and the spherical wrapping will handle the UV mapping correctly.
    
    Args:
        predictor: The SHARP gaussian predictor model.
        image: Panorama image array (H, W, C).
        device: Torch device.
        
    Returns:
        Flat Gaussians3D (not yet wrapped to sphere).
    """
    internal_shape = get_internal_shape()  # Always (1536, 1536)
    
    LOGGER.info(f"Internal processing shape: {internal_shape[0]}x{internal_shape[1]}")
    
    # Prepare image tensor
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    
    # Resize to square 1536x1536 (will stretch vertically for 2:1 panorama)
    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    
    # For the disparity factor, use a standard value
    # This affects depth scale but structure is preserved
    # Using 0.5 as a reasonable default for panoramas
    disparity_factor = torch.tensor([0.5]).float().to(device)
    
    LOGGER.info("Running inference...")
    gaussians_ndc = predictor(image_resized, disparity_factor)
    
    # Build intrinsics for unprojection
    # Use square intrinsics matching the 1536x1536 input
    f_px = internal_shape[0] / 2  # Focal length for ~90° FOV
    cx = internal_shape[0] / 2
    cy = internal_shape[1] / 2
    
    intrinsics = torch.tensor(
        [
            [f_px, 0, cx, 0],
            [0, f_px, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    
    # Unproject to metric space
    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=device),
        intrinsics,
        internal_shape,
    )
    
    return gaussians


# For direct script execution
if __name__ == "__main__":
    predict_pano_sphere_cli()
