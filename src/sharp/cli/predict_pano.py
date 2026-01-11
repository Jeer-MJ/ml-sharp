"""Contains `sharp predict-pano` CLI implementation for panoramic images.

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
from sharp.pano360gs import (
    CubeFace,
    CubemapConfig,
    extract_all_cubemap_faces,
    filter_by_opacity,
    filter_gaussians_in_face_frustum,
    get_face_focal_length,
    merge_gaussians,
    remove_duplicate_gaussians,
    save_panorama_ply,
    transform_face_gaussians,
)
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
    "--face-size",
    type=int,
    default=1024,
    help="Size of each cubemap face in pixels. Default: 1024",
)
@click.option(
    "--padding",
    type=float,
    default=0.20,
    help="Padding ratio for seamless blending (0.0-0.5). Default: 0.20",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_pano_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    face_size: int,
    padding: float,
    device: str,
    verbose: bool,
):
    """Predict 3D Gaussians from equirectangular panorama images.
    
    This command processes 360° panoramic images by:
    1. Projecting to 6 cubemap faces
    2. Running SHARP prediction on each face
    3. Transforming and merging results into a seamless 360° gaussian scene
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
    
    if width < 4096:
        LOGGER.warning(
            f"Image width {width}px is below recommended 4096px. "
            "Quality may be reduced."
        )
    
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
    
    # Configure cubemap extraction
    config = CubemapConfig(face_size=face_size, padding_ratio=padding)
    
    # Extract cubemap faces
    LOGGER.info("Extracting cubemap faces...")
    faces = extract_all_cubemap_faces(image, config)
    
    # Process each face
    all_gaussians = []
    face_names = ["FRONT", "RIGHT", "BACK", "LEFT", "TOP", "BOTTOM"]
    
    for face in CubeFace:
        LOGGER.info(f"Processing face {face.value}/5: {face_names[face.value]}")
        
        face_image = faces[face]
        face_gaussians = predict_face(
            gaussian_predictor,
            face_image,
            config,
            torch.device(device),
        )
        
        # Filter gaussians within frustum (keep slightly more than 90° for overlap)
        # Use edge fade to create a seamless transition between faces
        face_gaussians = filter_gaussians_in_face_frustum(
            face_gaussians, 
            padding_ratio=-0.05,  # Keep up to 1.05 (5% overlap)
            edge_fade_ratio=0.10,  # Fade from 0.95 to 1.05
        )
        
        # Transform to world coordinates
        world_gaussians = transform_face_gaussians(face_gaussians, face)
        
        n_gaussians = world_gaussians.mean_vectors.shape[1]
        LOGGER.info(f"  -> {n_gaussians} gaussians after filtering")
        
        all_gaussians.append(world_gaussians)
    
    # Merge all faces
    LOGGER.info("Merging all faces...")
    merged = merge_gaussians(all_gaussians)
    
    # Post-processing
    LOGGER.info("Post-processing...")
    merged = filter_by_opacity(merged, min_opacity=0.01)  # Slightly higher threshold
    merged = remove_duplicate_gaussians(merged, distance_threshold=0.008)  # More aggressive dedup
    
    # Save output
    output_path.mkdir(exist_ok=True, parents=True)
    ply_path = output_path / f"{input_path.stem}_pano360.ply"
    
    save_panorama_ply(merged, ply_path, image_shape=(height, width))
    
    n_final = merged.mean_vectors.shape[1]
    LOGGER.info(f"Done! Saved {n_final} gaussians to {ply_path}")


@torch.no_grad()
def predict_face(
    predictor: RGBGaussianPredictor,
    face_image: np.ndarray,
    config: CubemapConfig,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from a single cubemap face.
    
    Args:
        predictor: The SHARP gaussian predictor model.
        face_image: Cubemap face image array (H, W, C).
        config: Cubemap configuration.
        device: Torch device.
        
    Returns:
        Predicted Gaussians3D in face-local coordinates.
    """
    internal_shape = (1536, 1536)
    
    # Prepare image tensor
    image_pt = torch.from_numpy(face_image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    
    # Focal length formula: f = (size/2) / tan(fov/2)
    # For a 90° FOV cubemap face, f = face_size / 2
    # We use the internal core size (without padding) to maintain the 90° scale
    f_px = get_face_focal_length(config.face_size)
    
    # The disparity factor is f_px / width (normalized)
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    
    # Resize for model input
    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    
    # Run inference
    gaussians_ndc = predictor(image_resized, disparity_factor)
    
    # Build intrinsics for unprojection
    # Note: We use -f_px for the Y component because image coordinates have Y pointing
    # downward, while camera/world coordinates have Y pointing upward.
    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, -f_px, height / 2, 0],  # Negative f_y for Y-down image coordinates
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    
    # Scale intrinsics for internal resolution
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    
    # Unproject to metric space
    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=device),
        intrinsics_resized,
        internal_shape,
    )
    
    return gaussians


# For direct script execution
if __name__ == "__main__":
    predict_pano_cli()
