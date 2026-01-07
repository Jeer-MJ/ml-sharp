"""Unit tests for the pano360gs equirectangular projection module.

These tests verify the mathematical correctness of the spherical projection
functions used in Pano360GS.

Run with: pytest tests/test_equirectangular.py -v

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math

import pytest
import torch

from sharp.pano360gs import (
    PanoQuality,
    equirectangular_to_xyz,
    validate_equirectangular_aspect_ratio,
)
from sharp.pano360gs.equirectangular import (
    QUALITY_CONFIGS,
    create_spherical_coordinates,
    spherical_to_cartesian,
    compute_latitude_compensation,
    _quaternion_multiply,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================


class TestValidateEquirectangularAspectRatio:
    """Tests for aspect ratio validation."""
    
    def test_perfect_2_1_ratio(self):
        """Should pass for perfect 2:1 ratio."""
        assert validate_equirectangular_aspect_ratio(4096, 2048) is True
        assert validate_equirectangular_aspect_ratio(2048, 1024) is True
        assert validate_equirectangular_aspect_ratio(1000, 500) is True
    
    def test_slight_deviation_within_tolerance(self):
        """Should pass for small deviations within tolerance."""
        # 10% tolerance by default, so ratio should be between 1.8 and 2.2
        assert validate_equirectangular_aspect_ratio(2100, 1000) is True  # 2.1:1
        assert validate_equirectangular_aspect_ratio(1900, 1000) is True  # 1.9:1
    
    def test_large_deviation_fails(self):
        """Should raise ValueError for non-2:1 ratios."""
        with pytest.raises(ValueError, match="not suitable for equirectangular"):
            validate_equirectangular_aspect_ratio(1920, 1080)  # 16:9
        
        with pytest.raises(ValueError, match="not suitable for equirectangular"):
            validate_equirectangular_aspect_ratio(1024, 1024)  # 1:1


# ==============================================================================
# SPHERICAL COORDINATE TESTS
# ==============================================================================


class TestCreateSphericalCoordinates:
    """Tests for spherical coordinate grid creation."""
    
    def test_output_shapes(self, device):
        """Should return grids of correct shape."""
        width, height = 100, 50
        phi, theta = create_spherical_coordinates(width, height, device)
        
        assert phi.shape == (height, width)
        assert theta.shape == (height, width)
    
    def test_longitude_range(self, device):
        """Longitude (phi) should range from -π to +π."""
        phi, _ = create_spherical_coordinates(360, 180, device)
        
        # First column should be close to -π
        assert phi[0, 0].item() < -math.pi + 0.1
        # Last column should be close to +π
        assert phi[0, -1].item() > math.pi - 0.1
    
    def test_latitude_range(self, device):
        """Latitude (theta) should range from +π/2 to -π/2 (top to bottom)."""
        _, theta = create_spherical_coordinates(360, 180, device)
        
        # First row (top) should be close to +π/2 (north pole)
        assert theta[0, 0].item() > math.pi / 2 - 0.1
        # Last row (bottom) should be close to -π/2 (south pole)
        assert theta[-1, 0].item() < -math.pi / 2 + 0.1
    
    def test_center_point(self, device):
        """Center of image should correspond to φ=0, θ=0."""
        width, height = 100, 50
        phi, theta = create_spherical_coordinates(width, height, device)
        
        center_x = width // 2
        center_y = height // 2
        
        # Allow some tolerance due to pixel centering
        assert abs(phi[center_y, center_x].item()) < 0.1
        assert abs(theta[center_y, center_x].item()) < 0.1


# ==============================================================================
# CARTESIAN CONVERSION TESTS
# ==============================================================================


class TestSphericalToCartesian:
    """Tests for spherical to Cartesian conversion."""
    
    def test_forward_direction(self, device):
        """φ=0, θ=0, r=1 should give (0, 0, 1)."""
        phi = torch.tensor([0.0], device=device)
        theta = torch.tensor([0.0], device=device)
        radius = torch.tensor([1.0], device=device)
        
        xyz = spherical_to_cartesian(phi, theta, radius)
        
        assert xyz.shape == (1, 3)
        assert torch.allclose(xyz[0], torch.tensor([0.0, 0.0, 1.0], device=device), atol=1e-5)
    
    def test_right_direction(self, device):
        """φ=π/2, θ=0, r=1 should give (1, 0, 0)."""
        phi = torch.tensor([math.pi / 2], device=device)
        theta = torch.tensor([0.0], device=device)
        radius = torch.tensor([1.0], device=device)
        
        xyz = spherical_to_cartesian(phi, theta, radius)
        
        assert torch.allclose(xyz[0], torch.tensor([1.0, 0.0, 0.0], device=device), atol=1e-5)
    
    def test_up_direction(self, device):
        """φ=0, θ=π/2, r=1 should give (0, 1, 0)."""
        phi = torch.tensor([0.0], device=device)
        theta = torch.tensor([math.pi / 2], device=device)
        radius = torch.tensor([1.0], device=device)
        
        xyz = spherical_to_cartesian(phi, theta, radius)
        
        assert torch.allclose(xyz[0], torch.tensor([0.0, 1.0, 0.0], device=device), atol=1e-5)
    
    def test_backward_direction(self, device):
        """φ=π, θ=0, r=1 should give (0, 0, -1)."""
        phi = torch.tensor([math.pi], device=device)
        theta = torch.tensor([0.0], device=device)
        radius = torch.tensor([1.0], device=device)
        
        xyz = spherical_to_cartesian(phi, theta, radius)
        
        assert torch.allclose(xyz[0], torch.tensor([0.0, 0.0, -1.0], device=device), atol=1e-5)
    
    def test_unit_sphere(self, device):
        """All points with r=1 should have magnitude 1."""
        phi = torch.linspace(-math.pi, math.pi, 36, device=device)
        theta = torch.linspace(-math.pi/2, math.pi/2, 18, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing='xy')
        radius = torch.ones_like(phi)
        
        xyz = spherical_to_cartesian(phi.flatten(), theta.flatten(), radius.flatten())
        magnitudes = torch.linalg.norm(xyz, dim=-1)
        
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    
    def test_radius_scaling(self, device):
        """Points should scale with radius."""
        phi = torch.tensor([0.0], device=device)
        theta = torch.tensor([0.0], device=device)
        radius = torch.tensor([5.0], device=device)
        
        xyz = spherical_to_cartesian(phi, theta, radius)
        
        assert torch.allclose(xyz[0], torch.tensor([0.0, 0.0, 5.0], device=device), atol=1e-5)


# ==============================================================================
# DEPTH MAP TO XYZ TESTS
# ==============================================================================


class TestEquirectangularToXYZ:
    """Tests for the main depth-to-points conversion."""
    
    def test_output_shape_2d(self, device):
        """Should handle 2D input and return correct shape."""
        depth = torch.ones(180, 360, device=device)
        xyz = equirectangular_to_xyz(depth)
        
        assert xyz.shape == (180, 360, 3)
    
    def test_output_shape_3d(self, device):
        """Should handle 3D (batched) input."""
        depth = torch.ones(2, 180, 360, device=device)
        xyz = equirectangular_to_xyz(depth)
        
        assert xyz.shape == (2, 180, 360, 3)
    
    def test_unit_depth_on_sphere(self, device):
        """With depth=1 everywhere, all points should be on unit sphere."""
        depth = torch.ones(90, 180, device=device)
        xyz = equirectangular_to_xyz(depth)
        
        magnitudes = torch.linalg.norm(xyz, dim=-1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-4)
    
    def test_center_point_forward(self, device):
        """Center of image should point along +Z axis."""
        depth = torch.ones(100, 200, device=device)
        xyz = equirectangular_to_xyz(depth)
        
        # Center pixel
        center_point = xyz[50, 100]
        
        # Should be approximately (0, 0, 1)
        assert abs(center_point[0].item()) < 0.1  # x near 0
        assert abs(center_point[1].item()) < 0.1  # y near 0
        assert center_point[2].item() > 0.9       # z near 1


# ==============================================================================
# LATITUDE COMPENSATION TESTS
# ==============================================================================


class TestLatitudeCompensation:
    """Tests for latitude-based scale compensation."""
    
    def test_equator_no_compensation(self, device):
        """At equator (θ=0), compensation should be ~1."""
        theta = torch.tensor([0.0], device=device)
        comp = compute_latitude_compensation(theta)
        
        assert torch.allclose(comp, torch.ones_like(comp), atol=1e-4)
    
    def test_pole_maximum_compensation(self, device):
        """At poles, compensation should be at maximum (3.0)."""
        theta = torch.tensor([math.pi / 2], device=device)  # North pole
        comp = compute_latitude_compensation(theta)
        
        assert comp.item() == pytest.approx(3.0, rel=0.01)
    
    def test_compensation_increases_toward_poles(self, device):
        """Compensation should increase as we move toward poles."""
        theta = torch.tensor([0.0, math.pi/4, math.pi/3], device=device)
        comp = compute_latitude_compensation(theta)
        
        assert comp[0] < comp[1] < comp[2]


# ==============================================================================
# QUATERNION TESTS
# ==============================================================================


class TestQuaternionMultiply:
    """Tests for quaternion multiplication."""
    
    def test_identity_multiplication(self, device):
        """Multiplying by identity should return original."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        q = torch.tensor([0.707, 0.707, 0.0, 0.0], device=device)  # 90° X rotation
        
        result = _quaternion_multiply(identity, q)
        assert torch.allclose(result, q, atol=1e-4)
        
        result = _quaternion_multiply(q, identity)
        assert torch.allclose(result, q, atol=1e-4)
    
    def test_inverse_multiplication(self, device):
        """q * q^-1 should give identity."""
        q = torch.tensor([0.707, 0.707, 0.0, 0.0], device=device)
        q_inv = torch.tensor([0.707, -0.707, 0.0, 0.0], device=device)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        
        result = _quaternion_multiply(q, q_inv)
        assert torch.allclose(result, identity, atol=1e-3)
    
    def test_batched_multiplication(self, device):
        """Should work with batched inputs."""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]], device=device)
        q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]], device=device)
        
        result = _quaternion_multiply(q1, q2)
        assert result.shape == (2, 4)


# ==============================================================================
# QUALITY CONFIG TESTS
# ==============================================================================


class TestQualityConfigs:
    """Tests for quality preset configurations."""
    
    def test_all_presets_exist(self):
        """All quality presets should have configurations."""
        for quality in PanoQuality:
            assert quality in QUALITY_CONFIGS
    
    def test_resolution_factors_ordered(self):
        """Resolution factors should increase with quality."""
        factors = [QUALITY_CONFIGS[q].resolution_factor for q in 
                   [PanoQuality.LOW, PanoQuality.MEDIUM, PanoQuality.HIGH, PanoQuality.ULTRA]]
        
        assert factors == sorted(factors)
    
    def test_scale_multipliers_ordered(self):
        """Scale multipliers should decrease with quality (smaller splats at higher quality)."""
        multipliers = [QUALITY_CONFIGS[q].scale_multiplier for q in 
                       [PanoQuality.LOW, PanoQuality.MEDIUM, PanoQuality.HIGH, PanoQuality.ULTRA]]
        
        assert multipliers == sorted(multipliers, reverse=True)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_runs(self, device):
        """The complete equirectangular-to-xyz pipeline should run without errors."""
        # Create a synthetic depth map (2:1 ratio)
        width, height = 200, 100
        depth = torch.rand(height, width, device=device) * 10 + 1  # Depth 1-11
        
        # Convert to 3D
        xyz = equirectangular_to_xyz(depth)
        
        # Basic sanity checks
        assert xyz.shape == (height, width, 3)
        assert not torch.isnan(xyz).any()
        assert not torch.isinf(xyz).any()
    
    def test_deterministic_output(self, device):
        """Same input should always produce same output."""
        depth = torch.ones(50, 100, device=device)
        
        xyz1 = equirectangular_to_xyz(depth)
        xyz2 = equirectangular_to_xyz(depth)
        
        assert torch.allclose(xyz1, xyz2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
