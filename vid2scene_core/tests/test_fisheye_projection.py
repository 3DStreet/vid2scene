import cv2
import numpy as np
import pytest

from fisheye_projection import (
    FisheyeLensModel,
    build_remap_grid,
    closest_view_partition,
    derive_crop_size,
    get_lens_local_rotations,
    pinhole_camera_rays,
    pinhole_focal,
)


def make_model(**kwargs) -> FisheyeLensModel:
    defaults = dict(width=400, height=400, fov_deg=200.0)
    defaults.update(kwargs)
    return FisheyeLensModel(**defaults)


class TestFisheyeLensModel:
    def test_center_ray_projects_to_principal_point(self):
        model = make_model()
        uv, valid = model.project_rays(np.array([[0.0, 0.0, 1.0]]))
        assert valid[0]
        np.testing.assert_allclose(uv[0], [200.0, 200.0], atol=1e-9)

    def test_90deg_off_axis_radius(self):
        # Equidistant model: r = focal * theta, so a 90 degree ray lands at
        # radius (90 / 100) * image_circle_radius for a 200 degree lens.
        model = make_model()
        uv, valid = model.project_rays(np.array([[1.0, 0.0, 0.0]]))
        assert valid[0]
        expected_radius = 200.0 * (90.0 / 100.0)
        np.testing.assert_allclose(uv[0], [200.0 + expected_radius, 200.0], atol=1e-9)

    def test_rays_beyond_fov_are_invalid(self):
        model = make_model(fov_deg=190.0)
        # 100 degrees off-axis is outside a 190 degree (95 half-angle) lens
        theta = np.deg2rad(100.0)
        ray = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
        _, valid = model.project_rays(ray)
        assert not valid[0]

    def test_backward_ray_is_invalid(self):
        model = make_model()
        _, valid = model.project_rays(np.array([[0.0, 0.0, -1.0]]))
        assert not valid[0]

    @pytest.mark.parametrize(
        "distortion",
        [dict(), dict(k1=-0.02, k2=0.004, k3=-0.0008, k4=0.0001)],
    )
    def test_project_unproject_roundtrip(self, distortion):
        model = make_model(**distortion)
        rng = np.random.default_rng(0)
        # Random rays within the lens FOV
        theta = rng.uniform(0.0, np.deg2rad(95.0), size=500)
        phi = rng.uniform(-np.pi, np.pi, size=500)
        rays = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )
        uv, valid = model.project_rays(rays)
        assert valid.all()
        rays_back = model.unproject_points(uv)
        np.testing.assert_allclose(rays_back, rays, atol=1e-8)

    def test_defaults_resolve_from_frame_size(self):
        model = FisheyeLensModel(width=2880, height=2880, fov_deg=200.0)
        assert model.cx == 1440.0
        assert model.cy == 1440.0
        assert model.image_circle_radius == 1440.0
        np.testing.assert_allclose(
            model.focal_px_per_rad, 1440.0 / np.deg2rad(100.0)
        )


class TestPinholeRays:
    def test_center_pixel_points_forward(self):
        rays = pinhole_camera_rays(100, 75.0).reshape(100, 100, 3)
        center = (rays[49, 49] + rays[50, 50] + rays[49, 50] + rays[50, 49]) / 4
        center /= np.linalg.norm(center)
        np.testing.assert_allclose(center, [0.0, 0.0, 1.0], atol=1e-9)

    def test_horizontal_fov_matches(self):
        size, fov = 200, 75.0
        rays = pinhole_camera_rays(size, fov).reshape(size, size, 3)
        left = rays[size // 2, 0]
        right = rays[size // 2, -1]
        angle = np.rad2deg(np.arccos(np.clip(np.dot(left, right), -1, 1)))
        # Edge pixel centers sit half a pixel inside the exact FOV
        assert abs(angle - fov) < 1.0

    def test_pinhole_focal(self):
        np.testing.assert_allclose(pinhole_focal(100, 90.0), 50.0)


class TestViewGrid:
    def test_default_grid_covers_nadir_and_zenith(self):
        # The headline fix over the equirectangular path: down-pitched views
        # must actually contain the nadir (straight down, +y in the y-down
        # camera convention).
        rotations = get_lens_local_rotations((-60.0, 0.0, 60.0), (-60.0, 0.0, 60.0))
        axes = np.stack([r[2, :] for r in rotations])  # view axes in lens frame
        crop_half_fov = 75.0 / 2
        for pole in ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]):
            angles = np.rad2deg(np.arccos(np.clip(axes @ pole, -1, 1)))
            assert angles.min() < crop_half_fov - 5.0
        # The nadir is 90 degrees off the lens axis: inside a 200 degree lens
        model = make_model()
        _, valid = model.project_rays(np.array([[0.0, 1.0, 0.0]]))
        assert valid[0]

    def test_identity_rotation_looks_along_axis(self):
        rotations = get_lens_local_rotations((0.0,), (0.0,))
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)

    def test_negative_pitch_looks_down(self):
        rotations = get_lens_local_rotations((0.0,), (-60.0,))
        axis = rotations[0][2, :]
        assert axis[1] > 0.5  # +y is down


class TestRemapGrid:
    def test_center_view_fully_valid_and_samples_circle(self):
        model = make_model()
        # Green inside the image circle, black outside
        fisheye = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(fisheye, (200, 200), 200, (0, 255, 0), -1)

        map_x, map_y, valid = build_remap_grid(model, np.eye(3), 75.0, 128)
        assert valid.all()
        crop = cv2.remap(fisheye, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        assert (crop[:, :, 1] == 255).all()

    def test_view_past_lens_fov_is_partially_invalid_and_black(self):
        model = make_model()
        fisheye = np.full((400, 400, 3), 255, dtype=np.uint8)
        # Axis 90 degrees off: bottom of a 75 degree crop reaches ~127 degrees,
        # well past the 100 degree half-FOV of the lens.
        rotation = get_lens_local_rotations((0.0,), (-90.0,))[0]
        map_x, map_y, valid = build_remap_grid(model, rotation, 75.0, 128)
        assert valid.any() and not valid.all()
        crop = cv2.remap(fisheye, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        assert (crop[~valid] == 0).all()

    def test_derive_crop_size(self):
        model = make_model(width=2880, height=2880)
        size = derive_crop_size(model, 75.0, max_crop_size=1600)
        # focal ~825 px/rad => ~1266 px for a 75 degree crop
        assert 1200 < size < 1350
        assert size % 2 == 0
        assert derive_crop_size(model, 75.0, max_crop_size=1024) == 1024


class TestPartition:
    def test_rays_assigned_to_most_aligned_axis(self):
        axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rays = np.array(
            [[0.1, 0.0, 0.9], [0.9, 0.1, 0.1], [0.1, 0.9, 0.0]]
        )
        np.testing.assert_array_equal(closest_view_partition(rays, axes), [0, 1, 2])
