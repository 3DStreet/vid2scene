"""
Fisheye projection math for direct dual-fisheye (.insv) processing.

Pure numpy/scipy module (no pycolmap dependency) so the projection math is
independently unit-testable. Used by fisheye_sfm.py to render virtual pinhole
views directly from Insta360 fisheye sensor frames, skipping the
equirectangular intermediate that resamples (and badly stretches) the image
near the poles — i.e. exactly at the ground and sky.

Conventions match pano_sfm.py / COLMAP:
- Camera frame: x right, y down, z forward (optical axis).
- Continuous pixel coordinates place the center of the top-left pixel at
  (0.5, 0.5). cv2.remap maps are shifted by -0.5 accordingly.
- Rotations are "cam_from_X" matrices applied to row vectors as
  ``ray_in_X = ray_in_cam @ cam_from_X_r``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


@dataclass
class FisheyeLensModel:
    """Equidistant fisheye model with optional Kannala-Brandt distortion terms.

    The projection is r = focal * (theta + k1*theta^3 + k2*theta^5 +
    k3*theta^7 + k4*theta^9) where theta is the angle from the optical axis
    and r is the radial distance from the principal point in pixels.

    Defaults model an idealized Insta360-style lens: principal point at the
    image center, image circle inscribed in the (square) frame, and the
    nominal full lens FOV mapped linearly onto the image circle. Real lenses
    deviate from this; per-unit calibration can be supplied via the k terms,
    focal, and principal point (see docs/insv_fisheye.md).
    """

    width: int
    height: int
    fov_deg: float = 200.0
    focal_px_per_rad: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    image_circle_radius: float | None = None

    def __post_init__(self):
        if self.cx is None:
            self.cx = self.width / 2
        if self.cy is None:
            self.cy = self.height / 2
        if self.image_circle_radius is None:
            self.image_circle_radius = min(self.width, self.height) / 2
        if self.focal_px_per_rad is None:
            self.focal_px_per_rad = self.image_circle_radius / np.deg2rad(self.fov_deg / 2)

    def theta_to_radius(self, theta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Radial distance in pixels for angle theta (radians) from the optical axis."""
        t2 = theta * theta
        poly = 1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4)))
        return self.focal_px_per_rad * theta * poly

    def radius_to_theta(self, radius: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Invert theta_to_radius with Newton iterations (exact when k1..k4 are 0)."""
        radius = np.asarray(radius, dtype=np.float64)
        theta = radius / self.focal_px_per_rad
        for _ in range(20):
            t2 = theta * theta
            poly = 1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4)))
            d_poly = 1.0 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))
            residual = self.focal_px_per_rad * theta * poly - radius
            theta = theta - residual / (self.focal_px_per_rad * d_poly)
        return theta

    def project_rays(
        self, rays: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
        """Project rays (N, 3) in the lens frame to pixel coordinates.

        Returns (uv, valid) where uv is (N, 2) in COLMAP pixel convention
        (top-left pixel center at (0.5, 0.5)) and valid marks rays inside
        the lens FOV, the image circle, and the image bounds.
        """
        x, y, z = np.moveaxis(np.asarray(rays, dtype=np.float64), -1, 0)
        r_xy = np.hypot(x, y)
        theta = np.arctan2(r_xy, z)
        phi = np.arctan2(y, x)
        radius = self.theta_to_radius(theta)
        u = self.cx + radius * np.cos(phi)
        v = self.cy + radius * np.sin(phi)
        valid = (
            (theta <= np.deg2rad(self.fov_deg / 2))
            & (radius <= self.image_circle_radius)
            & (u >= 0.0)
            & (u <= self.width)
            & (v >= 0.0)
            & (v <= self.height)
        )
        return np.stack([u, v], axis=-1), valid

    def unproject_points(self, uv: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Unproject pixel coordinates (N, 2) to unit rays (N, 3) in the lens frame."""
        uv = np.asarray(uv, dtype=np.float64)
        dx = uv[..., 0] - self.cx
        dy = uv[..., 1] - self.cy
        radius = np.hypot(dx, dy)
        theta = self.radius_to_theta(radius)
        phi = np.arctan2(dy, dx)
        sin_theta = np.sin(theta)
        rays = np.stack(
            [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1
        )
        return rays


@dataclass
class MeiLensModel:
    """Unified omnidirectional (MEI/UCM) camera model with extended distortion.

    This is the model Insta360 uses for its per-unit factory calibration
    (parsed from the .insv trailer / .pb sidecar by insv_calibration.py). A
    ray is normalized onto the unit sphere, perspective-projected from a
    point ``xi`` above the sphere center onto the normalized plane
    ``m = (x, y) / (z + xi)``, distorted with radial (k1..k4 on r^2..r^8),
    tangential (p1, p2) and thin-prism (s1..s4) terms, and scaled by the
    pinhole-like intrinsics fx/fy/cx/cy.

    All parameters are expected in actual stream resolution and COLMAP pixel
    convention (top-left pixel center at (0.5, 0.5)). ``fov_deg`` bounds the
    valid cone: with xi >= 1 the projection itself accepts rays from (nearly)
    every direction, so validity must come from the known lens coverage.
    """

    width: int
    height: int
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    fov_deg: float = 200.0

    @property
    def focal_px_per_rad(self) -> float:
        """Angular resolution at the image center, for crop-size derivation.

        Near the optical axis r = fx * sin(theta) / (cos(theta) + xi)
        ~= fx * theta / (1 + xi).
        """
        return self.fx / (1.0 + self.xi)

    def project_rays(
        self, rays: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
        """Project rays (N, 3) in the lens frame to pixel coordinates.

        Same interface as FisheyeLensModel.project_rays: returns (uv, valid)
        with uv in COLMAP pixel convention and valid marking rays inside the
        lens FOV cone and the image bounds.
        """
        rays = np.asarray(rays, dtype=np.float64)
        norm = np.linalg.norm(rays, axis=-1)
        norm = np.maximum(norm, 1e-12)
        x, y, z = np.moveaxis(rays, -1, 0) / norm

        denom = z + self.xi
        theta = np.arctan2(np.hypot(x, y), z)
        valid = (denom > 1e-6) & (theta <= np.deg2rad(self.fov_deg / 2))

        safe_denom = np.where(valid, denom, 1.0)
        mx = x / safe_denom
        my = y / safe_denom

        r2 = mx * mx + my * my
        r4 = r2 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r2 * r4 + self.k4 * r4 * r4
        xd = (
            mx * radial
            + 2 * self.p1 * mx * my
            + self.p2 * (r2 + 2 * mx * mx)
            + self.s1 * r2
            + self.s2 * r4
        )
        yd = (
            my * radial
            + self.p1 * (r2 + 2 * my * my)
            + 2 * self.p2 * mx * my
            + self.s3 * r2
            + self.s4 * r4
        )

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        valid &= (u >= 0.0) & (u <= self.width) & (v >= 0.0) & (v <= self.height)
        return np.stack([u, v], axis=-1), valid


@dataclass
class FisheyeRigRenderOptions:
    """Virtual pinhole view layout rendered from each fisheye lens.

    Angles are lens-local: yaw/pitch (0, 0) looks along the lens optical
    axis. The default 3x3 grid at +/-60 degrees with 75 degree square crops
    covers each lens out to ~97 degrees off-axis, so the down-pitched views
    of both lenses see the nadir (ground directly under the camera) at full
    sensor resolution — the region the equirectangular path samples worst.
    """

    yaws_deg: Sequence[float] = (-60.0, 0.0, 60.0)
    pitches_deg: Sequence[float] = (-60.0, 0.0, 60.0)
    crop_fov_deg: float = 75.0
    crop_size: int | None = None  # None = derived from source resolution
    max_crop_size: int = 1600

    @property
    def num_views_per_lens(self) -> int:
        return len(self.yaws_deg) * len(self.pitches_deg)


def pinhole_focal(image_size: int, fov_deg: float) -> float:
    """Focal length in pixels for a square pinhole image with the given FOV."""
    return image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))


def pinhole_camera_rays(image_size: int, fov_deg: float) -> npt.NDArray[np.floating]:
    """Unit ray directions (image_size**2, 3) for a square pinhole camera.

    Rays are ordered row-major (y, then x) so a reshape to
    (image_size, image_size) is directly indexable as [row, col].
    """
    focal = pinhole_focal(image_size, fov_deg)
    coords = (np.arange(image_size, dtype=np.float64) + 0.5 - image_size / 2) / focal
    xx, yy = np.meshgrid(coords, coords)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def get_lens_local_rotations(
    yaws_deg: Sequence[float], pitches_deg: Sequence[float]
) -> list[npt.NDArray[np.floating]]:
    """cam_from_lens rotation matrices for a lens-local yaw/pitch view grid.

    Same Euler convention as pano_sfm.get_virtual_rotations: positive pitch
    looks up, positive yaw looks right (with the y-down camera frame).
    """
    rotations = []
    for pitch_deg in pitches_deg:
        for yaw_deg in yaws_deg:
            rotations.append(
                Rotation.from_euler("XY", [-pitch_deg, -yaw_deg], degrees=True).as_matrix()
            )
    return rotations


def get_lens_from_rig_rotations(
    lens1_yaw_deg: float = 180.0,
    lens1_roll_deg: float = 0.0,
    mount_corrections: Sequence[tuple[float, float, float]] | None = None,
) -> list[npt.NDArray[np.floating]]:
    """lens_from_rig rotations for the two back-to-back lenses.

    Lens 0 defines the rig frame; lens 1 nominally points the opposite way
    (180 degree yaw). The optional roll accounts for sensors mounted rotated
    relative to each other. mount_corrections are per-lens factory-measured
    (yaw, pitch, roll) deviations in degrees, applied in each lens' local
    frame on top of its nominal orientation (matching how insv-stitch
    composes the Insta360 factory extrinsics).
    """
    corrections = mount_corrections or ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Column-vector lens->rig corrections; transposed below into this file's
    # row-vector convention (ray_in_rig = ray_in_lens @ lens_from_rig).
    correction_cols = [
        Rotation.from_euler("YXZ", list(correction), degrees=True).as_matrix()
        for correction in corrections
    ]
    lens1_base = Rotation.from_euler(
        "ZY", [lens1_roll_deg, lens1_yaw_deg], degrees=True
    ).as_matrix()
    lens0_from_rig = correction_cols[0].T
    lens1_from_rig = correction_cols[1].T @ lens1_base
    return [lens0_from_rig, lens1_from_rig]


def derive_crop_size(lens_model: FisheyeLensModel, crop_fov_deg: float, max_crop_size: int) -> int:
    """Crop size whose center angular resolution matches the fisheye source."""
    size = lens_model.focal_px_per_rad * 2 * np.tan(np.deg2rad(crop_fov_deg) / 2)
    size = int(np.clip(round(size), 256, max_crop_size))
    return size + (size % 2)


def build_remap_grid(
    lens_model: FisheyeLensModel,
    cam_from_lens_r: npt.NDArray[np.floating],
    crop_fov_deg: float,
    crop_size: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Build cv2.remap maps rendering a pinhole view out of a fisheye image.

    Returns (map_x, map_y, valid), each (crop_size, crop_size). Invalid
    pixels (outside the lens FOV / image circle) map to (-1, -1) so
    cv2.remap with BORDER_CONSTANT renders them black.
    """
    rays_in_cam = pinhole_camera_rays(crop_size, crop_fov_deg)
    rays_in_lens = rays_in_cam @ cam_from_lens_r
    uv, valid = lens_model.project_rays(rays_in_lens)
    uv = uv - 0.5  # COLMAP to OpenCV pixel origin
    uv[~valid] = -1.0
    map_x = uv[:, 0].astype(np.float32).reshape(crop_size, crop_size)
    map_y = uv[:, 1].astype(np.float32).reshape(crop_size, crop_size)
    return map_x, map_y, valid.reshape(crop_size, crop_size)


def closest_view_partition(
    rays: npt.NDArray[np.floating], view_axes: npt.NDArray[np.floating]
) -> npt.NDArray[np.intp]:
    """Assign each ray (N, 3) to the view axis (V, 3) it is most aligned with.

    Used to build feature-extraction masks so overlapping virtual views don't
    contribute duplicate SIFT features of the same physical point, mirroring
    the closest-camera partition in pano_sfm.PanoProcessor.
    """
    return np.argmax(rays @ np.asarray(view_axes).T, axis=-1)
