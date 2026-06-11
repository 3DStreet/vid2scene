"""
Direct dual-fisheye (.insv) SfM pipeline for vid2scene.

Renders virtual pinhole views straight out of the two Insta360 fisheye sensor
streams and reconstructs them with the same rig-constrained SfM used for
equirectangular panoramas (pano_sfm.run_rig_sfm_pipeline).

Why: the equirectangular path (Insta360 Studio stitch -> pano_sfm) resamples
the sphere into a 2:1 image whose poles — the ground under the camera and the
sky — are maximally stretched, and the virtual rig only reaches -35 degrees of
pitch. Cropping pinhole views directly from the fisheye sources keeps native
sensor pixels everywhere, and the default view grid points views down past the
nadir, so the ground is covered by up to 6 views per frame pair instead of 0.

The lens model is an idealized equidistant fisheye by default (see
fisheye_projection.FisheyeLensModel); per-unit calibration can be supplied as
a JSON file. Factory calibration / IMU parsing from the .insv trailer is a
follow-up (see docs/insv_fisheye.md).
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap

import insv_extract
import pano_sfm
from fisheye_projection import (
    FisheyeLensModel,
    FisheyeRigRenderOptions,
    build_remap_grid,
    closest_view_partition,
    derive_crop_size,
    get_lens_local_rotations,
    pinhole_camera_rays,
    pinhole_focal,
)

logger = logging.getLogger(__name__)

INSV_RENDER_OPTIONS = FisheyeRigRenderOptions()


def get_lens_from_rig_rotations(
    lens1_yaw_deg: float = 180.0, lens1_roll_deg: float = 0.0
) -> list[npt.NDArray[np.floating]]:
    """lens_from_rig rotations for the two back-to-back lenses.

    Lens 0 defines the rig frame; lens 1 nominally points the opposite way
    (180 degree yaw). The optional roll accounts for sensors mounted rotated
    relative to each other.
    """
    lens0_from_rig = np.eye(3)
    lens1_from_rig = Rotation.from_euler(
        "ZY", [lens1_roll_deg, lens1_yaw_deg], degrees=True
    ).as_matrix()
    return [lens0_from_rig, lens1_from_rig]


def create_fisheye_rig_config(
    image_prefixes: Sequence[str],
    cams_from_rig_rotation: Sequence[npt.NDArray[np.floating]],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    """Create a RigConfig over all virtual views of both lenses."""
    rig_cameras = []
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    for idx, (prefix, cam_from_rig_rotation) in enumerate(
        zip(image_prefixes, cams_from_rig_rotation)
    ):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_rig_rotation @ cams_from_rig_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                cast(np.ndarray, zero_translation),
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=prefix,
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class FisheyeFrameProcessor:
    """Renders paired fisheye frames into virtual pinhole views with masks."""

    def __init__(
        self,
        lens_frame_dirs: Sequence[Path],
        output_image_dir: Path,
        render_options: FisheyeRigRenderOptions,
        mask_dir: Path | None = None,
        training_mask_dir: Path | None = None,
        lens_model_overrides: Sequence[dict] | None = None,
        lens1_yaw_deg: float = 180.0,
        lens1_roll_deg: float = 0.0,
    ):
        self.lens_frame_dirs = [Path(d) for d in lens_frame_dirs]
        self.output_image_dir = Path(output_image_dir)
        self.render_options = render_options
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.training_mask_dir = Path(training_mask_dir) if training_mask_dir else None
        self.lens_model_overrides = list(lens_model_overrides or [{}, {}])

        num_lenses = len(self.lens_frame_dirs)
        if num_lenses != 2:
            raise ValueError(f"Expected 2 lens directories, got {num_lenses}")

        self.cams_from_lens = get_lens_local_rotations(
            render_options.yaws_deg, render_options.pitches_deg
        )
        self.lens_from_rig = get_lens_from_rig_rotations(lens1_yaw_deg, lens1_roll_deg)

        # One virtual view per (lens, lens-local rotation)
        self.view_lens_idx: list[int] = []
        self.view_prefixes: list[str] = []
        self.views_cam_from_lens: list[npt.NDArray[np.floating]] = []
        self.views_cam_from_rig: list[npt.NDArray[np.floating]] = []
        for lens_idx in range(num_lenses):
            for cam_idx, cam_from_lens in enumerate(self.cams_from_lens):
                self.view_lens_idx.append(lens_idx)
                self.view_prefixes.append(f"lens{lens_idx}_cam{cam_idx}/")
                self.views_cam_from_lens.append(cam_from_lens)
                self.views_cam_from_rig.append(cam_from_lens @ self.lens_from_rig[lens_idx])

        self.rig_config = create_fisheye_rig_config(
            self.view_prefixes, self.views_cam_from_rig
        )

        self._lock = Lock()
        self._initialized = False
        self._camera: pycolmap.Camera | None = None
        self.lens_models: list[FisheyeLensModel] | None = None
        self.crop_size: int | None = None
        self._maps: list[tuple[np.ndarray, np.ndarray]] = []
        self._sfm_mask_png: list[bytes] = []
        self._training_mask_png: list[bytes] = []

    def _initialize(self, lens_images: Sequence[np.ndarray]) -> None:
        """Build lens models, remap grids, and static masks from the first frame pair."""
        self.lens_models = []
        for lens_idx, image in enumerate(lens_images):
            height, width = image.shape[:2]
            overrides = dict(self.lens_model_overrides[lens_idx])
            self.lens_models.append(FisheyeLensModel(width=width, height=height, **overrides))
            logger.info(
                f"Lens {lens_idx}: {width}x{height}, fov={self.lens_models[lens_idx].fov_deg} deg, "
                f"focal={self.lens_models[lens_idx].focal_px_per_rad:.1f} px/rad"
            )

        options = self.render_options
        self.crop_size = options.crop_size or derive_crop_size(
            self.lens_models[0], options.crop_fov_deg, options.max_crop_size
        )
        logger.info(
            f"Rendering {len(self.view_prefixes)} virtual views per frame pair "
            f"({self.crop_size}x{self.crop_size}, {options.crop_fov_deg} deg FOV)"
        )

        self._camera = pycolmap.Camera.create(
            0,
            pycolmap.CameraModelId.SIMPLE_PINHOLE,
            pinhole_focal(self.crop_size, options.crop_fov_deg),
            self.crop_size,
            self.crop_size,
        )
        for rig_camera in self.rig_config.cameras:
            rig_camera.camera = self._camera

        # View axes in the rig frame, for the closest-view feature partition
        view_axes_in_rig = np.stack([r[2, :] for r in self.views_cam_from_rig])
        rays_in_cam = pinhole_camera_rays(self.crop_size, options.crop_fov_deg)

        for view_idx, lens_idx in enumerate(self.view_lens_idx):
            map_x, map_y, valid = build_remap_grid(
                self.lens_models[lens_idx],
                self.views_cam_from_lens[view_idx],
                options.crop_fov_deg,
                self.crop_size,
            )
            self._maps.append((map_x, map_y))

            # SfM mask: valid fisheye pixels owned by this view in the partition
            rays_in_rig = rays_in_cam @ self.views_cam_from_rig[view_idx]
            closest_view = closest_view_partition(rays_in_rig, view_axes_in_rig)
            owned = (closest_view == view_idx).reshape(self.crop_size, self.crop_size)
            sfm_mask = ((owned & valid) * np.uint8(255))
            self._sfm_mask_png.append(_encode_png(sfm_mask))

            # Training mask: every valid pixel (gsplat may supervise overlaps)
            training_mask = (valid * np.uint8(255))
            self._training_mask_png.append(_encode_png(training_mask))

        self._initialized = True

    def process(self, frame_name: str) -> None:
        """Render one paired fisheye frame into all virtual views."""
        lens_images = []
        for lens_dir in self.lens_frame_dirs:
            image = cv2.imread(str(lens_dir / frame_name), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning(f"Skipping frame {frame_name}: unreadable in {lens_dir}")
                return
            lens_images.append(image)

        with self._lock:
            if not self._initialized:
                self._initialize(lens_images)

        for lens_idx, image in enumerate(lens_images):
            model = self.lens_models[lens_idx]
            if image.shape[1] != model.width or image.shape[0] != model.height:
                logger.warning(
                    f"Skipping frame {frame_name}: lens {lens_idx} size mismatch "
                    f"({image.shape[1]}x{image.shape[0]} vs {model.width}x{model.height})"
                )
                return

        for view_idx, prefix in enumerate(self.view_prefixes):
            map_x, map_y = self._maps[view_idx]
            view_image = cv2.remap(
                lens_images[self.view_lens_idx[view_idx]],
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            image_path = self.output_image_dir / prefix / frame_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(image_path), view_image)

            if self.mask_dir is not None:
                # COLMAP looks for masks named "<image_name>.png"
                mask_path = self.mask_dir / prefix / f"{frame_name}.png"
                mask_path.parent.mkdir(exist_ok=True, parents=True)
                mask_path.write_bytes(self._sfm_mask_png[view_idx])

            if self.training_mask_dir is not None:
                training_mask_path = self.training_mask_dir / prefix / frame_name
                training_mask_path.parent.mkdir(exist_ok=True, parents=True)
                training_mask_path.write_bytes(self._training_mask_png[view_idx])


def _encode_png(mask: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("Failed to encode mask PNG")
    return buffer.tobytes()


def render_fisheye_views(
    processor: FisheyeFrameProcessor,
    frame_names: Sequence[str],
    max_workers: int | None = None,
) -> None:
    """Render all paired frames in parallel."""
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 2) - 1)

    # Process the first frame alone so initialization (remap grids, masks)
    # happens once instead of racing in every worker.
    processor.process(frame_names[0])

    with tqdm(total=len(frame_names), desc="Rendering fisheye views") as pbar:
        pbar.update(1)
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, frame_name)
                for frame_name in frame_names[1:]
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)


def load_lens_calibration(calibration_path) -> dict:
    """Load the optional lens calibration JSON (see docs/insv_fisheye.md)."""
    with open(calibration_path) as f:
        calibration = json.load(f)
    logger.info(f"Loaded .insv lens calibration from {calibration_path}")
    return calibration


def build_lens_model_overrides(
    calibration: dict | None, lens_fov_deg: float | None
) -> list[dict]:
    """Merge global and per-lens calibration into FisheyeLensModel kwargs."""
    calibration = calibration or {}
    shared_keys = (
        "fov_deg", "focal_px_per_rad", "cx", "cy",
        "k1", "k2", "k3", "k4", "image_circle_radius",
    )
    shared = {k: calibration[k] for k in shared_keys if k in calibration}
    if lens_fov_deg is not None:
        shared["fov_deg"] = lens_fov_deg

    overrides = []
    per_lens = calibration.get("lenses", [{}, {}])
    for lens_idx in range(2):
        lens_overrides = dict(shared)
        if lens_idx < len(per_lens):
            lens_overrides.update(
                {k: per_lens[lens_idx][k] for k in shared_keys if k in per_lens[lens_idx]}
            )
        overrides.append(lens_overrides)
    return overrides


def run_insv_sfm(
    insv_path,
    output_path: Path,
    target_framecount: int = 90,
    mapper: str = "colmap",
    generate_masks: bool = True,
    render_options: FisheyeRigRenderOptions = INSV_RENDER_OPTIONS,
    lens_fov_deg: float | None = None,
    calibration_path=None,
    kill_check=None,
) -> Path | None:
    """
    Run the direct .insv dual-fisheye SfM pipeline.

    This function:
    1. Demuxes the .insv into paired per-lens fisheye frames (ffmpeg)
    2. Renders virtual pinhole views directly from each fisheye frame
    3. Runs the shared rig-constrained SfM pipeline (pano_sfm.run_rig_sfm_pipeline)

    Args:
        insv_path: Path to the .insv recording (either file of a two-file pair)
        output_path: Output directory for rendered images and SfM results
        target_framecount: Number of paired fisheye frames to extract
        mapper: Reconstruction method ("glomap" or "colmap")
        generate_masks: Generate feature-extraction masks (validity + view partition)
        render_options: Virtual view layout configuration
        lens_fov_deg: Override the assumed lens FOV (default 200)
        calibration_path: Optional lens calibration JSON
        kill_check: Optional callback to check if processing should abort

    Returns:
        Path to sparse reconstruction directory, or None if aborted/failed
    """
    pycolmap.set_random_seed(0)
    output_path = Path(output_path)
    image_dir = output_path / "images"
    mask_dir = output_path / "masks" if generate_masks else None
    training_mask_dir = output_path / "training_masks" if generate_masks else None
    image_dir.mkdir(exist_ok=True, parents=True)
    if mask_dir:
        mask_dir.mkdir(exist_ok=True, parents=True)
    if training_mask_dir:
        training_mask_dir.mkdir(exist_ok=True, parents=True)

    metadata = insv_extract.summarize_insv_metadata(
        insv_extract.read_insv_trailer_records(insv_path)
    )
    if metadata.get("record_ids"):
        logger.info(f".insv trailer records: {metadata['record_ids']}")
    if metadata.get("file_info_strings"):
        logger.info(f".insv file info: {metadata['file_info_strings']}")

    calibration = load_lens_calibration(calibration_path) if calibration_path else None
    lens_model_overrides = build_lens_model_overrides(calibration, lens_fov_deg)
    lens1_yaw_deg = (calibration or {}).get("lens1_yaw_deg", 180.0)
    lens1_roll_deg = (calibration or {}).get("lens1_roll_deg", 0.0)

    if kill_check and kill_check():
        logger.info("Job cancelled before .insv extraction")
        return None

    # ========== Step 1: Demux .insv into paired fisheye frames ==========
    frames_temp_dir = tempfile.mkdtemp(prefix="insv_fisheye_")
    try:
        lens_dirs = insv_extract.extract_dual_fisheye_frames(
            insv_path, frames_temp_dir, target_framecount
        )

        if kill_check and kill_check():
            logger.info("Job cancelled after .insv extraction")
            return None

        frame_names = sorted(
            set(os.listdir(lens_dirs[0])) & set(os.listdir(lens_dirs[1]))
        )
        if not frame_names:
            logger.error("No paired fisheye frames extracted!")
            return None

        # ========== Step 2: Render virtual pinhole views ==========
        processor = FisheyeFrameProcessor(
            lens_frame_dirs=[Path(d) for d in lens_dirs],
            output_image_dir=image_dir,
            render_options=render_options,
            mask_dir=mask_dir,
            training_mask_dir=training_mask_dir,
            lens_model_overrides=lens_model_overrides,
            lens1_yaw_deg=lens1_yaw_deg,
            lens1_roll_deg=lens1_roll_deg,
        )
        render_fisheye_views(processor, frame_names)
    finally:
        shutil.rmtree(frames_temp_dir, ignore_errors=True)

    if kill_check and kill_check():
        logger.info("Job cancelled after rendering fisheye views")
        return None

    # ========== Step 3: Shared rig SfM (features, rig, matching, mapping) ==========
    return pano_sfm.run_rig_sfm_pipeline(
        output_path=output_path,
        image_dir=image_dir,
        rig_config=processor.rig_config,
        mask_dir=mask_dir,
        mapper=mapper,
        kill_check=kill_check,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Direct .insv dual-fisheye SfM pipeline")
    parser.add_argument("--insv_path", type=Path, required=True,
                        help="Path to the .insv recording")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--target_framecount", type=int, default=90,
                        help="Number of paired fisheye frames to extract")
    parser.add_argument("--mapper", choices=["glomap", "colmap"], default="colmap",
                        help="Reconstruction method")
    parser.add_argument("--lens_fov", type=float, default=None,
                        help="Assumed full lens FOV in degrees (default 200)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Optional lens calibration JSON")

    args = parser.parse_args()

    result = run_insv_sfm(
        insv_path=args.insv_path,
        output_path=args.output_path,
        target_framecount=args.target_framecount,
        mapper=args.mapper,
        lens_fov_deg=args.lens_fov,
        calibration_path=args.calibration,
    )

    if result:
        logger.info(f"Success! Reconstruction saved to {result}")
    else:
        logger.error("Pipeline failed")
