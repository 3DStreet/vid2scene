"""
360 Panorama SfM Pipeline for vid2scene.

This module implements a rig-based approach for processing equirectangular/360 panoramas,
rendering virtual perspective cameras and running SfM with proper rig constraints.

Based on COLMAP's pano_sfm.py example with adaptations for vid2scene's HLOC pipeline.
"""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypeVar, cast
import logging
import shutil

import cv2
import numpy as np
import numpy.typing as npt
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap

logger = logging.getLogger(__name__)

N = TypeVar("N", bound=int)
NDArrayNx2 = np.ndarray[tuple[N, Literal[2]], np.dtype[np.float64]]
NDArray3x1 = np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float64]]


@dataclass
class PanoRenderOptions:
    """Configuration for virtual camera rendering from panoramas."""
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float
    
    @property
    def num_virtual_cameras(self) -> int:
        return self.num_steps_yaw * len(self.pitches_deg)


# Overlapping views configuration for robust matching
PANO_RENDER_OPTIONS = PanoRenderOptions(
    num_steps_yaw=4,
    pitches_deg=(-35.0, 0.0, 35.0),
    hfov_deg=90.0,
    vfov_deg=90.0,
)


def create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    """Create a virtual perspective camera sized proportionally to the panorama."""
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return pycolmap.Camera.create(
        0,
        pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal,
        image_width,
        image_height,
    )


def get_virtual_camera_rays(camera: pycolmap.Camera) -> npt.NDArray[np.floating]:
    """Get normalized ray directions for each pixel in a virtual camera."""
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy: NDArrayNx2 = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm: NDArrayNx2 = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Project rays into equirectangular (spherical) image coordinates."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas (2:1 aspect ratio) are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[npt.NDArray[np.floating]]:
    """Get relative rotation matrices for virtual cameras w.r.t. the panorama center."""
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        # Offset yaw for non-zero pitch to get better coverage
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[npt.NDArray[np.floating]],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    """Create a RigConfig defining the geometry of virtual cameras."""
    rig_cameras = []
    zero_translation = cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64))
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                zero_translation,
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


def detect_ego_object_sam3(
    pano_image_paths: list[Path],
    prompts: list[str] = ["car", "vehicle", "tripod", "selfie stick", "person"],
    score_threshold: float = 0.5,
    downscale_width: int = 1024,  # Downscale for speed, None = no downscale
) -> dict[str, np.ndarray]:
    """
    Detect ego-objects (car, tripod, person, etc.) in equirectangular panoramas using SAM3.
    
    Uses SAM3's text-prompted segmentation to identify the ego-vehicle or camera mount.
    Returns masks where 255 = valid (non-ego), 0 = ego-object (to mask out).
    
    Args:
        pano_image_paths: List of paths to panorama images
        prompts: List of text prompts to try for segmentation
        score_threshold: Minimum confidence score to accept a detection
        downscale_width: Width to downscale to for faster processing (None = no downscale)
        
    Returns:
        Dictionary of {image_name: mask} where 255 = valid, 0 = ego-object
        Returns empty dict if SAM3 is not available
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    masks_dict = {}
    
    try:
        logger.info(f"Loading SAM3 model for ego-object detection...")
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
        logger.info(f"Running SAM3 on {len(pano_image_paths)} frames with prompts: {prompts}")
        
        for pano_path in tqdm(pano_image_paths, desc="SAM3 ego detection"):
            # Load the panorama image
            image = PIL.Image.open(pano_path)
            original_size = image.size  # (width, height)
            
            # Downscale for speed
            if downscale_width and image.width > downscale_width:
                scale_factor = downscale_width / image.width
                new_height = int(image.height * scale_factor)
                image_small = image.resize((downscale_width, new_height), PIL.Image.LANCZOS)
            else:
                image_small = image
                scale_factor = 1.0
            
            small_h, small_w = image_small.height, image_small.width
            
            # Initialize inference state
            inference_state = processor.set_image(image_small)
            
            # Try each prompt and combine results
            combined_mask = np.zeros((small_h, small_w), dtype=np.uint8)
            
            for prompt in prompts:
                try:
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    
                    # Add high-confidence detections to combined mask
                    for mask, score in zip(masks, scores):
                        if score >= score_threshold:
                            # Convert mask to numpy if needed
                            if hasattr(mask, 'cpu'):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]  # Take first channel if 3D
                            combined_mask = np.maximum(combined_mask, (mask_np > 0.5).astype(np.uint8) * 255)
                            
                except Exception as e:
                    continue  # Skip this prompt for this frame
            
            # Upscale mask back to original size if needed
            if scale_factor != 1.0:
                combined_mask = cv2.resize(combined_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Invert: 255 = valid, 0 = ego-object (to mask out)
            ego_mask = 255 - combined_mask
            
            # Store with image name as key
            masks_dict[pano_path.name] = ego_mask
        
        detected_count = sum(1 for m in masks_dict.values() if np.sum(m < 255) > 0)
        logger.info(f"SAM3 detected ego-objects in {detected_count}/{len(pano_image_paths)} frames")
        
        return masks_dict
        
    except Exception as e:
        logger.error(f"SAM3 detection failed: {e}")
        return {}
        return None


def detect_static_regions(
    pano_image_paths: Sequence[Path],
    num_samples: int = 20,
    variance_threshold: float = 100.0,
    min_region_size: int = 1000,
) -> np.ndarray | None:
    """
    Detect static regions in equirectangular panoramas using temporal variance.
    
    Static objects like tripods, car hoods, or selfie sticks appear at the same
    pixel location in every frame and have low temporal variance.
    
    Args:
        pano_image_paths: List of paths to panorama images
        num_samples: Number of frames to sample for variance computation
        variance_threshold: Pixels with variance below this are considered static
        min_region_size: Minimum contiguous region size to consider (filters noise)
        
    Returns:
        Binary mask where 255 = valid (non-static), 0 = static region to mask out
        Returns None if detection fails or no static regions found
    """
    if len(pano_image_paths) < 3:
        logger.warning("Not enough panoramas for static detection (need at least 3)")
        return None
    
    # Sample frames evenly across the sequence
    indices = np.linspace(0, len(pano_image_paths) - 1, min(num_samples, len(pano_image_paths)))
    indices = [int(i) for i in indices]
    
    logger.info(f"Detecting static regions using {len(indices)} sample frames...")
    
    # Load sample frames
    frames = []
    target_size = None
    for idx in indices:
        try:
            img = PIL.Image.open(pano_image_paths[idx])
            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                continue  # Skip frames with different sizes
            frames.append(np.asarray(img).astype(np.float32))
        except Exception as e:
            logger.warning(f"Could not load {pano_image_paths[idx]}: {e}")
            continue
    
    if len(frames) < 3:
        logger.warning("Could not load enough frames for static detection")
        return None
    
    # Stack frames and compute per-pixel variance
    stack = np.stack(frames, axis=0)  # [N, H, W, C]
    
    # Compute variance across time, average across color channels
    variance = np.var(stack, axis=0).mean(axis=-1)  # [H, W]
    
    # Log variance statistics to help with threshold tuning
    logger.info(f"Variance stats: min={variance.min():.1f}, max={variance.max():.1f}, "
                f"mean={variance.mean():.1f}, median={np.median(variance):.1f}")
    logger.info(f"Using variance_threshold={variance_threshold}")
    
    # Create mask: low variance = static (mask out)
    static_mask = variance < variance_threshold
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    static_mask_uint8 = static_mask.astype(np.uint8) * 255
    
    # Close small holes
    static_mask_uint8 = cv2.morphologyEx(static_mask_uint8, cv2.MORPH_CLOSE, kernel)
    # Open to remove small noise
    static_mask_uint8 = cv2.morphologyEx(static_mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Filter by contiguous region size
    contours, _ = cv2.findContours(static_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(static_mask_uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_region_size:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    # Check if we found any significant static regions
    static_pixel_count = np.sum(filtered_mask > 0)
    total_pixels = filtered_mask.shape[0] * filtered_mask.shape[1]
    static_ratio = static_pixel_count / total_pixels
    
    if static_ratio < 0.001:  # Less than 0.1% of image
        logger.info("No significant static regions detected")
        return None
    
    logger.info(f"Detected static regions covering {static_ratio*100:.1f}% of panorama")
    
    # Return inverted mask: 255 = valid, 0 = static (to mask out)
    return 255 - filtered_mask


class PanoProcessor:
    """Processes panoramas into virtual perspective images with optional masks."""
    
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path | None,
        render_options: PanoRenderOptions,
        generate_masks: bool = False,
        ego_masks_dict: dict[str, np.ndarray] | None = None,  # Per-pano ego masks
        training_mask_dir: Path | None = None,
    ):
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir
        self.generate_masks = generate_masks
        self.ego_masks_dict = ego_masks_dict or {}  # {pano_name: mask}
        self.training_mask_dir = training_mask_dir  # For gsplat training (static mask only)

        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)

        # Compute camera optical axis directions for mask generation
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()

        # Initialized on first panorama to avoid recomputing per-image
        self._camera: pycolmap.Camera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray[np.floating] | None = None

    def process(self, pano_name: str) -> None:
        """Process a single panorama into virtual perspective images."""
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_pil_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logger.warning(f"Skipping file {pano_path} as it cannot be read.")
            return

        # Extract GPS EXIF data to preserve in output images
        pano_exif = pano_pil_image.getexif()
        pano_image = np.asarray(pano_pil_image)
        gpsonly_exif = PIL.Image.Exif()
        gps_ifd = pano_exif.get_ifd(PIL.ExifTags.IFD.GPSInfo)
        if gps_ifd:
            gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = gps_ifd

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            logger.warning(f"Skipping {pano_name}: not a 360° panorama (expected 2:1 aspect ratio)")
            return

        with self._lock:
            if self._camera is None:
                # First image - precompute camera and rays
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            else:
                # Verify consistent panorama sizes
                if (pano_width, pano_height) != self._pano_size:
                    logger.warning(f"Skipping {pano_name}: size mismatch (expected {self._pano_size})")
                    return

        # Render each virtual camera view
        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            
            # Remap panorama to perspective view
            image = cv2.remap(
                pano_image,
                x_coords,
                y_coords,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            
            # Generate mask if requested (each pixel belongs to closest virtual camera)
            if self.generate_masks and self.mask_dir:
                closest_camera = np.argmax(
                    rays_in_pano @ self.cam_centers_in_pano.T, -1
                )
                mask = (
                    ((closest_camera == cam_idx) * 255)
                    .astype(np.uint8)
                    .reshape(self._camera.width, self._camera.height)
                    .transpose()
                )
                
                # Apply ego mask if available for this panorama (mask out person, car, tripod, etc.)
                ego_mask = self.ego_masks_dict.get(pano_name)
                if ego_mask is not None:
                    # Sample the ego mask at the same coordinates as the perspective
                    ego_mask_perspective = cv2.remap(
                        ego_mask,
                        x_coords,
                        y_coords,
                        cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_WRAP,
                    )
                    
                    if self.training_mask_dir is not None:
                        training_mask_name = f"{self.rig_config.cameras[cam_idx].image_prefix}{pano_name}"
                        training_mask_path = self.training_mask_dir / training_mask_name
                        training_mask_path.parent.mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(str(training_mask_path), ego_mask_perspective)
                    
                    # AND the masks together: only valid where both are valid
                    mask = np.minimum(mask, ego_mask_perspective)
                
                mask_name = f"{self.rig_config.cameras[cam_idx].image_prefix}{pano_name}"
                mask_path = self.mask_dir / mask_name
                mask_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(mask_path), mask)

            # Save perspective image
            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            
            # PIL expects RGB, and we already have RGB from PIL.Image.open()
            # cv2.remap preserves color order, so no conversion needed
            pil_image = PIL.Image.fromarray(image)
            pil_image.save(image_path, exif=gpsonly_exif if gps_ifd else None)


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    render_options: PanoRenderOptions,
    mask_dir: Path | None = None,
    generate_masks: bool = False,
    ego_masks_dict: dict[str, np.ndarray] | None = None,
    training_mask_dir: Path | None = None,
    max_workers: int | None = None,
) -> pycolmap.RigConfig:
    """Render perspective images from all panoramas in parallel."""
    processor = PanoProcessor(
        pano_image_dir, output_image_dir, mask_dir, render_options, generate_masks, 
        ego_masks_dict, training_mask_dir
    )

    num_panos = len(pano_image_names)
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 2) - 1)
    
    logger.info(f"Rendering {num_panos} panoramas into {render_options.num_virtual_cameras} perspective views each...")
    
    with tqdm(total=num_panos, desc="Rendering perspectives") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    return processor.rig_config


def run_rig_sfm_pipeline(
    output_path: Path,
    image_dir: Path,
    rig_config: pycolmap.RigConfig,
    mask_dir: Path | None = None,
    mapper: str = "glomap",
    kill_check=None,
) -> Path | None:
    """
    Run rig-constrained SfM on pre-rendered virtual perspective images.

    This covers feature extraction, rig configuration, sequential matching,
    and mapping. Shared by the equirectangular pipeline (run_pano_sfm) and
    the direct dual-fisheye pipeline (fisheye_sfm.run_insv_sfm).

    Args:
        output_path: Output directory for the database and sparse reconstruction
        image_dir: Directory containing the rendered perspective images
        rig_config: Rig configuration describing the virtual camera geometry
        mask_dir: Optional directory with feature-extraction masks
        mapper: Reconstruction method ("glomap" for global, "colmap" for incremental)
        kill_check: Optional callback to check if processing should abort

    Returns:
        Path to sparse reconstruction directory, or None if aborted/failed
    """
    from run_command import run_command

    database_path = output_path / "database.db"
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True, parents=True)

    # Clean up old database if exists
    if database_path.exists():
        database_path.unlink()

    # ========== Step 2: Feature extraction ==========
    logger.info("Extracting features...")
    reader_options = pycolmap.ImageReaderOptions()
    if mask_dir and mask_dir.exists():
        reader_options.mask_path = str(mask_dir)

    # Feature extraction options for denser feature extraction
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 32768  # Default is 8192, very high for dense points
    extraction_options.sift.first_octave = -1  # Default is 0, -1 extracts more features at finest scale
    extraction_options.sift.peak_threshold = 0.003  # Default is 0.0067, lower = more features

    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        reader_options=reader_options,
        extraction_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    if kill_check and kill_check():
        logger.info("Job cancelled after feature extraction")
        return None

    # ========== Step 3: Apply rig configuration ==========
    logger.info("Applying rig configuration...")
    with pycolmap.Database.open(str(database_path)) as db:
        pycolmap.apply_rig_config([rig_config], db)

    if kill_check and kill_check():
        logger.info("Job cancelled after applying rig config")
        return None

    # ========== Step 4: Feature matching with COLMAP sequential matching ==========
    logger.info("Matching features using sequential strategy...")
    matching_options = pycolmap.FeatureMatchingOptions()
    # Note: rig_verification and skip_image_pairs_in_same_frame can cause issues
    # with some versions of pycolmap/glomap. Disable for now.
    # matching_options.rig_verification = True
    # matching_options.skip_image_pairs_in_same_frame = True

    seq_options = pycolmap.SequentialPairingOptions()
    seq_options.overlap = 30  # Default is ~10, increase for more matches
    seq_options.loop_detection = False
    pycolmap.match_sequential(
        str(database_path),
        pairing_options=seq_options,
        matching_options=matching_options,
    )

    if kill_check and kill_check():
        logger.info("Job cancelled after matching")
        return None

    # ========== Step 5: Run reconstruction ==========
    if mapper == "glomap":
        # GLOMAP global mapping with rig support
        logger.info("Running GLOMAP global mapping with rig support...")

        success = run_command([
            "glomap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_path),
            "--BundleAdjustment.optimize_rig_poses", "1",     # Optimize rig poses
            "--BundleAdjustment.optimize_intrinsics", "0",    # Don't change virtual cam intrinsics
            "--skip_view_graph_calibration", "1",             # Trust rig config
            "--ba_iteration_num", "5",
            "--skip_pruning", "0",
        ], kill_check=kill_check)

        if not success:
            logger.error("GLOMAP mapping failed!")
            return None

        # Re-register images that may have been pruned
        zero_dir = sparse_path / "0"
        if zero_dir.exists():
            logger.info("Running image registrator to recover pruned images...")
            run_command([
                "colmap", "image_registrator",
                "--database_path", str(database_path),
                "--input_path", str(zero_dir),
                "--output_path", str(zero_dir),
            ], kill_check=kill_check)
    else:
        # COLMAP incremental mapping with fixed rig poses
        logger.info("Running COLMAP incremental mapping with rig constraints...")
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_sensor_from_rig=False,  # Don't modify rig relative poses
            ba_refine_focal_length=False,     # Virtual cams have perfect intrinsics
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )

        recs = pycolmap.incremental_mapping(
            str(database_path), str(image_dir), str(sparse_path), opts
        )

        for idx, rec in recs.items():
            logger.info(f"Reconstruction #{idx}: {rec.summary()}")

        if not recs:
            logger.error("No reconstructions created!")
            return None

    return sparse_path


def run_pano_sfm(
    input_image_path: Path,
    output_path: Path,
    mapper: str = "glomap",
    generate_masks: bool = True,
    detect_static: bool = True,
    kill_check=None,
) -> Path | None:
    """
    Run the full panorama SfM pipeline.

    This function:
    1. (Optional) Detects static objects like tripods using temporal variance
    2. Renders perspective images from equirectangular panoramas
    3. Extracts features using pycolmap
    4. Applies rig configuration for multi-camera constraint
    5. Matches features with COLMAP sequential matching
    6. Runs reconstruction using GLOMAP (global) or COLMAP (incremental)

    Args:
        input_image_path: Directory containing equirectangular panorama images
        output_path: Output directory for rendered images and SfM results
        mapper: Reconstruction method ("glomap" for global, "colmap" for incremental)
        generate_masks: Whether to generate per-pixel masks for feature extraction
        detect_static: Whether to detect and mask static objects (tripod, car hood)
        kill_check: Optional callback to check if processing should abort

    Returns:
        Path to sparse reconstruction directory, or None if aborted/failed
    """
    pycolmap.set_random_seed(0)
    render_options = PANO_RENDER_OPTIONS

    # Setup directories
    image_dir = output_path / "images"
    mask_dir = output_path / "masks" if generate_masks else None
    image_dir.mkdir(exist_ok=True, parents=True)
    if mask_dir:
        mask_dir.mkdir(exist_ok=True, parents=True)

    # Find input panoramas
    pano_image_names = sorted(
        p.relative_to(input_image_path).as_posix()
        for p in input_image_path.rglob("*")
        if not p.is_dir() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
    )
    logger.info(f"Found {len(pano_image_names)} panorama images in {input_image_path}")
    
    if not pano_image_names:
        logger.error("No panorama images found!")
        return None

    if kill_check and kill_check():
        logger.info("Job cancelled before rendering")
        return None

    # ========== Step 0 (Optional): Detect ego-object using SAM3 ==========
    ego_masks_dict = {}  # {pano_name: mask}
    training_mask_dir = None
    if detect_static and generate_masks:
        pano_paths = [input_image_path / name for name in pano_image_names]
        
        # Use SAM3-based detection (text prompts like "car", "tripod", "person")
        ego_masks_dict = detect_ego_object_sam3(pano_paths)
        
        if ego_masks_dict:
            # Create training mask directory for gsplat
            training_mask_dir = output_path / "training_masks"
            training_mask_dir.mkdir(exist_ok=True, parents=True)
            
            # Save first mask for debugging
            first_mask = next(iter(ego_masks_dict.values()))
            static_mask_path = output_path / "static_mask.png"
            cv2.imwrite(str(static_mask_path), first_mask)
            logger.info(f"Saved sample ego mask to {static_mask_path}")
    
    # ========== Step 1: Render perspective images from panoramas ==========
    rig_config = render_perspective_images(
        pano_image_names,
        input_image_path,
        image_dir,
        render_options,
        mask_dir,
        generate_masks,
        ego_masks_dict,
        training_mask_dir,
    )
    
    if kill_check and kill_check():
        logger.info("Job cancelled after rendering")
        return None

    # ========== Steps 2-5: Features, rig config, matching, mapping ==========
    return run_rig_sfm_pipeline(
        output_path=output_path,
        image_dir=image_dir,
        rig_config=rig_config,
        mask_dir=mask_dir,
        mapper=mapper,
        kill_check=kill_check,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="360 Panorama SfM Pipeline")
    parser.add_argument("--input_image_path", type=Path, required=True,
                        help="Directory containing equirectangular panorama images")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--generate_masks", action="store_true",
                        help="Generate per-pixel masks for feature extraction")
    
    args = parser.parse_args()
    
    result = run_pano_sfm(
        input_image_path=args.input_image_path,
        output_path=args.output_path,
        generate_masks=args.generate_masks,
    )
    
    if result:
        logger.info(f"Success! Reconstruction saved to {result}")
    else:
        logger.error("Pipeline failed")
