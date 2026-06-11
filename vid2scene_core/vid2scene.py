import os
import argparse
import shutil
import extract_frames
import generate_sfm_hloc
import logging
import tempfile
import equirectangular_to_perspective
import pano_sfm
import fisheye_sfm
from create_background_sphere import add_background_sphere
from run_command import run_command
from preview_data_handler import PreviewDataHandler
from watchdog.observers import Observer
from apply_pilgram import apply_filters_to_directory, list_available_filters
from vggt_to_colmap import run_vggt_to_colmap
from quest_to_colmap import run_quest_to_colmap

logger = logging.getLogger(__name__)

SPZ_TO_PLY_EXECUTABLE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "spz", "build_native", "bin", "spz_convert")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Wrapper script to run frame extraction, SfM generation, and Gsplat."
    )
    parser.add_argument(
        "--video_path",
        help="Path to the video file. Required if --image_dir and --sfm_dir are not provided.",
    )
    parser.add_argument(
        "--image_dir",
        help="Use this directory for images instead of extracting frames from video.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to store the SfM model, Gsplat output, and images subdirectory if not already provided.",
    )
    parser.add_argument(
        "--sfm_dir",
        help="Use this directory as the SfM model. Assumes images subdirectory already exists.",
    )
    parser.add_argument(
        "--target_framecount",
        type=int,
        help="Target number of frames to extract.",
        default=600,
    )
    parser.add_argument(
        "--mock",
        help="For debugging. Don't actually run anything -- just output a dummy .ply file in the output directory.",
    )
    parser.add_argument(
        "--training_max_num_gaussians",
        type=int,
        help="Maximum number of Gaussians to use in Gsplat training.",
        default=1_000_000,
    )
    parser.add_argument(
        "--training_num_steps",
        type=int,
        help="Number of training steps to use in Gsplat.",
        default=30_000,
    )
    parser.add_argument(
        "--remove_background",
        help="Remove background from images before running SfM.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--equirectangular",
        help="Use equirectangular/360 video input. Uses rig-based SfM for better results.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--insv_fisheye",
        help="Process an Insta360 .insv recording directly from its dual fisheye "
             "streams (skips the equirectangular intermediate; better ground/sky "
             "detail). Enabled automatically when video_path ends in .insv.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--insv_lens_fov",
        type=float,
        help="Assumed full FOV of each fisheye lens in degrees for --insv_fisheye "
             "(default 200).",
        default=None,
    )
    parser.add_argument(
        "--insv_calibration",
        help="Path to a lens calibration JSON for --insv_fisheye "
             "(overrides the factory calibration; see docs/insv_fisheye.md).",
        default=None,
    )
    parser.add_argument(
        "--insv_no_factory_calibration",
        help="Ignore Insta360's per-unit factory lens calibration embedded in "
             "the .insv recording and use the idealized lens model instead.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_background_sphere",
        help="Add a fibonacci sphere for the background.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--apply_pilgram_filter",
        help="Apply a Pilgram filter to the images.",
        choices=list_available_filters(),
    )
    parser.add_argument(
        "--reconstruction_method",
        help="Reconstruction method to use for SfM.",
        choices=['glomap', 'colmap', 'vggt', 'quest'],
        default='glomap',
    )
    parser.add_argument(
        "--quest_project_dir",
        help="Path to Quest project directory (required when using 'quest' reconstruction method).",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        help="Maximum long-edge resolution in pixels for extracted frames.",
        default=1920,
    )
    parser.add_argument(
        "--stop_after_sfm",
        help="Stop after SfM artifacts are ready (skip gsplat training); resume later with --sfm_dir.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--apriltag_size",
        type=float,
        help="Physical size of AprilTag in meters (measure outer black border). "
             "Example: 0.15 for 15cm tags. Enables automatic scale calibration.",
    )
    return parser.parse_args()


def check_requirements(video_path, image_dir, sfm_dir, reconstruction_method='glomap', quest_project_dir=None):
    # For quest method, quest_project_dir is required instead of video/image/sfm
    if reconstruction_method == 'quest':
        if not quest_project_dir:
            logger.error("quest_project_dir must be provided when using 'quest' reconstruction method.")
            raise ValueError("quest_project_dir must be provided when using 'quest' reconstruction method.")
    elif not video_path and not image_dir and not sfm_dir:
        logger.error("Either video_path, image_dir, or sfm_dir must be provided.")
        raise ValueError("Either video_path, image_dir, or sfm_dir must be provided.")

    gsplat_script = os.getenv("GSPLAT_SCRIPT")
    if not gsplat_script:
        logger.error("GSPLAT_SCRIPT environment variable is not set.")
        raise ValueError("GSPLAT_SCRIPT environment variable is not set.")
    return gsplat_script


def prepare_directories(output_dir, sfm_dir):
    preview_data_path = os.path.join(output_dir, "preview_data")
    os.makedirs(preview_data_path, exist_ok=True)
    if sfm_dir:
        sfm_output_dir = sfm_dir
        images_output_dir = os.path.join(sfm_output_dir, "images")
        if not os.path.exists(images_output_dir):
            logger.error(
                f"The images subdirectory does not exist inside the provided SfM model directory: {sfm_output_dir}"
            )
            raise ValueError(f"The images subdirectory does not exist inside the provided SfM model directory: {sfm_output_dir}")
        logger.info(f"Using existing SfM model directory and images: {sfm_output_dir}")
    else:
        sfm_output_dir = os.path.join(output_dir, "sfm_output")
        os.makedirs(sfm_output_dir, exist_ok=True)
        images_output_dir = os.path.join(sfm_output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)

    result_dir = os.path.join(output_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    save_ply_path = os.path.join(output_dir, "ply", "splat.ply")
    os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)

    return sfm_output_dir, images_output_dir, result_dir, save_ply_path, preview_data_path


def handle_images(video_path, image_dir, images_output_dir, target_framecount, equirectangular=False, max_resolution=1920):
    if image_dir:
        logger.info(
            f"Copying images from provided directory: {image_dir} to {images_output_dir}"
        )
        shutil.copytree(image_dir, images_output_dir, dirs_exist_ok=True)
    else:

        logger.info(f"Extracting frames from video to {images_output_dir}...")
        # If equirectangular is true, we don't want to downscale the images, because we'll be splitting them later
        extract_frames.extract_frames(video_path, images_output_dir, target_framecount, downscale=not equirectangular, max_resolution=max_resolution)

    if equirectangular:
        # Make temp directory for planar projections
        temp_dir = tempfile.mkdtemp()
        # Split the images into 4 quadrants
        equirectangular_to_perspective.process_equirectangular_images(images_output_dir, temp_dir, crop_bottom=0.2, samples_per_im=14)
        
        # Now, copy and replace the images_output_dir with the temp_dir
        shutil.rmtree(images_output_dir)
        shutil.copytree(temp_dir, images_output_dir)
        # Remove the temp directory
        shutil.rmtree(temp_dir)


def align_sfm_orientation(sfm_output_dir, kill_check=None):
    """
    Run COLMAP's model_orientation_aligner to fix the up direction of the reconstruction.
    """
    sparse_dir = os.path.join(sfm_output_dir, "sparse", "0")
    aligned_dir = os.path.join(sfm_output_dir, "sparse", "0_aligned")
    os.makedirs(aligned_dir, exist_ok=True)
    
    if not os.path.exists(sparse_dir):
        logger.warning(f"Sparse directory not found: {sparse_dir}, skipping orientation alignment")
        return
    
    # Check if we should abort
    if kill_check and kill_check():
        logger.info("Job was deleted before orientation alignment, stopping")
        return
    
    # Run COLMAP model_orientation_aligner
    align_command = [
        "colmap", "model_orientation_aligner",
        "--input_path", sparse_dir,
        "--output_path", aligned_dir,
        "--image_path", os.path.join(sfm_output_dir, "images")
    ]
    
    logger.info("Running COLMAP orientation alignment to fix up direction...")
    logger.info(f"Command: {' '.join(align_command)}")
    
    success = run_command(align_command, log_info_output=True, kill_check=kill_check)
    
    if not success:
        logger.warning("Orientation alignment failed or was terminated")
        return
    
    # Check if alignment succeeded
    if not os.path.exists(aligned_dir):
        logger.warning("Aligned directory was not created, keeping original orientation")
        return
    
    # Backup original and replace with aligned version
    backup_dir = os.path.join(sfm_output_dir, "sparse", "0_before_alignment")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    logger.info(f"Backing up original to {backup_dir}")
    shutil.move(sparse_dir, backup_dir)
    
    logger.info(f"Replacing with aligned reconstruction")
    shutil.move(aligned_dir, sparse_dir)
    
    logger.info("Orientation alignment completed successfully")


def apply_apriltag_scale_calibration(sfm_output_dir, images_output_dir, apriltag_size_meters, apriltag_debug=False, kill_check=None):
    """Apply AprilTag-based scale calibration to the reconstruction."""
    if apriltag_size_meters is None or apriltag_size_meters <= 0:
        logger.info("No AprilTag calibration requested (--apriltag_size not specified)")
        return
    
    # Check if we should abort
    if kill_check and kill_check():
        logger.info("Job was deleted before AprilTag calibration, stopping")
        return
    
    try:
        from pathlib import Path
        from apriltag_calibration import calibrate_scale_with_apriltags
        
        logger.info(f"Calibrating scale using AprilTags (size: {apriltag_size_meters}m)...")
        
        scale_factor = calibrate_scale_with_apriltags(
            image_dir=Path(images_output_dir),
            sfm_dir=Path(sfm_output_dir),
            tag_size_meters=apriltag_size_meters,
            tag_family="tagStandard41h12",
            debug=apriltag_debug
        )
        
        logger.info(f"✓ Applied scale factor: {scale_factor:.4f}")
        
    except ImportError as e:
        logger.error("AprilTag calibration requires 'pupil-apriltags' package")
        logger.error("Install with: pip install pupil-apriltags")
        raise
    except Exception as e:
        logger.error(f"AprilTag calibration failed: {e}")
        raise


def select_best_sparse_subdir(sfm_output_dir):
    """
    Find the sparse subdirectory with the largest points3D.bin file and make it the 0 directory.
    This ensures gsplat uses the best reconstruction when COLMAP/GLOMAP produces multiple models.
    """
    sparse_dir = os.path.join(sfm_output_dir, "sparse")
    if not os.path.exists(sparse_dir):
        logger.warning(f"Sparse directory not found: {sparse_dir}, skipping best subdir selection")
        return
    
    sparse_subdirs = [os.path.join(sparse_dir, d) for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
    points3d_bin_files = [os.path.join(d, "points3D.bin") for d in sparse_subdirs if os.path.exists(os.path.join(d, "points3D.bin"))]
    
    if not points3d_bin_files:
        logger.warning(f"No points3D.bin files found in {sparse_dir}")
        return
    
    logger.info(f"Found {len(points3d_bin_files)} points3D.bin files in {sparse_dir}")
    largest_points3d_bin_file = max(points3d_bin_files, key=os.path.getsize)
    best_sparse_subdir = os.path.dirname(largest_points3d_bin_file)
    logger.info(f"Using {best_sparse_subdir} as the best sparse subdirectory")
    
    # Copy the best_sparse_subdir to 0 if it's not already there
    zero_dir = os.path.join(sparse_dir, "0")
    if best_sparse_subdir != zero_dir:
        logger.info(f"Copying {best_sparse_subdir} to {zero_dir}")
        # Move old 0 out of the way
        if os.path.exists(zero_dir):
            old_zero_dir = os.path.join(sparse_dir, "old_0")
            logger.info(f"Moving old zero_dir out of the way to {old_zero_dir}")
            if os.path.exists(old_zero_dir):
                shutil.rmtree(old_zero_dir)
            shutil.move(zero_dir, old_zero_dir)
        shutil.copytree(best_sparse_subdir, zero_dir)


def generate_sfm_point_cloud(
    sfm_dir, images_output_dir, sfm_output_dir, kill_check = None, reconstruction_method = 'glomap',
    apriltag_size_meters = None, quest_project_dir = None
):
    if not sfm_dir:
        if reconstruction_method == 'quest':
            logger.info("Generating 3D SfM point cloud using Quest reconstruction...")
            if not quest_project_dir:
                raise ValueError("quest_project_dir must be provided when using 'quest' reconstruction method.")
            sparse_dir = run_quest_to_colmap(
                quest_project_dir=quest_project_dir,
                output_dir=sfm_output_dir,
                kill_check=kill_check,
                use_colored_pointcloud=True,
                use_optimized_color_dataset=False,
                interval=1
            )
            if sparse_dir is None:
                return  # Processing was terminated
        elif reconstruction_method == 'vggt':
            logger.info("Generating 3D SfM point cloud using VGGT...")
            sparse_dir = run_vggt_to_colmap(
                images_output_dir, sfm_output_dir, kill_check=kill_check, mask_black_bg=False, mask_white_bg=False, mask_sky=False
            )
            if sparse_dir is None:
                return  # Processing was terminated
        else:
            logger.info("Generating 3D SfM point cloud...")
            generate_sfm_hloc.run_sfm(images_output_dir, sfm_output_dir, kill_check=kill_check, reconstruction_method=reconstruction_method)
    
    sparse_dir = os.path.join(sfm_output_dir, "sparse")
    if not os.path.exists(sparse_dir):
        raise ValueError(f"The sparse subdirectory does not exist inside the provided SfM model directory: {sfm_output_dir}")
    
    # For quest and vggt, the sparse/0 directory is already created, so we can skip the best subdir selection
    if reconstruction_method in ['quest', 'vggt']:
        zero_dir = os.path.join(sparse_dir, "0")
        if os.path.exists(zero_dir):
            logger.info(f"Using VGGT-generated sparse directory: {zero_dir}")
        else:
            raise ValueError(f"VGGT did not create the expected sparse/0 directory: {zero_dir}")
    else:
        # Select the best sparse subdirectory (largest points3D.bin) and make it the 0 directory
        select_best_sparse_subdir(sfm_output_dir)
    
    align_sfm_orientation(sfm_output_dir, kill_check=kill_check)



def run_gsplat(gsplat_script, sfm_output_dir, result_dir, save_ply_path, training_max_num_gaussians, training_num_steps,
               preview_data_path, on_new_preview=None, kill_check=None, normalize=True, mcmc_refine_every=None):

    steps_scaler = training_num_steps / 30_000.0
    gsplat_command = [
        "python",
        gsplat_script,
        "mcmc",
        "--data_dir",
        sfm_output_dir,
        "--data_factor",
        "1",
        "--result_dir",
        result_dir,
        "--save_ply_path",
        save_ply_path,
        "--use_bilateral_grid",
        "--use_fused_bilagrid",
        "--preview_data_path",
        preview_data_path,
        "--disable_eval",
        "--random_bkgd",
        "--mcmc_max_num_gaussians",
        str(training_max_num_gaussians),
        "--steps_scaler",
        str(steps_scaler)
    ]
    
    # Add mcmc_refine_every if provided
    if mcmc_refine_every is not None:
        gsplat_command.extend(["--mcmc_refine_every", str(mcmc_refine_every)])
        logger.info(f"Setting MCMC refine_every to {mcmc_refine_every}")
    
    # Disable world space normalization if normalize=False
    # This is used when data is already in world space (Quest) or when preserving real-world scale (AprilTag)
    if not normalize:
        gsplat_command.extend(["--no_normalize_world_space"])
        logger.info("Disabling world space normalization")
    else:
        logger.info("Using default world space normalization")


    logger.info("Running Gsplat script...")
    print(' '.join(gsplat_command))

    # Set up the directory watcher if a callback is provided
    observer = None
    if on_new_preview:
        event_handler = PreviewDataHandler(on_new_preview)
        observer = Observer()
        observer.schedule(event_handler, path=preview_data_path, recursive=False)
        observer.start()
        logger.info(f"Started watching directory: {preview_data_path}")

    try:
        # Run the command with kill_check. Pipe both streams so real trainer
        # errors surface in the logs (set TQDM_DISABLE=1 in the environment to
        # keep progress bars from flooding them).
        process_completed = run_command(gsplat_command, kill_check=kill_check, pipe_stderr=True, pipe_stdout=True)
        if not process_completed:
            logger.info("Gsplat process was terminated by kill_check")
            return None
    finally:
        # Stop the observer if it was started
        if observer:
            observer.stop()
            observer.join()
            logger.info(f"Stopped watching directory: {preview_data_path}")


def remove_backgrounds(images_output_dir, kill_check = None):
        logger.info(f"Removing background from images in {images_output_dir}...")
        # Create a temp directory for the background images using tmpfile
        temp_dir = tempfile.mkdtemp()
        
        # Run remove_background.py as a separate process instead of importing it because of torch issue
        remove_bg_script = os.path.join(os.path.dirname(__file__), "remove_background.py")
        remove_bg_command = [
            "python", 
            remove_bg_script,
            images_output_dir,
            temp_dir,
            "--bg_type", "rgba",
            "--mode", "fast",
            "--device", "cuda:0",
            "--resize", "static",
            "--threshold", "0.5"
        ]
        
        logger.info(f"Running background removal as separate process to workaround torch issue: \n {' '.join(remove_bg_command)}")
        run_command(remove_bg_command, kill_check=kill_check)
        
        # Copy the background images to the images_output_dir
        for file in os.listdir(temp_dir):
            shutil.copy(os.path.join(temp_dir, file), os.path.join(images_output_dir, file))
        # Remove the temp directory
        shutil.rmtree(temp_dir)
        

def apply_pilgram_filter(images_output_dir, filter_name):
    logger.info(f"Applying Pilgram filter '{filter_name}' to images in {images_output_dir}...")
    # Create a temp directory for the filtered images
    temp_dir = tempfile.mkdtemp()
    
    
    apply_filters_to_directory(images_output_dir, filter_name, temp_dir)
    
    
    # Copy the filtered images to the original directory
    for file in os.listdir(temp_dir):
        shutil.copy(os.path.join(temp_dir, file), os.path.join(images_output_dir, file))
    
    # Remove the temp directory
    shutil.rmtree(temp_dir)


def process_video_to_scene(
    video_path=None,
    image_dir=None,
    output_dir=None,
    sfm_dir=None,
    target_framecount=600,
    preview_data_handler=None,
    remove_background_from_images=False,
    equirectangular=False,
    insv_fisheye=False,
    insv_lens_fov=None,
    insv_calibration=None,
    insv_no_factory_calibration=False,
    use_background_sphere=False,
    apply_pilgram_filter_name=None,
    training_max_num_gaussians=1_000_000,
    training_num_steps=30_000,
    kill_check=None,
    reconstruction_method='glomap',
    apriltag_size_meters=None,
    mock=False,
    quest_project_dir=None,
    max_resolution=1920,
    stop_after_sfm=False,
    normalize_override=None,
):
    """
    Process a video into a 3D scene.

    Args:
        video_path: Path to the video file
        image_dir: Directory containing images to use instead of extracting frames
        output_dir: Directory to store output files
        sfm_dir: Directory containing pre-computed SfM data
        target_framecount: Target number of frames to extract from video
        preview_data_handler: Callback function when new preview data is available
        remove_background_from_images: Whether to remove backgrounds from images
        kill_check: Function that returns True if processing should be terminated
        equirectangular: Whether to use equirectangular/360 video input (uses rig-based SfM)
        insv_fisheye: Process an Insta360 .insv recording directly from its dual
            fisheye streams (skips the equirectangular intermediate). Enabled
            automatically when video_path ends in .insv.
        insv_lens_fov: Assumed full fisheye lens FOV in degrees (default 200)
        insv_calibration: Path to a lens calibration JSON, overriding the factory
            calibration (see docs/insv_fisheye.md)
        insv_no_factory_calibration: Ignore the per-unit factory calibration
            embedded in the recording and use the idealized lens model
        use_background_sphere: Whether to use a background sphere
        apply_pilgram_filter_name: Name of Pilgram filter to apply (if any)
        training_max_num_gaussians: Maximum number of Gaussians to use in Gsplat
        training_num_steps: Number of steps to use in Gsplat
        mock: If True, don't actually run processing, just return mock data
        reconstruction_method: Which reconstruction method to use ('glomap', 'colmap', 'vggt', or 'quest')
        apriltag_size_meters: Physical size of AprilTags in meters for scale calibration (None to disable)
        quest_project_dir: Path to Quest project directory (required when using 'quest' reconstruction method)
        max_resolution: Long-edge pixel cap applied when extracting frames (no upscaling)
        stop_after_sfm: If True, stop once all SfM-side artifacts (sparse model,
            images, background sphere/filters) are ready and return the SfM
            output directory instead of training. Resume later by calling again
            with sfm_dir=<that directory> (and the image-mutating flags off, so
            they are not applied twice).
        normalize_override: If not None, force gsplat world-space normalization
            on/off instead of deriving it from apriltag/quest. A resumed
            (sfm_dir=) run that had AprilTag scaling applied in its SfM stage
            must pass normalize_override=False, since passing
            apriltag_size_meters again would re-apply the scale factor.

    Returns:
        Path to the output PLY file; the SfM output directory if
        stop_after_sfm is True; or None if processing was terminated
    """
    gsplat_script = check_requirements(video_path, image_dir, sfm_dir, reconstruction_method, quest_project_dir)

    sfm_output_dir, images_output_dir, result_dir, save_ply_path, preview_data_path = prepare_directories(
        output_dir, sfm_dir
    )
    if mock:
        logger.info("Mocking...!")
        mock_ply_path = os.path.join(os.path.dirname(__file__), "assets", "mock_splat.ply")
        print(mock_ply_path)
        save_ply_path = os.path.join(output_dir, "ply", "splat.ply")
        os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)
        shutil.copy(mock_ply_path, save_ply_path)
        return save_ply_path

    # Check if we should abort before processing images
    if kill_check and kill_check():
        logger.info("Job was deleted before processing started, stopping")
        return None

    # Raw .insv recordings carry two fisheye streams; process them directly
    # unless the caller explicitly asked for the equirectangular path.
    if (
        video_path
        and str(video_path).lower().endswith(".insv")
        and not insv_fisheye
        and not equirectangular
    ):
        logger.info("Detected .insv input, enabling direct dual-fisheye processing")
        insv_fisheye = True

    if not sfm_dir:
        if insv_fisheye:
            from pathlib import Path
            if not video_path:
                raise ValueError("insv_fisheye requires video_path pointing at an .insv recording")
            render_options = fisheye_sfm.INSV_RENDER_OPTIONS
            views_per_frame = 2 * render_options.num_views_per_lens

            # Match the equirectangular path's total image budget: target
            # frames are bumped to at least 800 and each frame pair fans out
            # into views_per_frame perspective images.
            insv_target = max(target_framecount, 800)
            frame_pair_count = max(3, int(insv_target / (views_per_frame / 2)))
            logger.info(
                f".insv video: extracting {frame_pair_count} fisheye frame pairs x "
                f"{views_per_frame} virtual cameras = {frame_pair_count * views_per_frame} total images"
            )

            logger.info("Running direct dual-fisheye SfM pipeline with rig constraints")
            sparse_result = fisheye_sfm.run_insv_sfm(
                insv_path=video_path,
                output_path=Path(sfm_output_dir),
                target_framecount=frame_pair_count,
                mapper="colmap",  # COLMAP incremental mapper (more stable with pycolmap rig config)
                generate_masks=True,
                lens_fov_deg=insv_lens_fov,
                calibration_path=insv_calibration,
                use_factory_calibration=not insv_no_factory_calibration,
                kill_check=kill_check,
            )

            if sparse_result is None:
                logger.error("Direct dual-fisheye SfM pipeline failed or was cancelled")
                return None

            # Select the best sparse subdirectory (largest points3D.bin) for gsplat
            select_best_sparse_subdir(sfm_output_dir)

            # Run orientation alignment to fix up direction
            align_sfm_orientation(sfm_output_dir, kill_check=kill_check)

            # Update images_output_dir to point to rendered perspectives
            images_output_dir = os.path.join(sfm_output_dir, "images")
            logger.info(f"Dual-fisheye SfM completed, perspectives in {images_output_dir}")
        elif equirectangular:
            # Use new pano_sfm pipeline for 360 content
            from pathlib import Path
            render_options = pano_sfm.PANO_RENDER_OPTIONS
            num_virtual_cams = render_options.num_virtual_cameras
            
            # Increase target framecount for equirectangular - more frames = better quality
            equirect_target = max(target_framecount, 800)
            
            # Reduce panorama count to keep total images manageable
            pano_framecount = int(equirect_target / 6)  # ~133 panos for 800 target
            logger.info(f"360 video: extracting {pano_framecount} panoramas x {num_virtual_cams} virtual cameras = {pano_framecount * num_virtual_cams} total images")
            
            # Extract panorama frames (don't downscale - we need full resolution)
            pano_temp_dir = tempfile.mkdtemp(prefix="pano_frames_")
            if image_dir:
                logger.info(f"Copying panorama images from {image_dir} to {pano_temp_dir}")
                shutil.copytree(image_dir, pano_temp_dir, dirs_exist_ok=True)
            else:
                logger.info(f"Extracting panorama frames from video...")
                extract_frames.extract_frames(video_path, pano_temp_dir, pano_framecount, downscale=False)
            
            if kill_check and kill_check():
                logger.info("Job was deleted after extracting panorama frames, stopping")
                shutil.rmtree(pano_temp_dir)
                return None
            
            # Run pano_sfm pipeline (renders perspectives and runs SfM with COLMAP)
            logger.info("Running 360 SfM pipeline with overlapping rig and sequential matching")
            sparse_result = pano_sfm.run_pano_sfm(
                input_image_path=Path(pano_temp_dir),
                output_path=Path(sfm_output_dir),
                mapper="colmap",  # COLMAP incremental mapper (more stable with pycolmap rig config)
                generate_masks=True,  # Required for clean reconstruction
                kill_check=kill_check,
            )
            
            # Clean up temp directory
            shutil.rmtree(pano_temp_dir)
            
            if sparse_result is None:
                logger.error("360 SfM pipeline failed or was cancelled")
                return None
            
            # Select the best sparse subdirectory (largest points3D.bin) for gsplat
            select_best_sparse_subdir(sfm_output_dir)
            
            # Run orientation alignment to fix up direction
            align_sfm_orientation(sfm_output_dir, kill_check=kill_check)
            
            # Update images_output_dir to point to rendered perspectives
            images_output_dir = os.path.join(sfm_output_dir, "images")
            logger.info(f"360 SfM completed, perspectives in {images_output_dir}")
        else:
            if reconstruction_method == 'quest':
                logger.info("Using Quest reconstruction method - skipping image extraction")
            else:
                if reconstruction_method == 'vggt':
                    logger.warning("VGGT requires low framecount due to memory constraints, setting target_framecount to 35")
                    target_framecount = 35

                handle_images(video_path, image_dir, images_output_dir, target_framecount, equirectangular=False, max_resolution=max_resolution)
                
                # Check if we should abort after handling images
                if kill_check and kill_check():
                    logger.info("Job was deleted after handling images, stopping")
                    return None
        
            generate_sfm_point_cloud(
                sfm_dir, images_output_dir, sfm_output_dir, kill_check=kill_check, reconstruction_method=reconstruction_method,
                apriltag_size_meters=apriltag_size_meters, quest_project_dir=quest_project_dir
            )
        
        # Check if we should abort after generating point cloud
        if kill_check and kill_check():
            logger.info("Job was deleted after generating point cloud, stopping")
            return None

        
    # Apply AprilTag scale calibration if requested
    apply_apriltag_scale_calibration(
        sfm_output_dir, 
        images_output_dir, 
        apriltag_size_meters, 
        apriltag_debug=False,
        kill_check=kill_check
    )

    # For quest method, images_output_dir might not exist yet (it's created by quest reconstruction)
    if reconstruction_method == 'quest':
        images_output_dir = os.path.join(sfm_output_dir, "images")
        if not os.path.exists(images_output_dir):
            logger.warning(f"Images directory not found at {images_output_dir}")
            num_frames = 0
        else:
            num_frames = len(os.listdir(images_output_dir))
            logger.info(f"Number of frames from Quest reconstruction: {num_frames}")
    else:
        num_frames = len(os.listdir(images_output_dir))
        logger.info(f"Number of frames extracted: {num_frames}")

    if remove_background_from_images:
        remove_backgrounds(images_output_dir, kill_check=kill_check)
    elif use_background_sphere:
        add_background_sphere_to_sfm_model(sfm_output_dir)

    # Apply Pilgram filter if requested
    if apply_pilgram_filter_name:
        apply_pilgram_filter(images_output_dir, apply_pilgram_filter_name)

    # Everything gsplat consumes (sparse model, images, background sphere,
    # filters) is now in place — callers that split SfM and training across
    # machines stop here and resume with sfm_dir= pointed at this directory.
    if stop_after_sfm:
        logger.info(f"stop_after_sfm set; SfM artifacts ready in {sfm_output_dir}")
        return sfm_output_dir

    # Calculate mcmc_refine_every for Quest reconstruction (scale with number of images, rounded to nearest 100)
    mcmc_refine_every = None
    if reconstruction_method == 'quest' and num_frames > 0:
        # Scale with number of images, rounded to nearest 100
        mcmc_refine_every = round(num_frames / 100) * 100
        # Ensure a minimum value of 100
        mcmc_refine_every = max(200, mcmc_refine_every)
        logger.info(f"Setting MCMC refine_every to {mcmc_refine_every} based on {num_frames} frames")
    
    # Run gsplat with the kill_check function
    # Disable world space normalization if AprilTag calibration was used or if using Quest reconstruction
    # (Quest data already comes in world space, AprilTag preserves real-world scale)
    if normalize_override is not None:
        normalize = normalize_override
    else:
        normalize = apriltag_size_meters is None and reconstruction_method != 'quest'
    run_gsplat(gsplat_script, sfm_output_dir, result_dir, save_ply_path, training_max_num_gaussians, training_num_steps,
               preview_data_path, preview_data_handler, kill_check, normalize=normalize, mcmc_refine_every=mcmc_refine_every)
    
    # Check if we should stop after gsplat
    if kill_check and kill_check():
        logger.info("Job was deleted after running gsplat, stopping")
        return None

    logger.info(
        f"Process completed. Images saved in {images_output_dir}, SfM model saved in {sfm_output_dir}, and Gsplat output saved in {save_ply_path}."
    )
    return save_ply_path

def add_background_sphere_to_sfm_model(sfm_output_dir):
        logger.info("Adding background sphere to SfM model points3D.bin...")
        points3d_bin_path = os.path.join(sfm_output_dir, "sparse", "0", "points3D.bin")
        if not os.path.exists(points3d_bin_path):
            raise ValueError(f"The points3D.bin file does not exist in {points3d_bin_path}")
        points3d_bin_bg_path = os.path.join(sfm_output_dir, "sparse", "0", "points3D_bg.bin")
        add_background_sphere(
            points3d_bin_path,
            points3d_bin_bg_path
        )
        # replace the points3D.bin file with the points3D_bg.bin file
        shutil.copy(points3d_bin_bg_path, points3d_bin_path)


def convert_ply_to_spz(ply_path, spz_path, kill_check=None):
    spz_to_ply_command = [
        SPZ_TO_PLY_EXECUTABLE_PATH,
        ply_path,
        spz_path,
    ]
    logger.info(f"Converting PLY to SPZ: {ply_path} -> {spz_path}")
    
    # First check if we should abort
    if kill_check and kill_check():
        logger.info("Job was deleted before PLY to SPZ conversion, stopping")
        return False
        
    run_command(spz_to_ply_command, log_info_output=False, kill_check=kill_check)
    return True

def prune_ply(ply_path, new_ply_path, alpha_prune_threshold=12, kill_check=None):
    spz_to_ply_command = [
        SPZ_TO_PLY_EXECUTABLE_PATH,
        ply_path,
        new_ply_path,
        "-a",
        str(alpha_prune_threshold) 
    ]

    # First check if we should abort
    if kill_check and kill_check():
        logger.info("Job was deleted before PLY pruning, stopping")
        return False

    logger.info(f"Pruning PLY: {ply_path} -> {new_ply_path}")
    run_command(spz_to_ply_command, log_info_output=False, kill_check=kill_check)
    return True

if __name__ == "__main__":
    args = parse_arguments()
    process_video_to_scene(
        video_path=args.video_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        sfm_dir=args.sfm_dir,
        target_framecount=args.target_framecount,
        preview_data_handler=None,
        remove_background_from_images=args.remove_background,
        equirectangular=args.equirectangular,
        insv_fisheye=args.insv_fisheye,
        insv_lens_fov=args.insv_lens_fov,
        insv_calibration=args.insv_calibration,
        insv_no_factory_calibration=args.insv_no_factory_calibration,
        use_background_sphere=args.use_background_sphere,
        apply_pilgram_filter_name=args.apply_pilgram_filter,
        training_max_num_gaussians=args.training_max_num_gaussians,
        training_num_steps=args.training_num_steps,
        kill_check=None,
        mock=args.mock,
        reconstruction_method=args.reconstruction_method,
        apriltag_size_meters=args.apriltag_size,
        quest_project_dir=args.quest_project_dir,
        max_resolution=args.max_resolution,
        stop_after_sfm=args.stop_after_sfm,
    )
