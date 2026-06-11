import os
import subprocess
import argparse
import logging
logger = logging.getLogger(__name__)


def get_total_frames(video_path):
    """Uses ffprobe to get the total number of frames in the video."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]

    try:
        total_frames = int(subprocess.check_output(command).strip())
        logger.info(f"Total frames in video: {total_frames}")
    except subprocess.CalledProcessError:
        logger.error(f"Error: Unable to retrieve frame count from video {video_path}")
        return None

    return total_frames


def get_duration(video_path):
    """Uses ffprobe to get the video duration in seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]

    try:
        return float(subprocess.check_output(command).strip())
    except (subprocess.CalledProcessError, ValueError):
        logger.error(f"Error: Unable to retrieve duration from video {video_path}")
        return None


def extract_frames(video_path, output_dir, target_framecount=None, downscale=True, max_resolution=1920):
    """Extract frames using ffmpeg based on the target frame count.

    max_resolution caps the long edge (in pixels) when downscale is True;
    frames smaller than the cap are never upscaled."""

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_frames = get_total_frames(video_path)

    if total_frames is None:
        return

    # When the video has more frames than the target, resample by timestamp to
    # exactly the target count. A whole-frame stride (every Nth frame) only
    # hits the target when total is an exact multiple — a 901-frame video with
    # a 900 cap would yield 450. The fps filter strides fractionally; the rate
    # is biased up by half a frame and -frames:v trims any overshoot, so the
    # output is exactly target_framecount, evenly spaced, for any video length.
    frame_limit = None
    if target_framecount and target_framecount < total_frames:
        duration = get_duration(video_path)
        if not duration:
            return
        rate = (target_framecount + 0.5) / duration
        video_filter_string = f"fps={rate:.6f}"
        frame_limit = target_framecount
        logger.info(
            f"Resampling {total_frames} frames at {rate:.4f} fps to extract exactly {target_framecount} frames."
        )
    else:
        # No target, or the video is at/under it: extract every frame.
        video_filter_string = "select=not(mod(n\,1))"
        logger.info(f"Extracting all {total_frames} frames (target {target_framecount}).")

    logger.info(f"Total frames in video: {total_frames}")

    output_pattern = os.path.join(output_dir, "image_%04d.png")
    if downscale:
        video_filter_string += f",scale=if(gte(iw\,ih)\,min({max_resolution}\,iw)\,-2):if(lt(iw\,ih)\,min({max_resolution}\,ih)\,-2)"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        video_filter_string,
        "-vsync",
        "vfr",
        output_pattern,
    ]
    if frame_limit:
        ffmpeg_command[-1:-1] = ["-frames:v", str(frame_limit)]

    try:
        subprocess.run(ffmpeg_command, check=True)
        logger.info(f"Frame extraction complete. Frames saved to {output_dir}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during frame extraction: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video using ffmpeg."
    )
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("output_dir", help="Directory to save the extracted frames.")
    parser.add_argument(
        "--target_framecount",
        type=int,
        help="Target number of frames to extract.",
        default=None,
    )
    parser.add_argument(
        "--downscale",
        type=bool,
        help="Downscale the frames to --max_resolution on the long edge",
        default=True,
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        help="Maximum long-edge resolution in pixels when downscaling.",
        default=1920,
    )

    args = parser.parse_args()

    # Run frame extraction
    extract_frames(args.video_path, args.output_dir, args.target_framecount, downscale=args.downscale, max_resolution=args.max_resolution)


if __name__ == "__main__":
    main()
