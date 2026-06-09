import os
import subprocess
import argparse
import logging
from math import ceil
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


def extract_frames(video_path, output_dir, target_framecount=None, downscale=True, max_resolution=1920):
    """Extract frames using ffmpeg based on the target frame count.

    max_resolution caps the long edge (in pixels) when downscale is True;
    frames smaller than the cap are never upscaled."""

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_frames = get_total_frames(video_path)

    if total_frames is None:
        return

    # Calculate the interval based on the target frame count
    if target_framecount and target_framecount < total_frames:
        interval = max(1, ceil(total_frames / target_framecount))
    else:
        interval = 1  # If no target frame count is given or it's larger than total frames, extract every frame

    logger.info(f"Total frames in video: {total_frames}")
    logger.info(
        f"Extracting frames at an interval of {interval} to aim for {target_framecount} frames."
    )

    # ffmpeg command to extract frames at calculated intervals
    output_pattern = os.path.join(output_dir, "image_%04d.png")
    video_filter_string = f"select=not(mod(n\,{interval}))"
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
