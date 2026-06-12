"""
Insta360 .insv ingestion: demux the dual fisheye video streams to paired
per-lens frame directories, and best-effort parse the proprietary metadata
trailer.

An .insv file is an ISO-BMFF (MP4-like) container that ffmpeg can read
directly. Depending on the camera generation the two fisheye eyes are stored
as:
  - two separate video streams in one file (X3/X4/X5 single-file recordings),
  - two separate files, ``..._00_...insv`` and ``..._10_...insv`` (older
    cameras, and high-resolution modes on some models), or
  - one video stream with both fisheyes side by side (some older models).

After the MP4 body, Insta360 appends a proprietary trailer holding IMU
samples, exposure data, GPS, and file info. The format was reverse engineered
by the community (ExifTool's QuickTimeStream.pl and
https://subethasoftware.com/2022/06/08/insta360-one-x2-insv-file-format/):
the file ends with a 32-byte ASCII magic, and records are walked backwards
from EOF-78, each record being a (uint16 id, uint32 size) descriptor preceded
by ``size`` bytes of payload. Trailer parsing here is purely informational
(camera model detection / logging); every failure degrades to an empty result.
"""

import json
import logging
import os
import re
import shutil
import struct
import subprocess
from math import ceil
from pathlib import Path

logger = logging.getLogger(__name__)

INSV_TRAILER_MAGIC = b"8db42d694ccc418790edff439fe026bf"
INSV_TRAILER_FOOTER_SIZE = 78

# Known trailer record ids (per ExifTool QuickTimeStream.pl). Only 0x101 is
# interpreted here; the rest are listed for logging/debugging.
INSV_RECORD_NAMES = {
    0x101: "file_info",
    0x200: "preview_image",
    0x300: "gyro",
    0x400: "exposure",
    0x600: "timestamps",
    0x700: "gps",
}


def read_insv_trailer_records(path, max_records: int = 64) -> dict[int, bytes]:
    """Best-effort parse of the .insv trailer into {record_id: payload_bytes}.

    Returns an empty dict if the trailer is missing or doesn't parse; callers
    must treat all metadata as optional.
    """
    try:
        file_size = os.path.getsize(path)
        if file_size < INSV_TRAILER_FOOTER_SIZE:
            return {}
        with open(path, "rb") as f:
            f.seek(file_size - INSV_TRAILER_FOOTER_SIZE)
            footer = f.read(INSV_TRAILER_FOOTER_SIZE)
            if footer[-len(INSV_TRAILER_MAGIC):] != INSV_TRAILER_MAGIC:
                return {}

            records: dict[int, bytes] = {}
            # The first record descriptor is the start of the 78-byte footer;
            # each record's payload sits immediately before its descriptor,
            # and the previous record's descriptor before that payload.
            pos = file_size - INSV_TRAILER_FOOTER_SIZE
            for _ in range(max_records):
                if pos < 6:
                    break
                f.seek(pos)
                record_id, record_size = struct.unpack("<HI", f.read(6))
                if record_size == 0 or record_size > pos:
                    break
                if record_id == 0:
                    # Newer trailers (seen on X5 fw 1.9.x; trailer version 3 in
                    # the footer) chain an id-0 record at the top whose payload
                    # is an index table, and records below it no longer follow
                    # the strict payload+descriptor chain — so the table is
                    # authoritative for everything else.
                    f.seek(pos - record_size)
                    table = f.read(record_size)
                    records.update(
                        _parse_trailer_index(f, table, file_size, footer)
                    )
                    break
                f.seek(pos - record_size)
                records[record_id] = f.read(record_size)
                pos = pos - record_size - 6
            return records
    except (OSError, struct.error) as e:
        logger.warning(f"Could not parse .insv trailer of {path}: {e}")
        return {}


def _parse_trailer_index(f, table: bytes, file_size: int, footer: bytes) -> dict[int, bytes]:
    """Resolve a version-3 trailer index table into {record_id: payload}.

    The table is the payload of the id-0 record: 10-byte entries of
    (uint16 id, uint32 size, uint32 offset), offset relative to the trailer
    data start (= EOF minus the trailer size stored in the footer). Zero or
    out-of-range entries are slot padding and skipped. Small ids are the
    legacy record ids shifted right by 8 (2 -> 0x200 preview, 3 -> 0x300
    gyro, ...); 0x101 (file_info) keeps its legacy value.
    """
    trailer_size = struct.unpack("<I", footer[38:42])[0]
    trailer_start = file_size - trailer_size
    if not 0 < trailer_start < file_size:
        return {}
    records: dict[int, bytes] = {}
    for entry_pos in range(0, len(table) - 9, 10):
        rec_id, rec_size, rec_offset = struct.unpack_from("<HII", table, entry_pos)
        if rec_id == 0 or rec_size == 0 or rec_offset + rec_size > trailer_size:
            continue
        legacy_id = rec_id if rec_id >= 0x100 else rec_id << 8
        f.seek(trailer_start + rec_offset)
        records[legacy_id] = f.read(rec_size)
    return records


def summarize_insv_metadata(records: dict[int, bytes]) -> dict:
    """Extract loggable info from trailer records (camera model, serial, ...).

    Record 0x101 is a small protobuf message whose first three string fields
    are serial number, camera model, and firmware version. Rather than pull in
    a protobuf dependency, printable ASCII runs are extracted, which yields
    the same strings.
    """
    info: dict = {
        "record_ids": {
            f"0x{record_id:x}": INSV_RECORD_NAMES.get(record_id, "unknown")
            for record_id in sorted(records)
        }
    }
    file_info = records.get(0x101)
    if file_info:
        strings = [s.decode("ascii") for s in re.findall(rb"[ -~]{4,}", file_info)]
        info["file_info_strings"] = strings
    return info


def find_companion_lens_file(insv_path) -> Path | None:
    """Find the second-eye file of a two-file recording (``_00_`` <-> ``_10_``)."""
    path = Path(insv_path)
    name = path.name
    for this_tag, other_tag in (("_00_", "_10_"), ("_10_", "_00_")):
        if this_tag in name:
            companion = path.with_name(name.replace(this_tag, other_tag, 1))
            if companion.exists():
                return companion
            return None
    return None


def probe_video_streams(path) -> list[dict]:
    """Return [{width, height}] for each video stream, in ffmpeg ordinal order."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=width,height",
        "-print_format", "json",
        str(path),
    ]
    output = subprocess.check_output(command)
    streams = json.loads(output).get("streams", [])
    return [{"width": int(s["width"]), "height": int(s["height"])} for s in streams]


def count_stream_frames(path, stream_ordinal: int) -> int | None:
    """Count frames (packets) of video stream ``v:stream_ordinal``."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", f"v:{stream_ordinal}",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        return int(subprocess.check_output(command).strip())
    except (subprocess.CalledProcessError, ValueError):
        logger.error(f"Unable to count frames of stream v:{stream_ordinal} in {path}")
        return None


def looks_like_fisheye_frame(image) -> bool:
    """Heuristic: fisheye frames have a dark border outside the image circle.

    Used to distinguish a side-by-side dual-fisheye stream from an already
    stitched equirectangular video (both are 2:1).
    """
    import numpy as np

    h, w = image.shape[:2]
    margin = max(2, min(h, w) // 50)
    corners = [
        image[:margin, :margin],
        image[:margin, -margin:],
        image[-margin:, :margin],
        image[-margin:, -margin:],
    ]
    mean_corner = float(np.mean([np.mean(c) for c in corners]))
    return mean_corner < 20.0


def _extract_stream_frames(path, stream_ordinal: int, output_dir, interval: int, crop_filter: str | None = None):
    """Extract every ``interval``-th frame of one video stream as PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    video_filter = f"select=not(mod(n\\,{interval}))"
    if crop_filter:
        video_filter += f",{crop_filter}"
    output_pattern = os.path.join(output_dir, "frame_%05d.png")
    command = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(path),
        "-map", f"0:v:{stream_ordinal}",
        "-vf", video_filter,
        "-vsync", "vfr",
        "-start_number", "1",
        output_pattern,
    ]
    subprocess.run(command, check=True)


def _trim_to_paired_frames(lens_dirs: list[str]) -> int:
    """Delete trailing frames so both lens directories have the same count."""
    frame_lists = [sorted(os.listdir(d)) for d in lens_dirs]
    num_pairs = min(len(frames) for frames in frame_lists)
    for lens_dir, frames in zip(lens_dirs, frame_lists):
        for extra in frames[num_pairs:]:
            os.remove(os.path.join(lens_dir, extra))
    return num_pairs


def extract_dual_fisheye_frames(insv_path, output_dir, target_framecount: int) -> list[str]:
    """Demux an .insv recording into two paired per-lens frame directories.

    Returns [lens0_dir, lens1_dir] where each directory holds PNG frames named
    frame_%05d.png; frame N in lens0 was captured (approximately) simultaneously
    with frame N in lens1.
    """
    os.makedirs(output_dir, exist_ok=True)
    lens_dirs = [os.path.join(output_dir, "lens0"), os.path.join(output_dir, "lens1")]

    streams = probe_video_streams(insv_path)
    if not streams:
        raise ValueError(f"No video streams found in {insv_path}")
    companion = find_companion_lens_file(insv_path)

    if len(streams) >= 2:
        # Modern single-file recording: one video stream per lens. Streams
        # beyond the first two (e.g. low-res preview tracks) are ignored.
        if streams[0]["width"] != streams[1]["width"] or streams[0]["height"] != streams[1]["height"]:
            raise ValueError(
                f"First two video streams of {insv_path} differ in size "
                f"({streams[0]} vs {streams[1]}); not a dual-fisheye recording?"
            )
        total_frames = min(
            count_stream_frames(insv_path, 0) or 0,
            count_stream_frames(insv_path, 1) or 0,
        )
        interval = _frame_interval(total_frames, target_framecount)
        logger.info(
            f"Demuxing 2 fisheye streams ({streams[0]['width']}x{streams[0]['height']}) "
            f"from {insv_path}, every {interval}th of {total_frames} frames"
        )
        for lens_idx in (0, 1):
            _extract_stream_frames(insv_path, lens_idx, lens_dirs[lens_idx], interval)
    elif companion is not None:
        # Two-file recording: each file holds one lens. Which file is which
        # physical lens only flips the reconstruction's global yaw, so the
        # assignment doesn't need to be exact.
        paths = sorted([str(insv_path), str(companion)])
        total_frames = min(count_stream_frames(p, 0) or 0 for p in paths)
        interval = _frame_interval(total_frames, target_framecount)
        logger.info(
            f"Demuxing two-file recording {paths[0]} + {paths[1]}, "
            f"every {interval}th of {total_frames} frames"
        )
        for lens_idx, path in enumerate(paths):
            _extract_stream_frames(path, 0, lens_dirs[lens_idx], interval)
    elif streams[0]["width"] == 2 * streams[0]["height"]:
        # Single stream with both fisheyes side by side. Guard against
        # equirectangular input, which has the same 2:1 aspect ratio.
        _check_side_by_side_is_fisheye(insv_path)
        total_frames = count_stream_frames(insv_path, 0) or 0
        interval = _frame_interval(total_frames, target_framecount)
        logger.info(
            f"Splitting side-by-side dual fisheye stream of {insv_path}, "
            f"every {interval}th of {total_frames} frames"
        )
        _extract_stream_frames(insv_path, 0, lens_dirs[0], interval, crop_filter="crop=iw/2:ih:0:0")
        _extract_stream_frames(insv_path, 0, lens_dirs[1], interval, crop_filter="crop=iw/2:ih:iw/2:0")
    else:
        raise ValueError(
            f"Unsupported .insv layout in {insv_path}: single "
            f"{streams[0]['width']}x{streams[0]['height']} video stream and no "
            f"companion _00_/_10_ file"
        )

    num_pairs = _trim_to_paired_frames(lens_dirs)
    if num_pairs == 0:
        raise ValueError(f"No frames could be extracted from {insv_path}")
    logger.info(f"Extracted {num_pairs} paired fisheye frames to {output_dir}")
    return lens_dirs


def _frame_interval(total_frames: int, target_framecount: int) -> int:
    if total_frames <= 0:
        raise ValueError("Could not determine the video frame count")
    if target_framecount and target_framecount < total_frames:
        return max(1, ceil(total_frames / target_framecount))
    return 1


def _check_side_by_side_is_fisheye(insv_path):
    """Extract one frame and verify both halves look like fisheye circles."""
    import tempfile

    import cv2

    probe_dir = tempfile.mkdtemp(prefix="insv_probe_")
    try:
        probe_frame = os.path.join(probe_dir, "frame_%05d.png")
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error",
                "-i", str(insv_path),
                "-map", "0:v:0", "-frames:v", "1",
                probe_frame,
            ],
            check=True,
        )
        image = cv2.imread(os.path.join(probe_dir, "frame_00001.png"))
        if image is None:
            return  # Can't verify; let the pipeline proceed and fail later if wrong.
        half_width = image.shape[1] // 2
        halves = [image[:, :half_width], image[:, half_width:]]
        if not all(looks_like_fisheye_frame(h) for h in halves):
            raise ValueError(
                f"{insv_path} has a single 2:1 video stream that does not look like "
                "side-by-side fisheye footage (no dark image-circle borders). If this "
                "is an equirectangular video, use --equirectangular instead."
            )
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
