"""
Insta360 factory lens calibration parsing for .insv recordings.

Every Insta360 camera is calibrated per-unit at the factory and the result is
embedded in the recording itself, in two places:

- The metadata record of the .insv trailer (record 0x101, a protobuf message)
  holds ``offset_v3``: an underscore-separated float string with one MEI
  (unified omnidirectional) lens block per lens. Layout per the prost
  definitions in AdrianEddy/telemetry-parser (src/insta360/extra_info.rs) and
  the field comment in src/insta360/mod.rs:
  ``num, then per lens: xi fx fy cx cy yaw pitch roll tx ty tz
  k1 k2 k3 p1 p2 width height lensType flag``.
- Newer cameras (X5) additionally ship a ``<name>.insv.pb`` sidecar (under
  MISC/Camera01 on the SD card) whose base64 payload holds an extended
  calibration with 27 fields per lens, adding k4, thin-prism (s1..s4) and
  sensor-tilt terms. Layout reverse engineered and validated against
  in-camera stitching by BenjaminHenriksson/insv-stitch.

Calibration values are stored at a per-model reference resolution (e.g.
5376px per lens on the X5, 8000x6000 on the X4) while the demuxed video
streams are smaller center crops (window_crop_info in the same protobuf gives
the crop when present); this module converts them to actual stream resolution.

Everything here is best effort: any parse failure returns None and the
caller falls back to the idealized lens model. Pure stdlib (a ~40-line
protobuf wire-format walker avoids a protobuf dependency for nine fields).

Caveats, pending validation on real footage (see docs/insv_fisheye.md):
- The yaw/pitch mounting-correction sign convention is taken from
  insv-stitch; values are fractions of a degree, so a wrong sign is a small
  systematic error rather than a failure.
- Per-lens field 7 of the .pb block is ambiguous in community sources
  (roll vs half-FOV; observed ~89.8) and is not used.
- The lens translation (tx, ty, tz; the ~2-3cm baseline) is parsed but not
  applied, matching the zero-baseline rig assumption of the ER path.
"""

import base64
import logging
import re
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# Record key in insv_extract.read_insv_trailer_records: (format | id << 8)
# with format=1 (protobuf) and id=1 (metadata).
METADATA_RECORD_KEY = 0x101

_METADATA_STRING_FIELDS = {
    1: "serial_number",
    2: "camera_type",
    3: "fw_version",
    5: "offset",
    17: "original_offset",
    53: "offset_v2",
    54: "offset_v3",
    55: "original_offset_v2",
    56: "original_offset_v3",
}
_WINDOW_CROP_INFO_FIELD = 27  # submessage: src_width, src_height, dst_width, dst_height

_OFFSET_V3_LENS_FIELDS = (
    "xi", "fx", "fy", "cx", "cy",
    "yaw_deg", "pitch_deg", "roll_deg",
    "tx", "ty", "tz",
    "k1", "k2", "k3", "p1", "p2",
    "ref_width", "ref_height", "lens_type", "flag",
)

_PB_LENS_BLOCK_SIZE = 27


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data) or shift > 63:
            raise ValueError("Truncated or oversized varint")
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not byte & 0x80:
            return result, pos
        shift += 7


def iter_protobuf_fields(data: bytes):
    """Yield (field_number, wire_type, raw_value) for a protobuf message.

    Varints yield ints; length-delimited fields yield bytes; fixed32/fixed64
    yield their raw bytes. Raises ValueError on malformed input.
    """
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
        elif wire_type == 1:
            value, pos = data[pos:pos + 8], pos + 8
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            if pos + length > len(data):
                raise ValueError("Truncated length-delimited field")
            value, pos = data[pos:pos + length], pos + length
        elif wire_type == 5:
            value, pos = data[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported wire type {wire_type}")
        if pos > len(data):
            raise ValueError("Truncated field payload")
        yield field_number, wire_type, value


def parse_metadata_record(payload: bytes) -> dict:
    """Decode the interesting fields of the trailer's protobuf metadata record.

    Returns {} if the payload doesn't parse as a protobuf message.
    """
    info: dict = {}
    try:
        for field_number, wire_type, value in iter_protobuf_fields(payload):
            name = _METADATA_STRING_FIELDS.get(field_number)
            if name is not None and wire_type == 2:
                try:
                    info[name] = value.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            elif field_number == _WINDOW_CROP_INFO_FIELD and wire_type == 2:
                crop = [v for n, w, v in iter_protobuf_fields(value) if w == 0]
                if len(crop) >= 4:
                    info["window_crop"] = {
                        "src_width": crop[0], "src_height": crop[1],
                        "dst_width": crop[2], "dst_height": crop[3],
                    }
    except ValueError as e:
        logger.debug(f"Unparseable .insv metadata record: {e}")
        return {}
    return info


def _parse_float_list(text: str) -> list[float] | None:
    try:
        return [float(token) for token in text.split("_")]
    except (ValueError, AttributeError):
        return None


def parse_offset_v3(text: str) -> list[dict] | None:
    """Parse the trailer's offset_v3 string into per-lens MEI lens specs.

    The per-lens field count varies by camera/firmware: X4-era trailers
    write 20 fields (with a trailing per-lens flag), X5 fw 1.9 writes 19
    (no flag) followed by file-level trailing values. Both layouts are
    tried; the winner must put plausible values in the reference-dimension
    slots (a misaligned block lands lens_type/flag-scale values there).
    """
    values = _parse_float_list(text)
    if not values:
        return None
    num_lenses = int(values[0])
    if num_lenses < 1:
        return None
    for field_names in (_OFFSET_V3_LENS_FIELDS, _OFFSET_V3_LENS_FIELDS[:-1]):
        block = len(field_names)
        if len(values) < 1 + num_lenses * block:
            continue
        lenses = [
            dict(zip(field_names, values[1 + i * block: 1 + (i + 1) * block]))
            for i in range(num_lenses)
        ]
        if all(_offset_v3_lens_plausible(spec) for spec in lenses):
            return lenses
    return None


def _offset_v3_lens_plausible(spec: dict) -> bool:
    return (
        spec["fx"] > 0
        and spec["fy"] > 0
        and spec["ref_width"] >= spec["ref_height"] >= 1000
    )


def parse_pb_calibration(data: bytes) -> list[dict] | None:
    """Parse the extended calibration out of an .insv.pb sidecar file.

    The sidecar embeds a base64 blob containing an underscore-separated
    calibration string of 1 + 27 * num_lenses values. Field layout within
    each lens block (insv-stitch, validated against in-camera stitching):
    0=xi 1=fx 2=fy 3=cx 4=cy 5=yaw 6=pitch 7=(ambiguous, unused)
    8-10=tx,ty,tz 11-14=k1..k4 15=zero 16=p1 17=p2 18-21=s1..s4
    22-23=tauX,tauY 24=ref_width(full dual frame) 25=ref_height 26=type
    """
    text = data.decode("latin-1")
    for blob_match in re.finditer(r"[A-Za-z0-9+/=]{100,}", text):
        blob = blob_match.group()
        # Surrounding bytes that happen to be base64-alphabet characters can
        # get glued onto the blob and shift its framing; try all four
        # alignments, truncating the tail to a decodable length.
        for start in range(4):
            candidate = blob[start:]
            candidate = candidate[: len(candidate) - len(candidate) % 4]
            try:
                decoded = base64.b64decode(candidate).decode("latin-1")
            except (ValueError, UnicodeDecodeError):
                continue
            lenses = _find_pb_calibration_string(decoded)
            if lenses:
                return lenses
    return None


def _find_pb_calibration_string(decoded: str) -> list[dict] | None:
    for match in re.finditer(r"2_\d+\.\d+_", decoded):
        values = []
        for token in decoded[match.start():].split("_"):
            # The last number can have non-numeric bytes glued straight onto
            # it, so accept a numeric prefix and stop there.
            numeric = re.match(r"[-+]?\d+(?:\.\d+)?", token)
            if not numeric:
                break
            values.append(float(numeric.group()))
            if numeric.end() != len(token):
                break
        num_lenses = int(values[0]) if values else 0
        if num_lenses < 1 or len(values) < 1 + num_lenses * _PB_LENS_BLOCK_SIZE:
            continue
        lenses = []
        for lens_idx in range(num_lenses):
            b = values[1 + lens_idx * _PB_LENS_BLOCK_SIZE:]
            lenses.append({
                "xi": b[0], "fx": b[1], "fy": b[2], "cx": b[3], "cy": b[4],
                "yaw_deg": b[5], "pitch_deg": b[6], "roll_deg": 0.0,
                "tx": b[8], "ty": b[9], "tz": b[10],
                "k1": b[11], "k2": b[12], "k3": b[13], "k4": b[14],
                "p1": b[16], "p2": b[17],
                "s1": b[18], "s2": b[19], "s3": b[20], "s4": b[21],
                "ref_width": b[24], "ref_height": b[25], "lens_type": b[26],
            })
        return lenses
    return None


def find_pb_sidecar(insv_path) -> Path | None:
    """Locate the .pb calibration sidecar of an .insv recording.

    On the SD card it lives in MISC/Camera01/<name>.insv.pb beside
    DCIM/Camera01/<name>.insv; users copying files often place it next to
    the video instead, so both locations are checked.
    """
    insv_path = Path(insv_path)
    candidates = [
        insv_path.with_name(insv_path.name + ".pb"),
        insv_path.parent.parent.parent / "MISC" / "Camera01" / (insv_path.name + ".pb"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _normalize_lens_spec(spec: dict, lens_idx: int) -> dict:
    """Normalize a per-lens spec to a single-lens reference frame.

    Reference dims may describe the full side-by-side dual-fisheye frame
    (e.g. 16000x6000 on the X4, 10752x5376 on the X5) and cx may be stored
    in those full-frame coordinates; reduce both to one lens.
    """
    spec = dict(spec)
    ref_width = spec.get("ref_width", 0.0)
    ref_height = spec.get("ref_height", 0.0)
    if ref_width <= 0 or ref_height <= 0:
        return spec
    if ref_width >= 2 * ref_height:
        ref_width /= 2
        spec["ref_width"] = ref_width
    if spec["cx"] >= ref_width:
        offset = ref_width * (spec["cx"] // ref_width)
        logger.info(
            f"Lens {lens_idx}: principal point cx={spec['cx']:.1f} is in "
            f"full-frame coordinates, shifting by -{offset:.0f}"
        )
        spec["cx"] -= offset
    return spec


def load_factory_calibration(insv_path, trailer_records: dict[int, bytes]) -> dict | None:
    """Load the factory lens calibration of an .insv recording, best effort.

    Prefers the .pb sidecar (more distortion terms) over the trailer's
    offset_v3. Returns {"source", "camera_type", "lenses": [spec, spec]}
    where each spec is in single-lens reference resolution (see
    mei_model_kwargs_for_stream), or None when no calibration is found.
    """
    metadata = parse_metadata_record(trailer_records.get(METADATA_RECORD_KEY, b""))

    lenses = None
    source = None
    pb_path = find_pb_sidecar(insv_path)
    if pb_path is not None:
        try:
            lenses = parse_pb_calibration(pb_path.read_bytes())
        except OSError as e:
            logger.warning(f"Could not read calibration sidecar {pb_path}: {e}")
        if lenses:
            source = str(pb_path)
        else:
            logger.warning(f"No calibration block found in sidecar {pb_path}")

    if not lenses and metadata.get("offset_v3"):
        lenses = parse_offset_v3(metadata["offset_v3"])
        if lenses:
            source = ".insv trailer offset_v3"

    if not lenses:
        return None
    if len(lenses) != 2:
        logger.warning(
            f"Factory calibration describes {len(lenses)} lenses, expected 2; ignoring"
        )
        return None

    window_crop = metadata.get("window_crop")
    normalized = []
    for lens_idx, spec in enumerate(lenses):
        spec = _normalize_lens_spec(spec, lens_idx)
        if window_crop:
            spec["crop_width"] = window_crop["dst_width"]
            spec["crop_height"] = window_crop["dst_height"]
        normalized.append(spec)

    return {
        "source": source,
        "camera_type": metadata.get("camera_type"),
        "lenses": normalized,
    }


def mei_model_kwargs_for_stream(spec: dict, stream_width: int, stream_height: int) -> dict:
    """Convert a reference-resolution lens spec to MeiLensModel kwargs.

    The demuxed stream is assumed to be a centered crop of the reference
    frame, uniformly scaled. The crop region comes from window_crop_info when
    present (e.g. 5312px of the X5's 5376px reference) and otherwise from
    aspect-fitting the stream into the reference frame (e.g. 6000x6000 out
    of the X4's 8000x6000). 0.5 converts the calibration's OpenCV pixel
    origin to the COLMAP convention used by the render pipeline.
    """
    ref_width = spec.get("ref_width", 0.0) or stream_width
    ref_height = spec.get("ref_height", 0.0) or stream_height

    crop_width = spec.get("crop_width", 0.0)
    crop_height = spec.get("crop_height", 0.0)
    crop_usable = (
        0 < crop_width <= ref_width
        and 0 < crop_height <= ref_height
        and abs(crop_width / crop_height - stream_width / stream_height) < 0.02
    )
    if not crop_usable:
        fit_scale = max(stream_width / ref_width, stream_height / ref_height)
        crop_width = stream_width / fit_scale
        crop_height = stream_height / fit_scale

    crop_x = (ref_width - crop_width) / 2
    crop_y = (ref_height - crop_height) / 2
    scale_x = stream_width / crop_width
    scale_y = stream_height / crop_height

    kwargs = {
        "xi": spec["xi"],
        "fx": spec["fx"] * scale_x,
        "fy": spec["fy"] * scale_y,
        "cx": (spec["cx"] - crop_x) * scale_x + 0.5,
        "cy": (spec["cy"] - crop_y) * scale_y + 0.5,
    }
    for key in ("k1", "k2", "k3", "k4", "p1", "p2", "s1", "s2", "s3", "s4"):
        kwargs[key] = spec.get(key, 0.0)
    if "fov_deg" in spec:
        kwargs["fov_deg"] = spec["fov_deg"]

    center_offset = max(
        abs(kwargs["cx"] - stream_width / 2) / stream_width,
        abs(kwargs["cy"] - stream_height / 2) / stream_height,
    )
    if center_offset > 0.1:
        logger.warning(
            f"Factory principal point ({kwargs['cx']:.1f}, {kwargs['cy']:.1f}) is "
            f"far from the {stream_width}x{stream_height} stream center; the "
            "reference-to-stream scaling may be wrong for this camera model"
        )
    return kwargs


def get_mount_corrections(lenses: list[dict]) -> list[tuple[float, float, float]]:
    """Per-lens (yaw, pitch, roll) mounting corrections in degrees.

    These are the factory-measured sub-degree deviations of each lens from
    its nominal orientation (lens 1 nominally 180 deg yaw from lens 0).
    """
    return [
        (
            spec.get("yaw_deg", 0.0),
            spec.get("pitch_deg", 0.0),
            spec.get("roll_deg", 0.0),
        )
        for spec in lenses
    ]
