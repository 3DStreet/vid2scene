import base64
import logging

import numpy as np
import pytest

import insv_calibration
from insv_calibration import (
    METADATA_RECORD_KEY,
    load_factory_calibration,
    mei_model_kwargs_for_stream,
    parse_metadata_record,
    parse_offset_v3,
    parse_pb_calibration,
)


# ---------------------------------------------------------------------------
# Protobuf encoding helpers (test-side mirror of the wire format)
# ---------------------------------------------------------------------------

def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _string_field(number: int, text: str) -> bytes:
    payload = text.encode("utf-8")
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _varint_field(number: int, value: int) -> bytes:
    return _varint((number << 3) | 0) + _varint(value)


def _bytes_field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _fixed64_field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 1) + payload


def make_window_crop(src_w, src_h, dst_w, dst_h) -> bytes:
    return b"".join(
        _varint_field(i + 1, v) for i, v in enumerate((src_w, src_h, dst_w, dst_h))
    )


# X4-like offset_v3: 1 count + 2 lenses x 20 fields
# (xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 w h type flag)
OFFSET_V3_LENS0 = [1.9, 2870.0, 2871.0, 3987.5, 3012.7, 0.1, -0.2, 0.05,
                   0.0, 0.0, 0.0, 0.21, 1.5, -1.3, -0.0005, -0.0013,
                   16000.0, 6000.0, 2.0, 1.0]
OFFSET_V3_LENS1 = [1.9, 2868.0, 2869.0, 3990.1, 3010.2, -0.15, 0.3, -0.02,
                   0.0, 0.0, 0.0, 0.22, 1.4, -1.2, -0.0004, -0.0011,
                   16000.0, 6000.0, 2.0, 1.0]
OFFSET_V3_TEXT = "_".join(
    f"{v:.6f}" for v in [2.0] + OFFSET_V3_LENS0 + OFFSET_V3_LENS1
)

# X4 fw 1.9.21 offset_v3: also the 19-field layout (the 20-field X4 fixture
# above matches older firmware). Verbatim from a real X4 recording; per-lens
# landscape reference frame (8000x6000, no halving) with lens 1's cx in
# 16000-wide full dual-frame coordinates.
OFFSET_V3_X4_TEXT = (
    "2_1.948170_4613.480_4613.680_4007.080_2999.740_-0.045_0.017_89.764"
    "_0.000000_0.000000_0.000000_0.37750375_1.42040467_-4.14384651"
    "_0.00027564_-0.00161791_8000_6000_71"
    "_1.948170_4632.750_4632.940_12009.280_3005.090_0.015_0.177_89.567"
    "_-0.001166_-0.000030_-0.032311_0.39006260_1.31805623_-4.07983112"
    "_0.00039543_-0.00112667_8000_6000_71_197632"
)

# X5 fw 1.9 offset_v3: 19 fields per lens (no per-lens flag) plus one
# trailing file-level value. Verbatim from a real X5 recording
# (fw v1.9.6_build1); note the ~90 deg nominal roll of the portrait-mounted
# sensors and lens 1's cx in full dual-frame coordinates.
OFFSET_V3_X5_TEXT = (
    "2_2.000000_4280.730_4280.890_2693.330_2680.260_-0.467_0.351_90.426"
    "_0.000000_0.000000_0.000000_0.18062226_2.09071612_-3.32824707"
    "_-0.00253176_-0.00075190_10752_5376_113"
    "_2.000000_4278.380_4278.480_8069.920_2688.010_0.560_0.096_89.252"
    "_-0.001068_0.000384_-0.032249_0.17816089_2.12469268_-3.33294296"
    "_-0.00017225_-0.00264012_10752_5376_113_197632"
)


def make_metadata_payload(offset_v3: str | None = OFFSET_V3_TEXT,
                          window_crop: bytes | None = None) -> bytes:
    payload = (
        _string_field(1, "XAS1234567890")
        + _string_field(2, "Insta360 X5")
        + _string_field(3, "v1.0.0")
        + _varint_field(7, 1718000000)  # creation_time, must be skipped
        + _fixed64_field(25, b"\x00" * 8)  # rolling_shutter_time, skipped
        + _bytes_field(11, b"\x01\x02\x03\xff")  # gps blob, skipped
    )
    if offset_v3 is not None:
        payload += _string_field(54, offset_v3)
    if window_crop is not None:
        payload += _bytes_field(27, window_crop)
    return payload


class TestParseMetadataRecord:
    def test_extracts_strings_and_window_crop(self):
        payload = make_metadata_payload(
            window_crop=make_window_crop(5376, 5376, 5312, 5312)
        )
        info = parse_metadata_record(payload)
        assert info["serial_number"] == "XAS1234567890"
        assert info["camera_type"] == "Insta360 X5"
        assert info["fw_version"] == "v1.0.0"
        assert info["offset_v3"] == OFFSET_V3_TEXT
        assert info["window_crop"] == {
            "src_width": 5376, "src_height": 5376,
            "dst_width": 5312, "dst_height": 5312,
        }

    def test_garbage_returns_empty(self):
        assert parse_metadata_record(b"\xff\xff\xff\xff") == {}
        assert parse_metadata_record(b"") == {}


class TestParseOffsetV3:
    def test_parses_two_lens_blocks(self):
        lenses = parse_offset_v3(OFFSET_V3_TEXT)
        assert len(lenses) == 2
        lens0 = lenses[0]
        assert lens0["xi"] == pytest.approx(1.9)
        assert lens0["fx"] == pytest.approx(2870.0)
        assert lens0["cx"] == pytest.approx(3987.5)
        assert lens0["yaw_deg"] == pytest.approx(0.1)
        assert lens0["roll_deg"] == pytest.approx(0.05)
        assert lens0["k1"] == pytest.approx(0.21)
        assert lens0["p2"] == pytest.approx(-0.0013)
        assert lens0["ref_width"] == pytest.approx(16000.0)
        assert lenses[1]["fx"] == pytest.approx(2868.0)

    def test_parses_x5_19_field_blocks(self):
        lenses = parse_offset_v3(OFFSET_V3_X5_TEXT)
        assert len(lenses) == 2
        lens0, lens1 = lenses
        assert lens0["xi"] == pytest.approx(2.0)
        assert lens0["fx"] == pytest.approx(4280.730)
        assert lens0["roll_deg"] == pytest.approx(90.426)
        assert lens0["ref_width"] == pytest.approx(10752)
        assert lens0["ref_height"] == pytest.approx(5376)
        assert "flag" not in lens0
        assert lens1["fx"] == pytest.approx(4278.380)
        assert lens1["cx"] == pytest.approx(8069.920)
        assert lens1["lens_type"] == pytest.approx(113)

    def test_parses_x4_fw19_19_field_blocks(self):
        lenses = parse_offset_v3(OFFSET_V3_X4_TEXT)
        assert len(lenses) == 2
        assert lenses[0]["xi"] == pytest.approx(1.94817)
        assert lenses[0]["fx"] == pytest.approx(4613.480)
        assert lenses[0]["roll_deg"] == pytest.approx(89.764)
        assert lenses[0]["ref_width"] == pytest.approx(8000)
        assert lenses[0]["ref_height"] == pytest.approx(6000)
        assert "flag" not in lenses[0]
        assert lenses[1]["cx"] == pytest.approx(12009.280)

    def test_rejects_truncated_or_junk(self):
        assert parse_offset_v3("2_1.0_2.0_3.0") is None
        assert parse_offset_v3("not_numbers_at_all") is None
        assert parse_offset_v3("") is None


def make_pb_lens_block(fx, fy, cx, cy, yaw, pitch) -> list[float]:
    # 27 fields: xi fx fy cx cy yaw pitch field7 tx ty tz
    # k1 k2 k3 k4 zero p1 p2 s1 s2 s3 s4 tauX tauY ref_w ref_h type
    return [2.0, fx, fy, cx, cy, yaw, pitch, 89.816, 0.0, 0.0, 0.0,
            0.2199, 1.6416, -1.4439, -2.6206, 0.0, -0.000536, -0.001321,
            -0.00148, 0.0008, 0.00204, 0.00194, 0.02581, 0.00293,
            10752.0, 5376.0, 113.0]


def make_pb_sidecar_bytes() -> bytes:
    lens_blocks = (
        make_pb_lens_block(4271.09, 4272.20, 2680.96, 2680.49, -0.132, 0.434)
        + make_pb_lens_block(4268.30, 4269.10, 5376.0 + 2682.10, 2679.80, 0.090, -0.210)
    )
    # Real sidecars store the string as "2_<lens0 fields>_<lens1 fields>"
    # with an integer lens count.
    calibration = "2_" + "_".join(f"{v:.6f}" for v in lens_blocks)
    blob = base64.b64encode(f"junk-prefix {calibration} junk-suffix".encode("latin-1"))
    return b"\x0a\x10binary-protobuf" + blob + b"\x00\x01trailing"


class TestParsePbCalibration:
    def test_parses_extended_calibration(self):
        lenses = parse_pb_calibration(make_pb_sidecar_bytes())
        assert lenses is not None and len(lenses) == 2
        lens0 = lenses[0]
        assert lens0["xi"] == pytest.approx(2.0)
        assert lens0["fx"] == pytest.approx(4271.09)
        assert lens0["cx"] == pytest.approx(2680.96)
        assert lens0["yaw_deg"] == pytest.approx(-0.132)
        assert lens0["pitch_deg"] == pytest.approx(0.434)
        assert lens0["roll_deg"] == 0.0  # ambiguous field 7 is not used as roll
        assert lens0["k4"] == pytest.approx(-2.6206)
        assert lens0["p1"] == pytest.approx(-0.000536)
        assert lens0["s4"] == pytest.approx(0.00194)
        assert lens0["ref_width"] == pytest.approx(10752.0)

    def test_no_calibration_block(self):
        assert parse_pb_calibration(b"no base64 here") is None
        blob = base64.b64encode(b"x" * 200)
        assert parse_pb_calibration(blob) is None


class TestMeiModelKwargs:
    def test_x5_like_with_window_crop(self):
        # X5: 5376px per-lens reference, encoded video is a centered 5312px
        # crop scaled to 3840. fx scales by 3840/5312, cx shifts by the crop.
        spec = {
            "xi": 2.0, "fx": 4271.09, "fy": 4272.20, "cx": 2680.96, "cy": 2680.49,
            "k1": 0.2199, "ref_width": 5376.0, "ref_height": 5376.0,
            "crop_width": 5312, "crop_height": 5312,
        }
        kwargs = mei_model_kwargs_for_stream(spec, 3840, 3840)
        scale = 3840 / 5312
        assert kwargs["fx"] == pytest.approx(4271.09 * scale)
        assert kwargs["cx"] == pytest.approx((2680.96 - 32) * scale + 0.5)
        assert kwargs["cy"] == pytest.approx((2680.49 - 32) * scale + 0.5)
        assert kwargs["k1"] == pytest.approx(0.2199)
        assert kwargs["k4"] == 0.0
        # Lands near the stream center, as a principal point must
        assert abs(kwargs["cx"] - 1920) < 20

    def test_x4_like_cover_fit_without_crop_info(self):
        # X4: 8000x6000 per-lens reference (4:3 sensor), square 3840 video
        # -> centered 6000x6000 crop scaled by 0.64.
        spec = {
            "xi": 1.9, "fx": 2870.0, "fy": 2871.0, "cx": 3987.5, "cy": 3012.7,
            "ref_width": 8000.0, "ref_height": 6000.0,
        }
        kwargs = mei_model_kwargs_for_stream(spec, 3840, 3840)
        assert kwargs["fx"] == pytest.approx(2870.0 * 0.64)
        assert kwargs["cx"] == pytest.approx((3987.5 - 1000) * 0.64 + 0.5)
        assert kwargs["cy"] == pytest.approx(3012.7 * 0.64 + 0.5)
        assert abs(kwargs["cx"] - 1920) < 20
        assert abs(kwargs["cy"] - 1920) < 20

    def test_mismatched_crop_aspect_falls_back_to_cover_fit(self):
        spec = {
            "xi": 2.0, "fx": 4000.0, "fy": 4000.0, "cx": 2688.0, "cy": 2688.0,
            "ref_width": 5376.0, "ref_height": 5376.0,
            "crop_width": 5312, "crop_height": 2656,  # 2:1, not the stream's 1:1
        }
        kwargs = mei_model_kwargs_for_stream(spec, 3840, 3840)
        assert kwargs["fx"] == pytest.approx(4000.0 * 3840 / 5376)

    def test_fov_override_passes_through(self):
        spec = {
            "xi": 2.0, "fx": 4000.0, "fy": 4000.0, "cx": 2688.0, "cy": 2688.0,
            "ref_width": 5376.0, "ref_height": 5376.0, "fov_deg": 195.0,
        }
        assert mei_model_kwargs_for_stream(spec, 3840, 3840)["fov_deg"] == 195.0


class TestLoadFactoryCalibration:
    def test_from_trailer_offset_v3(self, tmp_path):
        records = {
            METADATA_RECORD_KEY: make_metadata_payload(
                window_crop=make_window_crop(6000, 6000, 6000, 6000)
            )
        }
        factory = load_factory_calibration(tmp_path / "video.insv", records)
        assert factory is not None
        assert factory["source"] == ".insv trailer offset_v3"
        assert factory["camera_type"] == "Insta360 X5"
        assert len(factory["lenses"]) == 2
        lens0 = factory["lenses"][0]
        # 16000x6000 full dual frame is normalized to one 8000x6000 lens
        assert lens0["ref_width"] == pytest.approx(8000.0)
        assert lens0["crop_width"] == 6000
        assert lens0["cx"] == pytest.approx(3987.5)

    def test_prefers_pb_sidecar(self, tmp_path):
        insv_path = tmp_path / "video.insv"
        insv_path.write_bytes(b"")
        (tmp_path / "video.insv.pb").write_bytes(make_pb_sidecar_bytes())
        records = {METADATA_RECORD_KEY: make_metadata_payload()}
        factory = load_factory_calibration(insv_path, records)
        assert factory is not None
        assert factory["source"].endswith("video.insv.pb")
        lens1 = factory["lenses"][1]
        # Full-frame cx of the second lens is shifted into lens-local coords
        assert lens1["cx"] == pytest.approx(2682.10)
        assert lens1["ref_width"] == pytest.approx(5376.0)

    def test_no_calibration_returns_none(self, tmp_path):
        records = {METADATA_RECORD_KEY: make_metadata_payload(offset_v3=None)}
        assert load_factory_calibration(tmp_path / "video.insv", records) is None
        assert load_factory_calibration(tmp_path / "video.insv", {}) is None

    # The pipeline fails an .insv job hard when calibration cannot be parsed,
    # so these warnings are the debugging surface — each rung of the ladder
    # must say which step broke and carry the raw data needed to fix it.
    def test_logs_which_rung_broke(self, tmp_path, caplog):
        path = tmp_path / "video.insv"
        with caplog.at_level(logging.WARNING):
            load_factory_calibration(path, {})
        assert "No .insv trailer records" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            load_factory_calibration(path, {0x300: b"\x00"})
        assert "no file_info record" in caplog.text
        assert "0x300" in caplog.text

        caplog.clear()
        records = {METADATA_RECORD_KEY: make_metadata_payload(offset_v3=None)}
        with caplog.at_level(logging.WARNING):
            load_factory_calibration(path, records)
        assert "no offset_v3" in caplog.text

    def test_logs_raw_string_when_offset_v3_unparseable(self, tmp_path, caplog):
        bad = "2_1.0_2.0_3.0"
        records = {METADATA_RECORD_KEY: make_metadata_payload(offset_v3=bad)}
        with caplog.at_level(logging.WARNING):
            assert load_factory_calibration(tmp_path / "video.insv", records) is None
        assert "unparseable" in caplog.text
        assert bad in caplog.text  # verbatim — it's the fix-enabling data


class TestMountCorrections:
    def test_extracted_from_lens_specs(self):
        lenses = parse_offset_v3(OFFSET_V3_TEXT)
        corrections = insv_calibration.get_mount_corrections(lenses)
        assert corrections[0] == pytest.approx((0.1, -0.2, 0.05))
        assert corrections[1] == pytest.approx((-0.15, 0.3, -0.02))

    def test_rig_rotations_with_corrections(self):
        from fisheye_projection import get_lens_from_rig_rotations

        # Identity corrections reproduce the nominal back-to-back rig
        nominal = get_lens_from_rig_rotations()
        with_zero = get_lens_from_rig_rotations(
            mount_corrections=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        )
        for a, b in zip(nominal, with_zero):
            np.testing.assert_allclose(a, b, atol=1e-12)

        # A small yaw correction tilts the lens optical axis by that angle
        corrected = get_lens_from_rig_rotations(
            mount_corrections=[(0.5, 0.0, 0.0), (0.0, 0.0, 0.0)]
        )
        axis_in_rig = np.array([0.0, 0.0, 1.0]) @ corrected[0]
        angle = np.rad2deg(np.arccos(np.clip(axis_in_rig @ [0, 0, 1], -1, 1)))
        assert angle == pytest.approx(0.5, abs=1e-9)
        # Lens 1 still points backward
        back_axis = np.array([0.0, 0.0, 1.0]) @ corrected[1]
        assert back_axis[2] == pytest.approx(-1.0)
