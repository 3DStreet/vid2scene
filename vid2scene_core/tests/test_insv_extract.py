import os
import struct

import numpy as np

import insv_extract
from insv_extract import (
    INSV_TRAILER_MAGIC,
    find_companion_lens_file,
    looks_like_fisheye_frame,
    read_insv_trailer_records,
    summarize_insv_metadata,
    _frame_interval,
    _trim_to_paired_frames,
)


def build_insv_trailer(records: list[tuple[int, bytes]]) -> bytes:
    """Build a synthetic trailer per the reverse-engineered layout.

    Records are laid out first-to-last; each record is its payload followed by
    a (uint16 id, uint32 size) descriptor. The final record's descriptor is
    the start of the 78-byte footer that ends with the 32-byte magic.
    """
    blob = b""
    for record_id, payload in records[:-1]:
        blob += payload + struct.pack("<HI", record_id, len(payload))
    last_id, last_payload = records[-1]
    blob += last_payload
    footer = struct.pack("<HI", last_id, len(last_payload)) + b"\x00" * 40 + INSV_TRAILER_MAGIC
    assert len(footer) == 78
    return blob + footer


def build_insv_trailer_v3(records: list[tuple[int, bytes]]) -> bytes:
    """Build a synthetic version-3 trailer (seen on X5 fw 1.9).

    Record payloads sit at the front of the trailer data region; an id-0
    record chained at the top holds an index table of 10-byte
    (uint16 id, uint32 size, uint32 offset) entries with offsets relative
    to the data start, and small table ids are the legacy ids >> 8. The
    footer carries the total trailer size and version 3.
    """
    data = b""
    entries = b"\x00" * 10  # leading padding slot, as in real trailers
    for record_id, payload in records:
        table_id = record_id if record_id == 0x101 else record_id >> 8
        entries += struct.pack("<HII", table_id, len(payload), len(data))
        data += payload
    body = data + entries
    trailer_size = len(body) + 78
    footer = (
        struct.pack("<HI", 0, len(entries))
        + b"\x00" * 32
        + struct.pack("<II", trailer_size, 3)
        + INSV_TRAILER_MAGIC
    )
    assert len(footer) == 78
    return body + footer


class TestTrailerParsing:
    def test_parses_records_walking_backwards(self, tmp_path):
        gyro = bytes(range(64))
        file_info = b"\x0a\x0cIAB123456789\x12\x0bInsta360 X4\x1a\x08v1.0.0.0"
        path = tmp_path / "video.insv"
        path.write_bytes(b"\x00" * 256 + build_insv_trailer([(0x300, gyro), (0x101, file_info)]))

        records = read_insv_trailer_records(path)
        assert records[0x101] == file_info
        assert records[0x300] == gyro

    def test_parses_v3_index_trailer(self, tmp_path):
        gyro = bytes(range(48))
        file_info = b"\x0a\x0eIAHYA2507QGEJ3\x12\x0bInsta360 X5\x1a\x08v1.9.6.0"
        path = tmp_path / "video.insv"
        path.write_bytes(
            b"\x00" * 256
            + build_insv_trailer_v3([(0x101, file_info), (0x300, gyro)])
        )

        records = read_insv_trailer_records(path)
        assert records[0x101] == file_info
        assert records[0x300] == gyro

    def test_missing_magic_returns_empty(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"\x00" * 512)
        assert read_insv_trailer_records(path) == {}

    def test_tiny_file_returns_empty(self, tmp_path):
        path = tmp_path / "video.insv"
        path.write_bytes(b"tiny")
        assert read_insv_trailer_records(path) == {}

    def test_walk_stops_at_file_body(self, tmp_path):
        # The body is zeros, so the walk terminates on a zero record id
        path = tmp_path / "video.insv"
        path.write_bytes(b"\x00" * 64 + build_insv_trailer([(0x400, b"\x01\x02")]))
        records = read_insv_trailer_records(path)
        assert records == {0x400: b"\x01\x02"}

    def test_summarize_extracts_model_strings(self):
        file_info = b"\x0a\x0cIAB123456789\x12\x0bInsta360 X4\x1a\x08v1.0.0.0"
        info = summarize_insv_metadata({0x101: file_info, 0x300: b"\x00"})
        assert "Insta360 X4" in info["file_info_strings"]
        assert "IAB123456789" in info["file_info_strings"]
        assert info["record_ids"] == {"0x101": "file_info", "0x300": "gyro"}


class TestCompanionFile:
    def test_finds_10_companion_of_00(self, tmp_path):
        first = tmp_path / "VID_20260101_120000_00_001.insv"
        second = tmp_path / "VID_20260101_120000_10_001.insv"
        first.touch()
        second.touch()
        assert find_companion_lens_file(first) == second
        assert find_companion_lens_file(second) == first

    def test_no_companion_returns_none(self, tmp_path):
        first = tmp_path / "VID_20260101_120000_00_001.insv"
        first.touch()
        assert find_companion_lens_file(first) is None

    def test_single_file_recording_returns_none(self, tmp_path):
        single = tmp_path / "VID_20260101_120000.insv"
        single.touch()
        assert find_companion_lens_file(single) is None


class TestFrameHelpers:
    def test_frame_interval(self):
        assert _frame_interval(1000, 100) == 10
        assert _frame_interval(100, 1000) == 1
        assert _frame_interval(1001, 100) == 11

    def test_trim_to_paired_frames(self, tmp_path):
        lens_dirs = [str(tmp_path / "lens0"), str(tmp_path / "lens1")]
        for d in lens_dirs:
            os.makedirs(d)
        for i in range(1, 6):
            (tmp_path / "lens0" / f"frame_{i:05d}.png").touch()
        for i in range(1, 4):
            (tmp_path / "lens1" / f"frame_{i:05d}.png").touch()

        assert _trim_to_paired_frames(lens_dirs) == 3
        assert sorted(os.listdir(lens_dirs[0])) == sorted(os.listdir(lens_dirs[1]))


class TestFisheyeHeuristic:
    def test_dark_corners_look_like_fisheye(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150] = 200  # bright circle-ish center, black corners
        assert looks_like_fisheye_frame(image)

    def test_full_frame_image_does_not(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert not looks_like_fisheye_frame(image)
