from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from src.inference.romm_rec2020_velvia_staged_native_v2 import (
    render_supported_prophoto_velvia_rec2020_staged_native_v2,
)
from src.inference.romm_rec2020_velvia_staged_streaming_v3 import (
    render_supported_prophoto_velvia_rec2020_staged_streaming_v3,
)
from src.preprocess import REC2020_SDR_CICP
from tests.test_u1_4c19_staged_prophoto_render import _write_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"


def _chunks(path: Path) -> dict[bytes, list[bytes]]:
    raw = path.read_bytes()
    result: dict[bytes, list[bytes]] = {}
    offset = 8
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        result.setdefault(kind, []).append(raw[offset + 8 : offset + 8 + length])
        offset += 12 + length
    return result


def test_streaming_rec2020_preserves_exact_samples_and_cicp(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    _write_source(source)
    render_supported_prophoto_velvia_rec2020_staged_native_v2(
        source,
        expected,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "expected_scratch",
        build_dir=tmp_path / "expected_build",
        row_chunk=13,
        thread_count=4,
    )
    first = render_supported_prophoto_velvia_rec2020_staged_streaming_v3(
        source,
        actual,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "actual_scratch",
        build_dir=tmp_path / "actual_build",
        row_chunk=13,
        thread_count=4,
    )
    expected_samples = cv2.imread(str(expected), cv2.IMREAD_UNCHANGED)
    actual_samples = cv2.imread(str(actual), cv2.IMREAD_UNCHANGED)
    assert expected_samples is not None and actual_samples is not None
    assert (actual_samples == expected_samples).all()
    chunks = _chunks(actual)
    assert chunks[b"cICP"] == [REC2020_SDR_CICP]
    assert b"iCCP" not in chunks
    assert first["output"]["sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()

    replay = tmp_path / "replay.png"
    second = render_supported_prophoto_velvia_rec2020_staged_streaming_v3(
        source,
        replay,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "replay_scratch",
        build_dir=tmp_path / "replay_build",
        row_chunk=13,
        thread_count=4,
    )
    assert replay.read_bytes() == actual.read_bytes()
    assert second == first
