from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocess.libavif_gainmap_encoder import (
    LIBAVIF_GAINMAP_ENCODER_PROFILE_ID,
    LibavifGainMapEncoderError,
    encode_gainmap_avif_create_only_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    hdr = tmp_path / "hdr.png"
    sdr = tmp_path / "sdr.png"
    runtime = tmp_path / "avifgainmaputil.exe"
    decoder = tmp_path / "avifdec.exe"
    hdr_pixels = np.full((2, 3, 3), 50000, dtype=np.uint16)
    sdr_pixels = np.full((2, 3, 3), 12000, dtype=np.uint16)
    assert cv2.imwrite(str(hdr), hdr_pixels)
    assert cv2.imwrite(str(sdr), sdr_pixels)
    runtime.write_bytes(b"runtime")
    decoder.write_bytes(b"decoder")
    return {
        "hdr_endpoint": hdr.resolve(),
        "sdr_endpoint": sdr.resolve(),
        "destination": (tmp_path / "output.avif").resolve(),
        "expected_hdr_sha256": _sha256(hdr),
        "expected_sdr_sha256": _sha256(sdr),
        "runtime": runtime.resolve(),
        "expected_runtime_sha256": _sha256(runtime),
        "decoder": decoder.resolve(),
        "expected_decoder_sha256": _sha256(decoder),
        "base_cicp": (1, 16, 0),
        "alternate_cicp": (1, 13, 0),
        "base_headroom": 1.3,
        "alternate_headroom": 0.0,
    }


def test_profile_identifier_is_versioned() -> None:
    assert LIBAVIF_GAINMAP_ENCODER_PROFILE_ID.endswith(".v1")


def test_wrong_endpoint_hash_rejects_before_execution(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    arguments["expected_hdr_sha256"] = "0" * 64
    with pytest.raises(LibavifGainMapEncoderError, match="endpoint identity"):
        encode_gainmap_avif_create_only_v1(**arguments)  # type: ignore[arg-type]
    assert not Path(arguments["destination"]).exists()


def test_wrong_profile_rejects_before_execution(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    arguments["base_headroom"] = 1.2
    with pytest.raises(LibavifGainMapEncoderError, match="profile identity"):
        encode_gainmap_avif_create_only_v1(**arguments)  # type: ignore[arg-type]
    assert not Path(arguments["destination"]).exists()


def test_foreign_destination_rejects_without_overwrite(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    destination = Path(arguments["destination"])
    destination.write_bytes(b"foreign\n")
    with pytest.raises(LibavifGainMapEncoderError, match="path preflight"):
        encode_gainmap_avif_create_only_v1(**arguments)  # type: ignore[arg-type]
    assert destination.read_bytes() == b"foreign\n"


def test_equal_endpoints_reject_before_execution(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    sdr = Path(arguments["sdr_endpoint"])
    hdr = Path(arguments["hdr_endpoint"])
    sdr.write_bytes(hdr.read_bytes())
    arguments["expected_sdr_sha256"] = _sha256(sdr)
    with pytest.raises(LibavifGainMapEncoderError, match="not distinct"):
        encode_gainmap_avif_create_only_v1(**arguments)  # type: ignore[arg-type]
