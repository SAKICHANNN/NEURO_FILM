from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.acquire_p269_hdrplus_one_burst import (
    P269Error,
    _canonical_bytes,
    _input_dng_probe,
    _safe_relative,
    _source_url,
    _verify_local,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p269_hdrplus_one_burst_acquisition_v1.json"


def test_p269_manifest_is_exact_and_bounded() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FORMAL_EXECUTION_LOCKED_AFTER_BOUNDED_PROBE_CORRECTION"
    assert len(config["objects"]) == 24
    assert sum(item["bytes"] for item in config["objects"]) == 256_482_210
    assert config["expected_total_bytes"] < config["maximum_total_bytes"]
    assert config["budgets"]["maximum_result_pixel_decodes"] == 0
    assert config["budgets"]["maximum_fit_train_inference_score_reads"] == 0


def test_p269_paths_fail_closed() -> None:
    assert _safe_relative("burst", "payload_N000.dng") == Path("burst/payload_N000.dng")
    for unsafe in ("../x.dng", "sub/x.dng", "C:/x.dng", ""):
        with pytest.raises(P269Error, match="unsafe object path"):
            _safe_relative("burst", unsafe)
    with pytest.raises(P269Error, match="unsafe role"):
        _safe_relative("../burst", "x.dng")


def test_p269_generation_pinned_url() -> None:
    url = _source_url(
        "gs://hdrplusdata/20171106_subset/bursts/id/", "payload_N000.dng", "123"
    )
    assert url == (
        "https://storage.googleapis.com/hdrplusdata/"
        "20171106_subset/bursts/id/payload_N000.dng?generation=123"
    )


def test_p269_local_identity_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    item = {
        "role": "burst",
        "path": "x.bin",
        "bytes": 3,
        "md5_base64": base64.b64encode(
            hashlib.md5(b"abc", usedforsecurity=False).digest()
        ).decode("ascii"),
    }
    assert _verify_local(path, item)["sha256"] == hashlib.sha256(b"abc").hexdigest()
    item["bytes"] = 4
    with pytest.raises(P269Error, match="local object identity mismatch"):
        _verify_local(path, item)


def test_p269_canonical_identity_is_order_independent() -> None:
    left = [{"role": "a", "path": "x"}, {"role": "b", "path": "y"}]
    right = list(reversed(left))
    left.sort(key=lambda row: (row["role"], row["path"]))
    right.sort(key=lambda row: (row["role"], row["path"]))
    assert _canonical_bytes(left) == _canonical_bytes(right)


def test_p269_probe_reads_only_requested_row_ranges(tmp_path: Path) -> None:
    values = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
    path = tmp_path / "probe.dng"
    tifffile.imwrite(path, values, rowsperstrip=1, metadata=None)
    probe = _input_dng_probe(path, 4)
    expected = np.ascontiguousarray(values[4:8, 6:10])
    assert probe["probe_shape"] == [4, 4]
    assert probe["file_payload_bytes_read"] == 32
    assert probe["full_raw_plane_decoded"] is False
    assert (
        probe["probe_u16le_sha256"]
        == hashlib.sha256(expected.astype("<u2").tobytes()).hexdigest()
    )
