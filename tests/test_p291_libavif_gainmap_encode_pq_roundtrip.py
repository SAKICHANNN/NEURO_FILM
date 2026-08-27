from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_p291_libavif_gainmap_encode_pq_roundtrip import (
    code_error,
    direct_bt709_pq_to_rec2020_pq_rgb16,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p291_libavif_gainmap_encode_pq_roundtrip_v1.json"


def test_code_error_is_exact_and_order_independent() -> None:
    first = np.array([[[0, 10, 65535], [9, 9, 9]]], dtype=np.uint16)
    second = np.array([[[1, 8, 65535], [9, 12, 6]]], dtype=np.uint16)
    assert code_error(first, second) == code_error(second, first)
    assert code_error(first, second) == {"maximum": 3.0, "median": 1.5, "p95": 3.0}


def test_direct_oracle_preserves_shape_type_and_black() -> None:
    values = np.zeros((2, 3, 3), dtype=np.uint16)
    output = direct_bt709_pq_to_rec2020_pq_rgb16(values)
    assert output.shape == values.shape
    assert output.dtype == np.uint16
    assert output.flags.c_contiguous
    assert np.count_nonzero(output) == 0


def test_p291_committed_integration_fails_closed_on_frozen_code_gate() -> None:
    report = run(CONFIG, reverse=False)
    assert report["status"] == "FAIL_CLOSED_LIBAVIF_GAINMAP_ENCODE_PQ_INTEGRATION"
    scientific = report["scientific"]
    assert scientific["gates"]["all-candidate-identities-exact"] is True
    assert scientific["gates"]["all-publication-gates-pass"] is True
    assert scientific["gates"]["all-error-bounds-pass"] is False
    assert scientific["error_gates"]["decoded-vs-direct-hdr"] is False
    assert scientific["error_gates"]["pq-vs-direct-hdr"] is False
    assert scientific["candidate_count"] == "2/3"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert scientific["media_sha256"] == config["fixture"]["media_sha256"]
