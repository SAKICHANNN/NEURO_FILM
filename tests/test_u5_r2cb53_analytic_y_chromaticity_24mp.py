from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_u5_r2cb53_analytic_y_chromaticity_24mp import (
    _validate_contract,
    generate_fields,
    worker,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb53_analytic_y_chromaticity_24mp_resources_v1.json"


def test_cb53_contract_binds_unchanged_cb52_operator() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(config)
    assert config["workload"]["shape"] == [4000, 6000, 3]
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == 4 * 1024**3
    assert config["claim_ceiling"].startswith("Local Windows/Python 24MP")
    assert config["workload"]["worker_timeout_seconds"] == 180.0


def test_cb53_generator_is_exact_and_bounded() -> None:
    first = generate_fields((19, 31, 3), row_chunk=7)
    second = generate_fields((19, 31, 3), row_chunk=5)
    for left, right in zip(first, second, strict=True):
        assert left.dtype == np.float32
        assert np.array_equal(left, right)
        assert np.isfinite(left).all()
        assert left.min() >= 0.0
        assert left.max() <= 1.0


def test_cb53_small_worker_preserves_core_invariants() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    config["workload"]["shape"] = [96, 128, 3]
    result = worker(config)
    assert result["finite_and_bounded"] is True
    assert result["selected_facts"]["global_dose"] >= 0.5
    assert result["selected_facts"]["selected_new_boundary_fraction"] == 0.0
    assert result["selected_facts"]["selected_lstar_inversion_fraction"] == 0.0
    assert result["maximum_luminance_error"] <= 1e-6
