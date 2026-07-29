from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fresh_native_standard_confirmation import (
    ARMS,
    boundary_metrics,
    validate_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json"
)


def test_p8bp_preflight_is_hash_bound_and_comparison_is_fixed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = validate_preflight(ROOT, config)
    assert len(manifest) == 9
    assert tuple(config["comparison"]["arms"]) == ARMS
    assert config["comparison"]["automatic_gate"][
        "maximum_new_boundary_fraction_vs_ao6"
    ] == 0.0


def test_boundary_metric_counts_only_new_candidate_boundaries() -> None:
    reference = np.full((2, 2, 3), 0.5, dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0, 0] = 0.0
    metrics = boundary_metrics(candidate, reference)
    assert metrics["output_code_boundary_fraction"] == 0.25
    assert metrics["new_boundary_fraction_vs_ao6"] == 0.25
    reference[0, 0, 1] = 1.0
    metrics = boundary_metrics(candidate, reference)
    assert metrics["new_boundary_fraction_vs_ao6"] == 0.0
