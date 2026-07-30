from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from src.color_match.evaluation import KnownOperatorSampleMetrics


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_canoncgt_reference_match.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_canoncgt_reference_match",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_canoncgt_contract_is_external_evaluation_only() -> None:
    config = json.loads(
        (
            ROOT / "configs" / "reference_match_canoncgt_external_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["external"]["license"] == "Apache-2.0"
    assert config["external"]["parameter_count"] == 5_056_383
    assert config["external"]["lut_size"] == 17
    assert config["execution"] == {
        "official_code_unmodified": True,
        "official_weights_unmodified": True,
        "evaluation_only": True,
        "targets_used_only_after_render": True,
        "product_integration_allowed": False,
        "output_rgb_is_not_retained": True,
    }


def test_srgb_linearization_is_exact_at_breakpoint_and_endpoints() -> None:
    encoded = np.asarray([0.0, 0.04045, 1.0], dtype=np.float32)
    linear = MODULE._linearize_srgb(encoded)
    assert linear.dtype == np.float32
    assert linear[0] == 0.0
    assert np.isclose(linear[1], 0.04045 / 12.92, atol=1e-8)
    assert np.isclose(linear[2], 1.0, atol=1e-7)


def test_mode_summary_requires_every_independent_gate() -> None:
    gates = {
        "minimum_improved_cross_content_fraction": 0.75,
        "minimum_median_improvement_fraction": 0.25,
        "minimum_worst_improvement_fraction": -0.1,
        "maximum_new_boundary_fraction": 0.005,
        "maximum_preclip_out_of_gamut_fraction": 0.005,
    }
    rows = [
        KnownOperatorSampleMetrics("a", 1.0, 1.0, 0.5, 1.0, 0.5, 0.0),
        KnownOperatorSampleMetrics("b", 1.0, 1.0, 0.6, 1.0, 0.4, 0.0),
        KnownOperatorSampleMetrics("c", 1.0, 1.0, 0.7, 1.0, 0.3, 0.0),
        KnownOperatorSampleMetrics("d", 1.0, 1.0, 1.05, 1.0, -0.05, 0.0),
    ]
    summary = MODULE._mode_summary(rows, [0.0] * 4, gates)
    assert summary["passes"] is True
    failing = [
        *rows[:-1],
        KnownOperatorSampleMetrics(
            "d",
            1.0,
            1.0,
            1.05,
            1.0,
            -0.05,
            0.006,
        ),
    ]
    assert MODULE._mode_summary(
        failing,
        [0.0] * 4,
        gates,
    )["passes"] is False
