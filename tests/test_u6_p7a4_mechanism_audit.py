from __future__ import annotations

import json
from pathlib import Path

from src.eval.physical_joint_ablation import load_contracts
from src.eval.physical_joint_mechanism import _mechanism_controls


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p7a4_mechanism_controlled_severe_audit_v1.json"


def test_mechanism_controls_are_repeat_exact_and_finite_support() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / config["parent_audit"]).read_text())
    _, runtime = load_contracts(ROOT, parent)
    result = _mechanism_controls(config["synthetic_controls"], runtime)
    assert result["decisions"]["flat_invariance"] is True
    assert result["decisions"]["flat_full_vs_cheap"] is True
    assert result["decisions"]["repeat_exact"] is True
    assert result["decisions"]["finite_support"] is False
    assert result["impulse"]["inside_halo_max_abs"] > 0.0
    assert result["impulse"]["outside_halo_max_abs"] > 1e-12


def test_p7a4_parent_reports_are_exactly_bound() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (
        config["inherited_report_run_a_sha256"]
        == config["inherited_report_run_b_sha256"]
    )
    assert config["visual_refresh"]["randomize_each_image_independently"] is True
