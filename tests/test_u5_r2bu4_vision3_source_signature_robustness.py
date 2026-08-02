from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vision3_source_signature_robustness import (
    SourceSignatureRobustnessError,
    audit_source_signature,
    canonical_json,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu4_vision3_source_signature_robustness_v1.json"


def test_contract_rejects_post_score_grid_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["comparison"]["mtf_frequencies_cycles_per_mm"][-1] = 62.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SourceSignatureRobustnessError, match="contract drift"):
        load_contract(changed)


def test_audit_is_exact_and_non_renderable() -> None:
    config = load_contract(CONFIG)
    first = audit_source_signature(config, ROOT)
    second = audit_source_signature(config, ROOT)
    assert canonical_json(first) == canonical_json(second)
    assert first["case_count"] == 60
    assert first["gate_results"]["exact_bundle_recompile"] is True
    assert first["gate_results"]["no_render_authority"] is True


def test_audit_applies_all_frozen_controls() -> None:
    report = audit_source_signature(load_contract(CONFIG), ROOT)
    assert {row["perturbation"] for row in report["rows"]} == {
        "nominal",
        "all_y_plus",
        "all_y_minus",
        "alternating_xy",
        "inverse_alternating_xy",
    }
    assert {tuple(row["channels"]) for row in report["rows"]} == {
        ("blue", "green", "red"),
        ("blue", "green"),
        ("blue", "red"),
        ("green", "red"),
    }
    assert report["cyclic_label_control_accuracy"] <= 1.0 / 3.0
