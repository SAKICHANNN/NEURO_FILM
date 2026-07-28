from __future__ import annotations

import json
from pathlib import Path

from src.eval.physical_joint_source_context import (
    evaluate_mechanism_smoke,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p7b_source_context_joint_ablation_v1.json"


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7b_contract_binds_closed_p7a4_and_runtime_parent() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    assert runtime_parent["node"] == "U6.P7A3"


def test_source_context_mechanism_smoke_passes_frozen_controls() -> None:
    result = evaluate_mechanism_smoke(root=ROOT, config=_config())
    assert result["automatic_pass"] is True
    assert result["decisions"] == {
        "colour_only_equivalence": True,
        "flat_invariance": True,
        "flat_full_vs_cheap": True,
        "finite_support": True,
        "repeat_exact": True,
    }
    assert result["impulse"]["outside_halo_max_abs"] <= 1e-12
