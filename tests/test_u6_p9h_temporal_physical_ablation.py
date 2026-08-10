from __future__ import annotations

from pathlib import Path

from src.eval.temporal_physical_ablation import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9h_temporal_physical_ablation_v1.json"


def test_formal_temporal_physical_ablation_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["effect_switch_count"] == 3
    assert set(report["measurements"]["effect_delta_rms"]) == {
        "exposure_flicker",
        "density_grain",
        "gate_weave",
    }


def test_contract_preserves_three_typed_stages() -> None:
    contract = load_contract(CONTRACT)
    order = contract["physical_order"]
    assert order.index("temporal_exposure_multiplier") < order.index(
        "characteristic_development"
    )
    assert order.index("density_conditioned_grain") < order.index(
        "scanner_gate_weave_sampling"
    )
