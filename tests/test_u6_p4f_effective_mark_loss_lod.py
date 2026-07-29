from __future__ import annotations

from scripts.run_u6_p4f_effective_mark_loss_lod import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_effective_mark_loss_lod import (
    evaluate_effective_mark_loss_lod,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4f_effective_mark_loss_lod_v1.json"


def test_contract_is_exact_and_analytic_only() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["compiler"]["empirical_parameter_fit_allowed"] is False
    assert contract["photograph_access_allowed"] is False
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_effective_mark_loss_closes_on_reference_mean_gate() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = evaluate_effective_mark_loss_lod(contract, parent)
    assert report["automatic_pass"] is False
    assert report["checks"]["reference_local_mean"] is False
    assert all(
        value
        for name, value in report["checks"].items()
        if name != "reference_local_mean"
    )
