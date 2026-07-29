from __future__ import annotations

from scripts.run_u6_p4h_two_cumulant_lod import CONFIG_SHA256, ROOT
from src.eval.physical_two_cumulant_lod import (
    evaluate_two_cumulant_lod,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4h_two_cumulant_lod_v1.json"


def test_contract_is_exact_and_has_no_empirical_fit() -> None:
    contract, parent, decision = load_contract(
        ROOT, CONFIG, CONFIG_SHA256
    )
    assert contract["compiler"]["empirical_parameter_fit_allowed"] is False
    assert contract["photograph_access_allowed"] is False
    assert decision["results"]["failed_gates"] == ["reference_variance"]
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_two_cumulant_compiler_passes_frozen_lod_gates() -> None:
    contract, parent, decision = load_contract(
        ROOT, CONFIG, CONFIG_SHA256
    )
    report = evaluate_two_cumulant_lod(contract, parent, decision)
    assert report["automatic_pass"] is True
    assert all(report["checks"].values())
