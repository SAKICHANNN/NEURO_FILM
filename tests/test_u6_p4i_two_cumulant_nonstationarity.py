from __future__ import annotations

from scripts.run_u6_p4i_two_cumulant_nonstationarity import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_two_cumulant_nonstationarity import (
    evaluate_two_cumulant_nonstationarity,
    load_contract,
)


CONFIG = (
    ROOT / "configs/u6_p4i_two_cumulant_nonstationarity_stress_v1.json"
)


def test_contract_is_exact_and_synthetic_only() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["compiler"]["family"] == "P4H analytic two-cumulant"
    assert contract["photograph_access_allowed"] is False
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_two_cumulant_nonstationarity_closes_on_frozen_gates() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = evaluate_two_cumulant_nonstationarity(contract, parent)
    assert report["automatic_pass"] is False
    assert {
        name for name, value in report["checks"].items() if not value
    } == {
        "density_map_correlation",
        "region_mean",
        "region_variance",
        "checkerboard_contrast",
        "display_noise_negative",
    }
    assert report["checks"]["step_center"] is True
    assert report["checks"]["island_centroid"] is True
