from __future__ import annotations

from scripts.run_u6_p4g_film_structure_boundary_semantics import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_structure_boundary_semantics import (
    evaluate_structure_boundary_semantics,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4g_film_structure_boundary_semantics_v1.json"


def test_contract_is_exact_and_crop_boundary_only() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["boundary_candidates"]["candidate"] == (
        "normalized-support-v1"
    )
    assert contract["photograph_access_allowed"] is False
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_normalized_support_closes_on_frozen_variance_gate() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = evaluate_structure_boundary_semantics(contract, parent)
    assert report["automatic_pass"] is False
    assert report["checks"]["reference_variance"] is False
    assert all(
        value
        for name, value in report["checks"].items()
        if name != "reference_variance"
    )
    assert (
        report[
            "incumbent_zero_fill_factor_2_maximum_reference_local_mean_difference"
        ]
        >= 0.03
    )
