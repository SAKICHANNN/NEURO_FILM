from __future__ import annotations

from scripts.run_u6_p4j_adaptive_exact_area_fallback import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_adaptive_exact_area_fallback import (
    evaluate_adaptive_exact_area_fallback,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4j_adaptive_exact_area_fallback_v1.json"


def test_contract_is_exact_and_not_sparse_product_claim() -> None:
    contract, parent, p4i = load_contract(
        ROOT, CONFIG, CONFIG_SHA256
    )
    assert contract["router"]["sparse_product_executor_claim_allowed"] is False
    assert contract["photograph_access_allowed"] is False
    assert p4i["node"] == "U6.P4I"
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_adaptive_exact_area_fallback_closes_on_frozen_gates() -> None:
    contract, parent, p4i = load_contract(
        ROOT, CONFIG, CONFIG_SHA256
    )
    report = evaluate_adaptive_exact_area_fallback(contract, parent, p4i)
    assert report["automatic_pass"] is False
    assert {
        name for name, value in report["checks"].items() if not value
    } == {"density_map_correlation", "display_noise_negative"}
    assert report["checks"]["region_mean"] is True
    assert report["checks"]["region_variance"] is True
    assert report["checks"]["checkerboard_contrast"] is True
    assert report["checks"]["fallback_mean"] is True
