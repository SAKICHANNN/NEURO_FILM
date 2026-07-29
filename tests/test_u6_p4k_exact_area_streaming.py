from __future__ import annotations

from scripts.run_u6_p4k_exact_area_streaming import CONFIG_SHA256, ROOT
from src.eval.physical_exact_area_streaming import (
    evaluate_exact_area_streaming,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4k_exact_area_streaming_v1.json"


def test_contract_is_exact_reference_only() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["executor"]["full_expanded_target_allowed"] is False
    assert contract["photograph_access_allowed"] is False
    assert parent["model"]["input_domain"] == "developed_optical_density"


def test_exact_area_region_partitions_are_byte_exact() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = evaluate_exact_area_streaming(
        contract, parent, include_stream_scale=False
    )
    assert report["automatic_pass"] is True
    assert all(report["checks"].values())
