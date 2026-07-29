from __future__ import annotations

from scripts.run_u6_p4e_density_conditioned_lod_audit import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_density_conditioned_lod import (
    evaluate_density_conditioned_lod,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4e_density_conditioned_lod_audit_v1.json"


def test_contract_and_parent_are_exact() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["lod_factors"] == [2, 4]
    assert parent["model"]["input_domain"] == "developed_optical_density"
    assert contract["photograph_access_allowed"] is False


def test_lod_candidate_closes_on_frozen_reference_mean_gate() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = evaluate_density_conditioned_lod(contract, parent)
    assert report["automatic_pass"] is False
    assert report["checks"]["reference_local_mean"] is False
    assert all(
        value
        for name, value in report["checks"].items()
        if name != "reference_local_mean"
    )
    for split in ("development", "confirmation"):
        for row in report[split]["factors"]:
            assert row["candidate_target_column_correlation"] >= 0.98
            assert abs(row["constant_rate_target_column_correlation"]) <= 0.5
            assert row["display_noise_out_of_domain_fraction"] >= 0.001
            assert row["repeat_exact"] is True
            assert row["row_partition_exact"] is True
