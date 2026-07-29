from __future__ import annotations

from scripts.run_u6_p4q_constant_rate_integration_live import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_constant_rate_integration_live import (
    load_contract,
    small_integration_evidence,
)


CONFIG = ROOT / "configs/u6_p4q_constant_rate_integration_live_v1.json"


def test_contract_is_explicit_and_default_legacy() -> None:
    contract, _ = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["executor"]["mode"] == "scalar-cdf-v1"
    assert contract["executor"]["legacy_default_mode"] == "legacy-v1"
    assert contract["executor"]["nonconstant_rate_behavior"] == "fail-closed"


def test_small_integration_preserves_legacy_and_rejects_spatial() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    evidence = small_integration_evidence(contract, parent)
    assert all(
        row["legacy_expected"]
        for row in evidence["partitions"].values()
    )
    assert all(
        row["scalar_legacy_exact"]
        for row in evidence["partitions"].values()
    )
    assert evidence["nonconstant_scalar_request_rejected"] is True
