from __future__ import annotations

from pathlib import Path

from src.eval.physical_moment_lod import (
    evaluate_moment_corrected_lod,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p4c2_moment_corrected_lod_v1.json"


def test_contract_loads_with_disjoint_seeds() -> None:
    contract = load_contract(CONTRACT)
    split = contract["split"]
    assert contract["node"] == "U6.P4C2"
    assert len(
        {
            split["development_reference_seed"],
            split["confirmatory_reference_seed"],
            split["development_direct_seed"],
            split["confirmatory_direct_seed"],
        }
    ) == 4


def test_frozen_moment_corrected_lod_report_passes() -> None:
    report = evaluate_moment_corrected_lod(load_contract(CONTRACT))
    assert report["stable_evidence_id"] == (
        "d823530e3ad08c4120d5bf551ddfceb5b452413c7fa9c89d5d146942cc2b1da1"
    )
    assert report["automatic_pass"] is True
    assert all(report["decisions"].values())
