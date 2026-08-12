from pathlib import Path

import pytest

from src.eval.cloud_scanner_combined_ablation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dm_cloud_scanner_combined_ablation_v1.json"


def test_contract_parent() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4dm-cloud-scanner-combined-ablation-v1"
    )


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        CONTRACT.read_text().replace(
            "retain_material_separated_scanner_nuisance_chain", "wrong"
        )
    )
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
