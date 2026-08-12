from pathlib import Path

import pytest

from src.eval.target_resolution_cloud_two_moment_compiler_v4 import (
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4du_target_resolution_cloud_two_moment_compiler_v4.json"


def test_contract_parent() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4du-")


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text().replace("close_moment", "wrong_moment"))
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
