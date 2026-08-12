from __future__ import annotations

from pathlib import Path

from src.eval.compiled_cloud_profile_restart import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dy_compiled_cloud_profile_restart_v1.json"


def test_p4dy_parent_is_exact() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4dy-")


def test_p4dy_restart_evaluation_obeys_frozen_gates() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())
