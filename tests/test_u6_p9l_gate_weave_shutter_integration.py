from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.gate_weave_shutter_integration import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9l_gate_weave_shutter_integration_v1.json"


@pytest.fixture(scope="module")
def report() -> dict:
    return run_audit(root=ROOT, contract=load_contract(CONTRACT))


def test_formal_gate_weave_shutter_audit_is_deterministic(report: dict) -> None:
    assert report["measurements"]["candidate_replay_byte_exact"] is True
    assert report["measurements"]["candidate_reverse_frame_order_byte_exact"] is True
    assert report["measurements"]["candidate_partition_byte_exact"] is True


def test_candidate_tracks_dense_shutter_reference(report: dict) -> None:
    measurements = report["measurements"]
    assert measurements["candidate_dense_reference_rmse"] < 0.001
    assert measurements["candidate_rmse_improvement_over_instantaneous"] > 0.0
