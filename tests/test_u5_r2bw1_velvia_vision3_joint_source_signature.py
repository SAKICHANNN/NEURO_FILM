from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.velvia_vision3_joint_source_signature import (
    JointSourceSignatureError,
    audit_joint_source_signature,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bw1_velvia_vision3_joint_source_signature_v1.json"


def test_bw1_closes_before_joint_score_on_failed_parent_axis() -> None:
    report = audit_joint_source_signature(load_contract(CONTRACT), ROOT)
    assert report["decision"] == "close_before_joint_score_on_parent_source_calibration"
    assert report["preflight"]["bw0_decision_ok"]
    assert report["preflight"]["vision3_profile_decision_ok"]
    assert not report["preflight"]["upstream_axis_calibration_ok"]
    assert report["joint_feature_rows_computed"] == 0
    assert report["granularity_unit_conversions_applied"] == 0


def test_bw1_contract_drift_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_joint_distance_each_profile"] = 0.099
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JointSourceSignatureError, match="contract drift"):
        load_contract(path)


def test_bw1_direct_runner_is_deterministic(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_u5_r2bw1_velvia_vision3_joint_source_signature.py"
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
