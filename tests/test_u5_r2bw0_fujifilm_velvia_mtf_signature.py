from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.fujifilm_velvia_mtf_signature import (
    REPORT_SCHEMA,
    VelviaMtfSignatureError,
    audit_signature,
    canonical_json,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bw0_fujifilm_velvia_mtf_revision_signature_v1.json"


def test_bw0_real_audit_is_repeat_exact(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    first = audit_signature(config, ROOT, overlay_dir=tmp_path / "first")
    second = audit_signature(config, ROOT, overlay_dir=tmp_path / "second")
    assert first["schema"] == REPORT_SCHEMA
    assert canonical_json(first) == canonical_json(second)
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert set(first["revisions"]) == set(config["comparison"]["velvia_revisions"])
    assert set(first["current_velvia_to_kodak_log_shape_rmse"]) == set(
        config["comparison"]["kodak_stocks"]
    )
    assert all(
        first["gate_results"][key]
        for key in (
            "source_integrity",
            "embedded_graph_pixel_identity",
            "source_ink_trace",
            "trace_count_and_response_range",
            "visual_overlays_emitted",
        )
    )


def test_bw0_contract_drift_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["minimum_each_kodak_log_shape_rmse"] = 0.079
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VelviaMtfSignatureError, match="contract drift"):
        load_contract(path)


def test_bw0_parent_hash_drift_rejected(tmp_path: Path) -> None:
    config = copy.deepcopy(load_contract(CONFIG))
    config["trace"]["sha256"] = "0" * 64
    with pytest.raises(VelviaMtfSignatureError, match="parent integrity mismatch"):
        audit_signature(config, ROOT, overlay_dir=tmp_path)


def test_bw0_repository_script_is_directly_executable(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_u5_r2bw0_fujifilm_velvia_mtf_signature.py"),
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--overlay-dir",
            str(tmp_path / "overlays"),
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "decision=close_exact_velvia_revision_source_signature" in completed.stdout
