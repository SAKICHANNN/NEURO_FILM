from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.spectral_metamer_camera_operator import (
    SpectralMetamerError,
    evaluate_contract,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bx1_spectral_metamer_camera_operator_v1.json"


def test_bx1_same_white_spectra_have_no_material_shared_operator_penalty() -> None:
    report = evaluate_contract(CONTRACT, ROOT)
    assert not report["passed"]
    assert report["decision"] == "close_exact_spectral_metamer_fixture"
    assert report["gate_results"]["observer_white_xyz"]
    assert report["gate_results"]["observer_chromaticity"]
    assert report["gate_results"]["operator_separation"]
    assert not report["gate_results"]["shared_penalty_each"]
    assert not report["gate_results"]["shared_penalty_median"]
    assert report["confirmation_target_reads_before_operator_freeze"] == 0


def test_bx1_parent_and_contract_drift_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixture"]["null_amplitude_fraction_of_positive_limit"] = 0.79
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SpectralMetamerError, match="boundary drift"):
        load_contract(path, ROOT)


def test_bx1_runner_is_byte_deterministic(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_u5_r2bx1_spectral_metamer_camera_operator.py"
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
