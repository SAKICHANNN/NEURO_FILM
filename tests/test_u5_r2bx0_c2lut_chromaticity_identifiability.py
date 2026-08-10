from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.c2lut_chromaticity_identifiability import (
    ChromaticityLUTError,
    evaluate_contract,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bx0_c2lut_chromaticity_identifiability_v1.json"


def test_bx0_capacity_passes_but_metamer_fixture_is_underpowered() -> None:
    report = evaluate_contract(CONTRACT)
    assert report["capacity_pass"]
    assert not report["descriptor_sufficiency_pass"]
    assert report["decision"] == "retain_capacity_close_underpowered_metamer_fixture"
    assert all(report["capacity_gates"].values())
    assert report["identifiability_gates"]["same_descriptor_same_prediction"]
    assert not report["identifiability_gates"]["different_spectra_different_truth"]
    assert not report["identifiability_gates"]["descriptor_only_error_small"]
    assert report["identifiability_gates"]["hidden_oracle_exact"]


def test_bx0_contract_drift_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixture"]["low_rank_basis_rank"] = 5
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChromaticityLUTError, match="boundary drift"):
        load_contract(path)


def test_bx0_runner_is_byte_deterministic(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_u5_r2bx0_c2lut_chromaticity_identifiability.py"
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
