from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_step_wedge_oecf_correction import (
    CorrectedOecfError,
    _density_from_code,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ak_uchicago_step_wedge_oecf_correction_v1.json"


def test_contract_restricts_density_truth_to_positive_wedge() -> None:
    payload = load_contract(CONTRACT)
    assert payload["mode"].startswith("post-source-inspection")
    assert payload["negative_wedge_role"].startswith("unscored")
    assert (
        payload["supersedes"]["experiment_id"] == "u6.p6aj-uchicago-step-wedge-oecf-v1"
    )


def test_contract_rejects_missing_step(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["positive_wedge_row_boundaries"].pop()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorrectedOecfError):
        load_contract(path)


def test_log_oecf_roundtrip_fixture() -> None:
    density = np.asarray([0.1, 0.8, 1.6, 2.5], dtype=np.float64)
    code = 2300.0 + 65535.0 * np.power(10.0, -density / 1.42)
    recovered = _density_from_code(code, 2300.0, 1.42)
    assert np.max(np.abs(recovered - density)) < 1e-12
