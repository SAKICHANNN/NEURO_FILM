from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_step_wedge_oecf import (
    StepWedgeOecfError,
    _axis_values,
    _rank_correlation,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6aj_uchicago_step_wedge_oecf_v1.json"


def test_contract_is_frozen() -> None:
    payload = load_contract(CONTRACT)
    assert payload["measurement"]["confirmation_step_indices_zero_based"] == [
        2,
        5,
        8,
        11,
        14,
        17,
        20,
    ]
    assert payload["gates"]["two_wedges_same_best_parametric_family"] is True


def test_contract_rejects_overlap(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["measurement"]["confirmation_step_indices_zero_based"].append(0)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StepWedgeOecfError):
        load_contract(path)


def test_axis_values_and_rank_are_deterministic() -> None:
    image = np.repeat(np.arange(210, dtype=np.uint16)[:, None], 8, axis=1)
    values = _axis_values(image, 0, 21, 0.1)
    assert values.shape == (21,)
    assert np.all(np.diff(values) > 0.0)
    assert _rank_correlation(np.arange(21), values) == pytest.approx(1.0)
    assert _rank_correlation(np.arange(21), values[::-1]) == pytest.approx(-1.0)
