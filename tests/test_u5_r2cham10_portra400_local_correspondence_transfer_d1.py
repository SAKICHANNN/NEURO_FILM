from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.portra400_local_correspondence_transfer_d1 import (
    _patch_rows,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham10_portra400_local_correspondence_transfer_d1_v1.json"


def test_contract_freezes_primary_and_nuisance_roles() -> None:
    contract = load_contract(CONFIG)
    assert [row["primary"] for row in contract["natural_pairs"]] == [True, False]
    assert contract["gates"]["minimum_primary_improvement_over_identity_fraction"] == 0.05


def test_patch_rows_rejects_nonuniform_or_edge_support() -> None:
    source = np.full((31, 31, 3), 128, dtype=np.uint8)
    target = source.copy()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))["correspondence"]
    src, dst, folds = _patch_rows(source, target, np.array([[15.0, 15.0]]), np.array([[15.0, 15.0]]), spec)
    assert src.shape == dst.shape == (1, 3)
    assert folds.tolist() == [0]
    target[10:21, 10:21] = np.indices((11, 11)).sum(axis=0)[..., None] % 2 * 255
    with pytest.raises(ValueError, match="no eligible"):
        _patch_rows(source, target, np.array([[15.0, 15.0]]), np.array([[15.0, 15.0]]), spec)


def test_contract_rejects_role_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["natural_pairs"][1]["primary"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
