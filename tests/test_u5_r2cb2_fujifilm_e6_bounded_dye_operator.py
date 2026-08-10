from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_e6_bounded_dye_operator import (
    CONTRACT_SHA256,
    FujifilmDyeOperatorError,
    _balanced_curves,
    evaluate_operator,
    hash_file,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb2_fujifilm_e6_bounded_dye_operator_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB2"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmDyeOperatorError, match="contract hash drift"):
        load_contract(changed)


def test_nonnegative_dye_balance_is_deterministic() -> None:
    curves = np.array(
        [[1.0, 0.5, 0.1], [0.1, 1.0, 0.5], [0.5, 0.1, 1.0]],
        dtype=np.float64,
    )
    first = _balanced_curves(curves)
    second = _balanced_curves(curves)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert np.all(first[1] >= 0.0)


def test_negative_digitized_density_fails_closed() -> None:
    curves = np.array(
        [[1.0, -0.1, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]],
        dtype=np.float64,
    )
    with pytest.raises(FujifilmDyeOperatorError, match="negative dye"):
        _balanced_curves(curves)


def test_formal_operator_compile_is_exact(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = evaluate_operator(contract, ROOT, tmp_path / "a")
    second = evaluate_operator(contract, ROOT, tmp_path / "b")
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert len(first["pairs"]) == 3
