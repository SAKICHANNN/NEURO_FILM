from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_dye_basis_measured_conformance import (
    CONTRACT_SHA256,
    FujifilmDyeConformanceError,
    _fit_basis_rows,
    evaluate_conformance,
    hash_file,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb4_fujifilm_dye_basis_measured_conformance_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB4"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmDyeConformanceError, match="contract hash drift"):
        load_contract(changed)


def test_nonnegative_basis_fit_is_exact_for_representable_rows() -> None:
    basis = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    coefficient = np.array([[0.2, 0.4], [0.5, 0.1]])
    target = coefficient @ basis.T
    actual_coefficient, rmse = _fit_basis_rows(target, basis)
    assert np.allclose(actual_coefficient, coefficient, atol=1e-12)
    assert np.max(rmse) <= 1e-12


def test_negative_density_fails_closed() -> None:
    with pytest.raises(FujifilmDyeConformanceError, match="NNLS input"):
        _fit_basis_rows(np.array([[0.1, -0.1]]), np.ones((2, 1)))


def test_formal_measured_conformance_is_exact() -> None:
    contract = load_contract(CONFIG)
    first = evaluate_conformance(contract, ROOT)
    second = evaluate_conformance(contract, ROOT)
    assert first == second
    assert first["passed"] is True
    assert first["failed_gates"] == []
    assert first["cells"]["joint-held"]["rows"] == 1152
