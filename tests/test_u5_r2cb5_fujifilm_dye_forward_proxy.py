from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.fujifilm_dye_forward_proxy import (
    CONTRACT_SHA256,
    FujifilmDyeForwardError,
    _fit_nonnegative_mapping,
    evaluate_forward_proxy,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb5_fujifilm_dye_forward_proxy_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB5"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmDyeForwardError, match="contract hash drift"):
        load_contract(changed)


def test_nonnegative_mapping_recovers_exact_positive_linear_data() -> None:
    source = np.column_stack((np.ones(16), np.linspace(0.0, 1.0, 16)))
    weights = np.array([[0.2, 0.4], [0.5, 0.1]])
    target = source @ weights
    actual = _fit_nonnegative_mapping(source, target)
    assert np.allclose(actual, weights, atol=1e-10)
    assert np.all(actual >= 0.0)


def test_negative_mapping_target_fails_closed() -> None:
    with pytest.raises(FujifilmDyeForwardError, match="fit input"):
        _fit_nonnegative_mapping(np.ones((16, 2)), -np.ones((16, 1)))


def test_formal_forward_proxy_is_repeat_exact_and_closed() -> None:
    contract = load_contract(CONFIG)
    first = evaluate_forward_proxy(contract, ROOT)
    second = evaluate_forward_proxy(contract, ROOT)
    assert first == second
    assert first["passed"] is False
    assert first["decision"] == "close_low_capacity_recorder_rgb_to_dye_forward_family"
    assert set(first["failed_gates"]) == {"absolute_error", "direct_spectral"}
