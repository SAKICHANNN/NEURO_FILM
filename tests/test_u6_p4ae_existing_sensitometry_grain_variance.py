from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_existing_sensitometry_grain import (
    ExistingSensitometryGrainError,
    evaluate_existing,
    load_contract,
)
from src.film_physics.characteristic_slope_grain import (
    propagate_poisson_exposure_variance,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p4ae_existing_sensitometry_grain_variance_v1.json"
)


def test_general_variance_propagator_validates_derivative() -> None:
    exposure = np.asarray([0.1, 0.2], dtype=np.float64)
    derivative = np.asarray([1.0, 2.0], dtype=np.float64)
    variance = propagate_poisson_exposure_variance(
        exposure, derivative, photon_scale=10.0
    )
    assert np.array_equal(variance, np.asarray([0.01, 0.08]))
    with pytest.raises(ValueError, match="derivative"):
        propagate_poisson_exposure_variance(
            exposure, derivative[:1], photon_scale=10.0
        )


def test_contract_rejects_tail_gate_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["maximum_low_tail_to_peak_variance_ratio"] = 1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExistingSensitometryGrainError, match="contract drift"):
        load_contract(path)


def test_existing_sensitometry_audit_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_existing(contract, ROOT)
    second = evaluate_existing(contract, ROOT)
    assert first == second
    assert len(first["metrics"]["channel_reports"]) == 3
    assert first["passed"] is all(first["gate_results"].values())
    assert first["decision"] == (
        "open_synthetic_derivative_conditioned_structure_compiler"
        if first["passed"]
        else "close_existing_sensitometry_grain_variance_without_rescue"
    )
