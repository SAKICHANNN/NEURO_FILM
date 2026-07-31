from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_callier_operator import (
    CallierOperatorError,
    evaluate_operator,
    load_contract,
    profile_from_contract,
)
from src.film_physics.callier import (
    CallierProfile,
    apply_callier_components,
    callier_q_factor,
    invert_callier_components,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6n_generic_callier_operator_v1.json"


def _profile() -> CallierProfile:
    return profile_from_contract(load_contract(CONTRACT))


def test_profile_rejects_non_spectral_order() -> None:
    with pytest.raises(ValueError, match="red < green < blue"):
        CallierProfile("bad", 1.4, 0.35, (1.0, 0.75, 0.55), 4.05)


def test_dye_and_diffuse_controls_are_exact_identity() -> None:
    profile = _profile()
    rng = np.random.default_rng(1)
    dye = rng.uniform(0.0, 2.0, size=(11, 13, 3)).astype(np.float64)
    silver = rng.uniform(0.0, 1.0, size=dye.shape).astype(np.float64)
    assert np.array_equal(
        apply_callier_components(dye, np.zeros_like(silver), profile, collimation=1.0),
        dye,
    )
    assert np.array_equal(
        apply_callier_components(dye, silver, profile, collimation=0.0),
        dye + silver,
    )


def test_q_is_bounded_and_inverse_recovers_density() -> None:
    profile = _profile()
    density = np.linspace(0.0, 4.05, 1001)[:, None, None]
    density = np.broadcast_to(density, (1001, 1, 3)).copy()
    q = callier_q_factor(density, profile, collimation=1.0)
    assert np.min(q) >= 1.0
    assert np.max(q) <= 1.35
    directed = density * q
    recovered = invert_callier_components(
        directed, np.zeros_like(density), profile, collimation=1.0
    )
    assert np.max(np.abs(recovered - density)) <= 1e-12


def test_invalid_input_fails_without_clipping() -> None:
    profile = _profile()
    valid = np.zeros((2, 3, 3), dtype=np.float64)
    invalid = valid.copy()
    invalid[0, 0, 0] = -1e-6
    with pytest.raises(ValueError, match="finite"):
        apply_callier_components(valid, invalid, profile, collimation=1.0)
    with pytest.raises(ValueError, match="collimation"):
        callier_q_factor(valid, profile, collimation=float("nan"))


def test_contract_rejects_post_result_retuning(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["synthetic_evaluation"]["post_result_retuning_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CallierOperatorError, match="contract drift"):
        load_contract(path)


def test_synthetic_evaluation_is_repeat_exact_and_passes() -> None:
    config = load_contract(CONTRACT)
    first = evaluate_operator(config)
    second = evaluate_operator(config)
    assert first == second
    assert first["passed"]
    assert first["decision"] == "retain_generic_callier_primitive"
    assert all(first["gate_results"].values())
