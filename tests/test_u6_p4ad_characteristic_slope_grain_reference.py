from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_characteristic_slope_grain import (
    CharacteristicSlopeGrainError,
    evaluate_reference,
    load_contract,
)
from src.film_physics.characteristic_slope_grain import (
    HillCharacteristicProfile,
    developed_density_variance,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p4ad_characteristic_slope_grain_reference_v1.json"
)


def test_hill_profile_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError, match="invalid Hill"):
        HillCharacteristicProfile(0.1, 2.0, 0.18, 0.5)
    profile = HillCharacteristicProfile(0.1, 2.0, 0.18, 1.6)
    with pytest.raises(ValueError, match="exposure"):
        profile.density(np.asarray([0.0]))
    with pytest.raises(ValueError, match="photon scale"):
        developed_density_variance(
            np.asarray([0.1]), profile, photon_scale=0.0
        )


def test_contract_rejects_curve_parameter_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["gamma"] = 1.7
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicSlopeGrainError, match="contract drift"):
        load_contract(path)


def test_analytic_reference_is_repeat_exact_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_reference(contract, ROOT)
    second = evaluate_reference(contract, ROOT)
    assert first == second
    assert first["passed"]
    assert all(first["gate_results"].values())
    assert first["decision"] == (
        "open_existing_sensitometry_derivative_propagation_audit"
    )


def test_photon_scale_changes_amplitude_not_normalized_shape() -> None:
    profile = HillCharacteristicProfile(0.1, 2.0, 0.18, 1.6)
    exposure = np.geomspace(1e-4, 16.0, 2049)
    low = developed_density_variance(exposure, profile, photon_scale=128.0)
    high = developed_density_variance(exposure, profile, photon_scale=8192.0)
    assert np.allclose(low / np.max(low), high / np.max(high), atol=1e-15)
    assert np.allclose(low / high, 64.0, rtol=1e-14, atol=0.0)
