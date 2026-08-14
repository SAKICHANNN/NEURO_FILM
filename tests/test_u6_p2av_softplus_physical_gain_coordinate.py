from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_gain_density_interpolation import load_contract, run_audit
from src.film_physics.positive_density_field import (
    PhysicalGainSoftplusDensityParameterProfile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2av_softplus_physical_gain_coordinate_v1.json"


def test_p2av_fresh_density_confirmation_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["parameter_refit_count_zero"] is True
    assert first["measurements"]["minimum_developed_density"] > 0.0


def test_p2av_physical_gain_profile_exact_nodes_and_no_extrapolation() -> None:
    nodes = load_contract(CONTRACT)["parameter_nodes"]
    profile = PhysicalGainSoftplusDensityParameterProfile(
        source_evidence_id="a" * 64,
        densities=tuple(nodes["densities"]),
        fitted_a=tuple(nodes["a"]),
        fitted_b=tuple(nodes["b"]),
        target_sigma_d=tuple(nodes["target_aperture_sigma_d"]),
    )
    actual_a, actual_b = profile.parameters(
        np.asarray(profile.densities), np.asarray(profile.target_sigma_d)
    )
    assert np.array_equal(actual_a, np.asarray(profile.fitted_a))
    assert np.array_equal(actual_b, np.asarray(profile.fitted_b))
    with pytest.raises(ValueError):
        profile.parameters(0.0, 0.01)
    with pytest.raises(ValueError):
        profile.parameters(2.0, 0.01)
