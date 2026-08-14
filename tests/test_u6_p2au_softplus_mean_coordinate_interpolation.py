from pathlib import Path

import numpy as np
import pytest

from src.eval.mean_anchored_density_interpolation import load_contract, run_audit
from src.film_physics.positive_density_field import (
    MeanAnchoredSoftplusDensityParameterProfile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2au_softplus_mean_coordinate_interpolation_v1.json"


def test_p2au_fresh_density_confirmation_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is False
    assert (
        first["gate_results"]["maximum_confirmation_p95_sigma_relative_error"] is False
    )
    assert first["measurements"]["parameter_refit_count_zero"] is True
    assert first["measurements"]["minimum_developed_density"] > 0.0


def test_p2au_mean_anchored_profile_exact_nodes_and_no_extrapolation() -> None:
    nodes = load_contract(CONTRACT)["parameter_nodes"]
    profile = MeanAnchoredSoftplusDensityParameterProfile(
        source_evidence_id="a" * 64,
        densities=tuple(nodes["densities"]),
        fitted_a=tuple(nodes["a"]),
        b=tuple(nodes["b"]),
    )
    actual_a, actual_b = profile.parameters(np.asarray(profile.densities))
    assert np.array_equal(actual_a, np.asarray(profile.fitted_a))
    assert np.array_equal(actual_b, np.asarray(profile.b))
    with pytest.raises(ValueError):
        profile.parameters(0.0)
    with pytest.raises(ValueError):
        profile.parameters(2.0)
