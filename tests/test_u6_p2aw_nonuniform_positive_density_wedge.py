from pathlib import Path

import numpy as np
import pytest

from src.eval.nonuniform_positive_density_wedge import load_contract, run_audit
from src.film_physics.positive_density_field import softplus_density_field

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2aw_nonuniform_positive_density_wedge_v1.json"


def test_p2aw_nonuniform_wedge_replays_and_fails_closed() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["gate_results"]["maximum_plateau_p95_sigma_relative_error"] is False
    assert first["measurements"]["irregular_tile_exact"] is True
    assert first["measurements"]["minimum_developed_density"] > 0.0


def test_softplus_density_field_accepts_broadcast_parameters() -> None:
    unit = np.arange(12, dtype=np.float64).reshape(3, 4) / 12.0
    a = np.linspace(-2.0, 0.0, 4, dtype=np.float64)[None, :]
    b = np.linspace(0.01, 0.04, 3, dtype=np.float64)[:, None]
    actual = softplus_density_field(unit, a=a, b=b)
    expected = np.logaddexp(0.0, a + b * unit)
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError):
        softplus_density_field(unit, a=np.ones((2, 2)), b=0.1)
