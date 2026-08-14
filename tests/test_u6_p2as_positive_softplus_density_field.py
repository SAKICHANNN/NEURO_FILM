from pathlib import Path

import numpy as np
import pytest

from src.eval.positive_softplus_density_field import load_contract, run_audit
from src.film_physics.positive_density_field import softplus_density_field

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2as_positive_softplus_density_field_v1.json"


def test_p2as_positive_density_confirmation_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["minimum_developed_density"] > 0.0
    assert first["measurements"]["confirmation_parameter_refit_count_zero"] is True
    assert first["measurements"]["hard_clipping_count_zero"] is True


def test_softplus_density_field_is_positive_and_rejects_invalid_inputs() -> None:
    unit = np.asarray([[-100.0, 0.0, 100.0]], dtype=np.float64)
    output = softplus_density_field(unit, a=-2.0, b=0.2)
    assert np.all(np.isfinite(output))
    assert np.all(output > 0.0)
    with pytest.raises(ValueError):
        softplus_density_field(unit, a=-2.0, b=-0.1)
    with pytest.raises(ValueError):
        softplus_density_field(np.asarray([np.nan]), a=-2.0, b=0.1)
