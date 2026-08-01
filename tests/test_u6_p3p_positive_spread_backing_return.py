from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.backing_return import backing_return_profile_from_contract
from src.film_physics.contracts import PhysicalDomain, PhysicalDomainArray, PhysicalScale, PhysicalUnit
from src.film_physics.positive_spread_return import apply_positive_spread_backing_return


ROOT = Path(__file__).resolve().parents[1]


def _profile():
    return backing_return_profile_from_contract(json.loads((ROOT / "configs/u6_p3d_backing_return_reference_v1.json").read_text(encoding="utf-8")))


def _array(values: np.ndarray) -> PhysicalDomainArray:
    profile = _profile()
    return PhysicalDomainArray(values, PhysicalDomain.LAYER_EXPOSURE, PhysicalUnit.RELATIVE_LAYER_EXPOSURE, ("red", "green", "blue"), PhysicalScale(profile.pixel_pitch_um))


def test_positive_spread_preserves_constants_and_adds_only_outside_impulse() -> None:
    profile = _profile()
    constant = np.full((65, 67, 3), 0.18, dtype=np.float64)
    constant_result = apply_positive_spread_backing_return(_array(constant), profile)
    assert np.count_nonzero(constant_result.residual) == 0
    impulse = np.full((129, 129, 3), 0.01, dtype=np.float64)
    impulse[64, 64] = 4.0
    result = apply_positive_spread_backing_return(_array(impulse), profile)
    assert np.count_nonzero(result.residual[64, 64]) == 0
    assert float(np.sum(result.residual)) > 0.0
    assert np.all(result.exposure.values >= impulse)


def test_positive_spread_is_repeat_exact() -> None:
    rng = np.random.default_rng(3031)
    values = rng.uniform(0.0, 4.0, size=(41, 43, 3)).astype(np.float64)
    first = apply_positive_spread_backing_return(_array(values), _profile())
    second = apply_positive_spread_backing_return(_array(values), _profile())
    assert np.array_equal(first.exposure.values, second.exposure.values)
    assert np.array_equal(first.residual, second.residual)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_positive_spread_rejects_non_float64(dtype: np.dtype) -> None:
    values = np.zeros((7, 9, 3), dtype=dtype)
    with pytest.raises(TypeError):
        apply_positive_spread_backing_return(_array(values), _profile())
