from __future__ import annotations

import numpy as np
import pytest

from scripts.run_u6_p4p_constant_rate_poisson_executor import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_constant_rate_poisson_executor import (
    counter_poisson_constant_rate_field,
    load_contract,
)
from src.film_physics.density_conditioned_structure import (
    counter_poisson_rate_field,
)


CONFIG = ROOT / "configs/u6_p4p_constant_rate_poisson_executor_v1.json"


def test_contract_keeps_default_and_spatial_paths_closed() -> None:
    contract = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["candidate"]["default_integration_allowed"] is False
    assert (
        contract["candidate"]["general_spatial_rate_path_change_allowed"]
        is False
    )


@pytest.mark.parametrize(
    ("value", "maximum"),
    [(0.0, 0.0), (0.1, 1.0), (8.0, 64.0), (127.5, 1024.0)],
)
def test_candidate_is_exact_to_legacy(value: float, maximum: float) -> None:
    shape = (31, 37)
    full = (53, 61)
    origin = (11, 7)
    legacy = counter_poisson_rate_field(
        np.full(shape, value, dtype=np.float64),
        full,
        origin_yx=origin,
        seed=260801,
        maximum_rate=maximum,
    )
    candidate = counter_poisson_constant_rate_field(
        value,
        full,
        origin_yx=origin,
        shape=shape,
        seed=260801,
        maximum_rate=maximum,
    )
    assert np.array_equal(candidate, legacy)


def test_candidate_rejects_invalid_maximum() -> None:
    with pytest.raises(ValueError):
        counter_poisson_constant_rate_field(
            2.0,
            (5, 7),
            origin_yx=(0, 0),
            shape=(5, 7),
            seed=0,
            maximum_rate=1.0,
        )
