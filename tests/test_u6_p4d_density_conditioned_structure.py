from __future__ import annotations

import numpy as np
import pytest

from scripts.run_u6_p4d_density_conditioned_structure import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_density_conditioned_structure import (
    evaluate_density_conditioned_structure,
    load_contract,
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    counter_poisson_rate_field,
    render_density_conditioned_structure,
    render_density_conditioned_structure_region,
)


CONFIG = ROOT / "configs/u6_p4d_density_conditioned_structure_v1.json"


def test_contract_is_frozen_and_non_display_domain() -> None:
    contract = load_contract(CONFIG, CONFIG_SHA256)
    assert contract["model"]["input_domain"] == "developed_optical_density"
    assert contract["model"]["display_rgb_noise_allowed"] is False
    assert contract["photograph_access_allowed"] is False


def test_variable_rate_poisson_zero_and_domain() -> None:
    rate = np.array([[0.0, 0.1], [4.0, 64.0]], dtype=np.float64)
    counts = counter_poisson_rate_field(
        rate, rate.shape, origin_yx=(0, 0), seed=7
    )
    assert counts.dtype == np.uint16
    assert counts[0, 0] == 0
    with pytest.raises(ValueError, match="\\[0, 64\\]"):
        counter_poisson_rate_field(
            np.array([[65.0]]), (1, 1), origin_yx=(0, 0), seed=7
        )


def test_density_structure_repeat_partition_and_domain() -> None:
    contract = load_contract(CONFIG, CONFIG_SHA256)
    profiles = profiles_from_contract(contract)
    target = np.linspace(0.0, 2.0, 67, dtype=np.float64)
    target = np.broadcast_to(target[None, :, None], (71, 67, 3)).copy()
    full = render_density_conditioned_structure(target, profiles)
    repeat = render_density_conditioned_structure(target, profiles)
    assert np.array_equal(full.density, repeat.density)
    assert np.array_equal(full.transmittance, repeat.transmittance)
    assert float(np.min(full.density)) >= 0.0
    assert 0.0 < float(np.min(full.transmittance)) <= 1.0
    assembled = np.empty_like(full.density)
    for y0 in range(0, target.shape[0], 13):
        y1 = min(target.shape[0], y0 + 13)
        region = render_density_conditioned_structure_region(
            target,
            profiles,
            origin_yx=(y0, 0),
            shape=(y1 - y0, target.shape[1]),
        )
        assembled[y0:y1] = region.density
    assert np.array_equal(full.density, assembled)


def test_zero_density_is_exact_clear_base() -> None:
    contract = load_contract(CONFIG, CONFIG_SHA256)
    profiles = profiles_from_contract(contract)
    result = render_density_conditioned_structure(
        np.zeros((17, 19, 3), dtype=np.float64), profiles
    )
    assert np.count_nonzero(result.density) == 0
    assert np.all(result.transmittance == 1.0)


def test_frozen_evaluation_passes() -> None:
    contract = load_contract(CONFIG, CONFIG_SHA256)
    report = evaluate_density_conditioned_structure(contract)
    assert report["automatic_pass"] is True
    assert all(report["checks"].values())
