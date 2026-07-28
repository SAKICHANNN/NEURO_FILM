from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.physical_structure_lod import (
    _area_mean,
    evaluate_physical_lod,
    load_contract,
)
from src.film_physics import (
    CompoundPoissonProfile,
    rescale_compound_poisson_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p4c_physical_lod_v1.json"


def test_contract_loads() -> None:
    assert load_contract(CONTRACT)["node"] == "U6.P4C1"


def test_area_mean_preserves_constant_and_mean() -> None:
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    output = _area_mean(values, 4)
    assert output.shape == (2, 2)
    assert float(np.mean(output)) == float(np.mean(values))
    assert np.array_equal(
        _area_mean(np.full((8, 8), 3.5, np.float32), 2),
        np.full((4, 4), 3.5, np.float32),
    )


def test_profile_rescale_preserves_physical_shot_expectation() -> None:
    base = CompoundPoissonProfile(
        "density-shot", 0.25, 0.6, 0.2, 2.0, 4
    )
    coarse = rescale_compound_poisson_profile(
        base, pixel_size_factor=4, seed=5
    )
    assert coarse.poisson_rate == 4.0
    assert coarse.correlation_sigma_pixels == 0.15
    assert coarse.baseline == base.baseline
    assert coarse.scale == 0.125
    assert coarse.poisson_rate * coarse.scale == (
        base.poisson_rate * base.scale
    )


def test_frozen_direct_lod_report_closes_naive_rescale() -> None:
    report = evaluate_physical_lod(load_contract(CONTRACT))
    assert report["stable_evidence_id"] == (
        "4425133616330e7cc186f6fe8d2cc8b4409a8f373c7bc6ba1bc01f2dae379c96"
    )
    assert report["automatic_pass"] is False
    assert report["decisions"] == {
        "acf": True,
        "brightening": True,
        "domain": True,
        "mean": False,
        "nps": True,
        "partition": True,
        "repeat": True,
        "variance": False,
    }
