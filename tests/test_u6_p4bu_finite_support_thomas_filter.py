from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.finite_support_thomas_filter import (
    FiniteSupportThomasFilterError,
    _validate_contract,
    evaluate_finite_support_thomas_filter,
)
from src.film_physics.finite_support_thomas import (
    gaussian_kernel_1d,
    render_finite_support_thomas_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bu_finite_support_thomas_filter_v1.json"


def test_gaussian_kernel_is_normalized_and_compact() -> None:
    kernel = gaussian_kernel_1d(1.3253699417769098, 4.0)
    assert kernel.size == 11
    assert np.sum(kernel) == pytest.approx(1.0)
    assert np.array_equal(kernel, kernel[::-1])


def test_small_region_matches_row_partitions() -> None:
    kwargs = {
        "full_shape": (41, 47),
        "particle_sigma_pixels": 1.3253699417769098,
        "cluster_sigma_pixels": 1.1268619873805532,
        "mean_offspring": 27.765942352935905,
        "component_seeds": (101, 103),
        "realization_seed": 107,
        "truncate": 4.0,
    }
    full = render_finite_support_thomas_region(
        origin_yx=(0, 0), shape=(41, 47), **kwargs
    )
    assembled = np.empty_like(full)
    for y0 in range(0, 41, 7):
        height = min(7, 41 - y0)
        assembled[y0 : y0 + height] = render_finite_support_thomas_region(
            origin_yx=(y0, 0), shape=(height, 47), **kwargs
        )
    assert np.array_equal(full, assembled)


def test_contract_rejects_halo_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["filter"]["maximum_halo_pixels"] = 8
    with pytest.raises(FiniteSupportThomasFilterError):
        _validate_contract(contract)


def test_formal_evaluation_repeats_and_is_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_finite_support_thomas_filter(contract, ROOT)
    second = evaluate_finite_support_thomas_filter(contract, ROOT)
    assert first == second
    assert set(first["checks"]) == {
        "radial_signature",
        "covariance",
        "interior_variance",
        "interior_mean",
        "row_partition_exact",
        "repeat_exact",
        "seed_distinct_fields",
        "finite",
    }
