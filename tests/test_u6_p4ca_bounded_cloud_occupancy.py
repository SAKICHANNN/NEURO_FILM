from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.bounded_cloud_occupancy import (
    BoundedCloudOccupancyEvaluationError,
    _validate_contract,
    evaluate_bounded_cloud_occupancy,
)
from src.film_physics.bounded_cloud_occupancy import (
    BoundedCloudOccupancyError,
    build_bounded_cloud_dc_receipt,
    counter_binomial_region,
    render_dc_projected_bounded_cloud_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ca_bounded_cloud_occupancy_v1.json"


def _small_receipt():
    return build_bounded_cloud_dc_receipt(
        (43, 47),
        profile_id="test-bounded-cloud",
        particle_sigma_pixels=1.3253699417769098,
        cluster_sigma_pixels=1.1268619873805532,
        mean_offspring=27.765942352935905,
        component_seeds=(2611923443488327891, 11400714819323198485),
        realization_seed=2608025001,
        trials=16,
        probability=0.5,
        truncate=4.0,
        canonical_row_block_height=11,
    )


def test_binomial_counts_are_bounded_and_coordinate_exact() -> None:
    full = counter_binomial_region(
        (31, 37), origin_yx=(0, 0), shape=(31, 37), trials=16, probability=0.5, seed=91
    )
    assembled = np.empty_like(full)
    for y0 in range(0, 31, 7):
        height = min(7, 31 - y0)
        assembled[y0 : y0 + height] = counter_binomial_region(
            (31, 37),
            origin_yx=(y0, 0),
            shape=(height, 37),
            trials=16,
            probability=0.5,
            seed=91,
        )
    assert np.array_equal(full, assembled)
    assert int(np.min(full)) >= 0
    assert int(np.max(full)) <= 16


def test_dc_projection_is_exact_and_tamper_rejected() -> None:
    receipt = _small_receipt()
    full = render_dc_projected_bounded_cloud_region(
        receipt, origin_yx=(0, 0), shape=receipt.full_shape
    )
    assembled = np.empty_like(full)
    for y0 in range(0, receipt.full_shape[0], 7):
        height = min(7, receipt.full_shape[0] - y0)
        assembled[y0 : y0 + height] = render_dc_projected_bounded_cloud_region(
            receipt, origin_yx=(y0, 0), shape=(height, receipt.full_shape[1])
        )
    assert np.array_equal(full, assembled)
    assert abs(float(np.mean(full, dtype=np.float64))) <= 1e-15
    with pytest.raises(BoundedCloudOccupancyError):
        render_dc_projected_bounded_cloud_region(
            replace(receipt, raw_mean=receipt.raw_mean + 1.0),
            origin_yx=(0, 0),
            shape=(1, 1),
        )


def test_contract_rejects_tail_gate_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["evaluation"]["maximum_extreme_count_ratio_vs_gaussian"] = 1.0
    with pytest.raises(BoundedCloudOccupancyEvaluationError):
        _validate_contract(contract)


def test_formal_evaluation_is_exact_and_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_bounded_cloud_occupancy(contract, ROOT)
    second = evaluate_bounded_cloud_occupancy(contract, ROOT)
    assert first == second
    assert set(first["checks"]) == {
        "extreme_tail_reduction",
        "maximum_absolute_field",
        "quantile_9999",
        "radial_signature",
        "covariance",
        "variance",
        "projected_mean",
        "row_partition_exact",
        "repeat_receipt_exact",
        "tampered_receipt_rejected",
        "seed_distinct_fields",
        "finite",
    }
