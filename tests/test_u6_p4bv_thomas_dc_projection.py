from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.thomas_dc_projection import (
    ThomasDcProjectionEvaluationError,
    _validate_contract,
    evaluate_thomas_dc_projection,
)
from src.film_physics.thomas_dc_projection import (
    ThomasDcProjectionError,
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bv_thomas_dc_projection_v1.json"


def _small_receipt():
    return build_thomas_dc_receipt(
        (41, 47),
        profile_id="test-thomas",
        particle_sigma_pixels=1.3253699417769098,
        cluster_sigma_pixels=1.1268619873805532,
        mean_offspring=27.765942352935905,
        component_seeds=(2611923443488327891, 11400714819323198485),
        realization_seed=107,
        truncate=4.0,
        canonical_row_block_height=11,
    )


def test_receipt_projection_is_zero_mean_and_row_exact() -> None:
    receipt = _small_receipt()
    full = render_dc_projected_thomas_region(
        receipt, origin_yx=(0, 0), shape=receipt.full_shape
    )
    assembled = np.empty_like(full)
    for y0 in range(0, receipt.full_shape[0], 7):
        height = min(7, receipt.full_shape[0] - y0)
        assembled[y0 : y0 + height] = render_dc_projected_thomas_region(
            receipt, origin_yx=(y0, 0), shape=(height, receipt.full_shape[1])
        )
    assert np.array_equal(full, assembled)
    assert abs(float(np.mean(full, dtype=np.float64))) <= 1e-15


def test_tampered_receipt_fails_before_render() -> None:
    receipt = _small_receipt()
    with pytest.raises(ThomasDcProjectionError):
        render_dc_projected_thomas_region(
            replace(receipt, raw_mean=receipt.raw_mean + 1.0),
            origin_yx=(0, 0),
            shape=(1, 1),
        )


def test_contract_rejects_variance_rescaling() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["receipt"]["allow_variance_rescaling"] = True
    with pytest.raises(ThomasDcProjectionEvaluationError):
        _validate_contract(contract)


def test_formal_evaluation_is_exact_and_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_thomas_dc_projection(contract, ROOT)
    second = evaluate_thomas_dc_projection(contract, ROOT)
    assert first == second
    assert first["automatic_pass"]
    assert set(first["checks"]) == {
        "projected_mean",
        "nonzero_spectrum",
        "centered_covariance",
        "variance",
        "interior_variance",
        "row_partition_exact",
        "repeat_receipt_exact",
        "tampered_receipt_rejected",
        "seed_distinct_fields",
        "finite",
    }
