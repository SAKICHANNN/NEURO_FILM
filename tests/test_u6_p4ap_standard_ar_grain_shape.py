from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.standard_ar_grain_shape import load_contract
from src.film_physics.autoregressive_structure import (
    ar_impulse_metrics,
    causal_ar_normal_region,
    causal_ar_positions,
    quantize_ar_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ap_standard_ar_grain_shape_v1.json"


def test_causal_ar_positions_are_nested_and_have_standard_counts() -> None:
    rows = {lag: causal_ar_positions(lag) for lag in (1, 2, 3)}
    assert [len(rows[lag]) for lag in (1, 2, 3)] == [4, 12, 24]
    assert set(rows[1]) < set(rows[2]) < set(rows[3])
    assert all(dy < 0 or (dy == 0 and dx < 0) for dy, dx in rows[3])


def test_quantized_ar_field_repeats_and_partitions_exactly() -> None:
    raw = np.asarray([0.12, 0.16, 0.08, 0.10], dtype=np.float64)
    coefficients = quantize_ar_coefficients(
        raw,
        step=1.0 / 512.0,
        minimum=-0.25,
        maximum=127.0 / 512.0,
    )
    metrics = ar_impulse_metrics(
        coefficients,
        lag=1,
        shape=(257, 257),
        tail_width=16,
    )
    kwargs = {
        "full_shape": (137, 143),
        "lag": 1,
        "coefficients": coefficients,
        "seed": 260921,
        "normalization_energy": metrics["energy"],
    }
    full = causal_ar_normal_region(
        **kwargs,
        origin_yx=(0, 0),
        shape=kwargs["full_shape"],
    )
    repeat = causal_ar_normal_region(
        **kwargs,
        origin_yx=(0, 0),
        shape=kwargs["full_shape"],
    )
    stitched = np.concatenate(
        [
            causal_ar_normal_region(
                **kwargs,
                origin_yx=(y0, 0),
                shape=(min(31, 137 - y0), 143),
            )
            for y0 in range(0, 137, 31)
        ],
        axis=0,
    )
    assert np.array_equal(full, repeat)
    assert np.array_equal(full, stitched)
    assert metrics["tail_energy_fraction"] < 1e-8
    assert np.isfinite(full).all()


def test_coefficient_quantization_never_clips_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside the frozen envelope"):
        quantize_ar_coefficients(
            np.asarray([0.30]),
            step=1.0 / 512.0,
            minimum=-0.25,
            maximum=127.0 / 512.0,
        )


def test_contract_keeps_confirmation_sealed_and_claim_bounded() -> None:
    contract = load_contract(CONTRACT)
    assert contract["split"]["confirmation_pixels_read_after_candidate_freeze"]
    assert contract["split"]["stock_labels_available_to_fit"] is False
    assert contract["model"]["display_rgb_noise_allowed"] is False
    assert "not AFGS1 wire conformance" in contract["claim_ceiling"]
