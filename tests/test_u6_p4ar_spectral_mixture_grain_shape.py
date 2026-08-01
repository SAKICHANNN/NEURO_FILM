from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.spectral_mixture_grain_shape import load_contract
from src.film_physics.spectral_structure import (
    log_gaussian_spectral_psd,
    quantize_simplex_weights,
    spectral_normal_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ar_spectral_mixture_grain_shape_v1.json"


def test_simplex_quantization_is_nonnegative_exact_and_deterministic() -> None:
    raw = np.asarray([0.11, 0.22, 0.33, 0.34], dtype=np.float64)
    first = quantize_simplex_weights(raw, denominator=1024)
    second = quantize_simplex_weights(raw, denominator=1024)
    assert np.array_equal(first, second)
    assert np.all(first >= 0.0)
    assert np.sum(np.rint(first * 1024)).item() == 1024
    assert np.sum(first).item() == 1.0


def test_spectral_psd_is_nonnegative_and_has_zero_dc() -> None:
    psd = log_gaussian_spectral_psd(
        (137, 143),
        centers_cycles_per_pixel=np.asarray([0.03, 0.09, 0.27]),
        bandwidth_octaves=0.55,
        weights=np.asarray([0.2, 0.5, 0.3]),
        white_floor_fraction=0.005,
    )
    assert psd.shape == (137, 72)
    assert psd[0, 0] == 0.0
    assert np.all(psd >= 0.0)
    assert np.isfinite(psd).all()


def test_spectral_field_repeats_and_arbitrary_rows_partition_exactly() -> None:
    kwargs = {
        "full_shape": (137, 143),
        "centers_cycles_per_pixel": np.asarray([0.03, 0.09, 0.27]),
        "bandwidth_octaves": 0.55,
        "weights": np.asarray([0.2, 0.5, 0.3]),
        "white_floor_fraction": 0.005,
        "seed": 261229,
    }
    full = spectral_normal_region(**kwargs, origin_yx=(0, 0), shape=(137, 143))
    repeat = spectral_normal_region(**kwargs, origin_yx=(0, 0), shape=(137, 143))
    stitched = np.concatenate(
        [
            spectral_normal_region(
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
    assert np.isfinite(full).all()


def test_invalid_spectral_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid spectral simplex weights"):
        quantize_simplex_weights(np.asarray([0.5, -0.1]), denominator=1024)
    with pytest.raises(ValueError, match="spectral mixture parameters are invalid"):
        log_gaussian_spectral_psd(
            (64, 64),
            centers_cycles_per_pixel=np.asarray([0.1, 0.05]),
            bandwidth_octaves=0.55,
            weights=np.asarray([0.5, 0.5]),
            white_floor_fraction=0.005,
        )


def test_contract_keeps_shared_fit_and_confirmation_sealed() -> None:
    contract = load_contract(CONTRACT)
    assert contract["split"]["stock_labels_available_to_fit"] is False
    assert contract["split"]["confirmation_pixels_read_after_bundle_freeze"]
    assert contract["model"]["per_channel_weights_allowed"] is False
    assert contract["model"]["stock_specific_fit_allowed"] is False
    assert "not physical particle geometry" in contract["claim_ceiling"]
