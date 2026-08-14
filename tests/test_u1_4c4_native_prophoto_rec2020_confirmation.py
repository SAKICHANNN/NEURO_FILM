from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.rec2020_native_prophoto_confirmation import (
    CONTRACT_SHA256,
    NativeProPhotoError,
    load_contract,
    luminance_axis_interior_compress,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c4_native_prophoto_rec2020_confirmation_v1.json"


def test_contract_identity_and_scope_are_frozen() -> None:
    contract = load_contract(CONTRACT)
    assert CONTRACT_SHA256 == "6e3caf8e65a71c798e97dff26e3af21c56193409decdb1f5c48b7b794733d607"
    assert contract["experiment_id"] == "U1.4C4"
    assert contract["ingress"]["hard_clipping_allowed"] is False
    assert contract["ingress"]["source_or_operator_fitting_allowed"] is False
    assert contract["production_default_changed"] is False


def test_luminance_axis_compression_maps_only_extended_pixels() -> None:
    source = np.asarray(
        [[[0.2, 0.5, 0.8], [-0.1, 0.4, 1.1], [0.0, 1.0, 0.4]]],
        dtype=np.float32,
    )
    weights = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
    frozen = source.copy()
    mapped, scale, out_of_gamut = luminance_axis_interior_compress(
        source, luminance_weights=weights, margin=2.0 / 65535.0
    )
    np.testing.assert_array_equal(source, frozen)
    np.testing.assert_array_equal(mapped[0, 0], source[0, 0])
    np.testing.assert_array_equal(mapped[0, 2], source[0, 2])
    assert out_of_gamut.tolist() == [[False, True, False]]
    assert 0.0 <= float(scale[0, 1]) < 1.0
    assert float(np.min(mapped)) >= 0.0
    assert float(np.max(mapped)) <= 1.0
    np.testing.assert_allclose(
        np.matmul(mapped.astype(np.float64), weights),
        np.matmul(source.astype(np.float64), weights),
        atol=2e-6,
        rtol=0.0,
    )


def test_compression_preserves_residual_direction() -> None:
    source = np.asarray([[[-0.2, 0.5, 1.2]]], dtype=np.float32)
    weights = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
    mapped, scale, _ = luminance_axis_interior_compress(
        source, luminance_weights=weights, margin=2.0 / 65535.0
    )
    luminance = np.matmul(source.astype(np.float64), weights)[..., None]
    neutral = np.repeat(luminance, 3, axis=-1)
    expected = neutral + scale[..., None].astype(np.float64) * (
        source.astype(np.float64) - neutral
    )
    np.testing.assert_allclose(mapped, expected.astype(np.float32), atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("margin", [0.0, -0.1, 0.5, float("nan")])
def test_invalid_margin_fails_closed(margin: float) -> None:
    source = np.full((2, 2, 3), 0.5, dtype=np.float32)
    weights = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
    with pytest.raises(NativeProPhotoError):
        luminance_axis_interior_compress(
            source, luminance_weights=weights, margin=margin
        )


def test_invalid_source_or_weights_fail_closed() -> None:
    source = np.full((2, 2, 3), 0.5, dtype=np.float32)
    weights = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
    invalid = source.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(NativeProPhotoError):
        luminance_axis_interior_compress(
            invalid, luminance_weights=weights, margin=2.0 / 65535.0
        )
    with pytest.raises(NativeProPhotoError):
        luminance_axis_interior_compress(
            source,
            luminance_weights=np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
            margin=2.0 / 65535.0,
        )
