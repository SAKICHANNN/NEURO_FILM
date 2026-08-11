from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.gate_weave_sampling import load_contract, run_audit
from src.film_physics.gate_weave_sampling import (
    GateWeaveSamplingError,
    integrate_padded_translation,
    sample_padded_translation,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9b_gate_weave_sampling_v1.json"


def test_bilinear_sampler_matches_integer_crop_and_constant() -> None:
    source = np.arange(8 * 10, dtype=np.float64).reshape(8, 10)
    sampled = sample_padded_translation(
        source,
        output_shape=(4, 6),
        padding_yx=(2, 2),
        offset_yx=(1.0, -1.0),
        interpolation="bilinear",
    )
    assert np.array_equal(sampled, source[3:7, 1:7])
    constant = np.full((8, 10, 3), 0.25)
    shifted = sample_padded_translation(
        constant,
        output_shape=(4, 6),
        padding_yx=(2, 2),
        offset_yx=(0.37, -0.71),
        interpolation="bilinear",
    )
    assert np.array_equal(shifted, np.full((4, 6, 3), 0.25))


def test_sampler_rejects_out_of_padding_and_unknown_interpolation() -> None:
    source = np.zeros((8, 10), dtype=np.float64)
    with pytest.raises(GateWeaveSamplingError):
        sample_padded_translation(
            source,
            output_shape=(4, 6),
            padding_yx=(2, 2),
            offset_yx=(2.1, 0.0),
            interpolation="bilinear",
        )
    with pytest.raises(GateWeaveSamplingError):
        sample_padded_translation(
            source,
            output_shape=(4, 6),
            padding_yx=(2, 2),
            offset_yx=(0.0, 0.0),
            interpolation="cubic",  # type: ignore[arg-type]
        )


def test_shutter_integration_preserves_static_and_linear_fields() -> None:
    y = np.arange(10, dtype=np.float64)[:, None]
    x = np.arange(12, dtype=np.float64)[None, :]
    source = 0.1 + 0.01 * y + 0.02 * x
    static = integrate_padded_translation(
        source,
        output_shape=(6, 8),
        padding_yx=(2, 2),
        start_offset_yx=(0.25, -0.5),
        end_offset_yx=(0.25, -0.5),
        sample_count=8,
    )
    point = sample_padded_translation(
        source,
        output_shape=(6, 8),
        padding_yx=(2, 2),
        offset_yx=(0.25, -0.5),
        interpolation="bilinear",
    )
    assert np.array_equal(static, point)

    integrated = integrate_padded_translation(
        source,
        output_shape=(6, 8),
        padding_yx=(2, 2),
        start_offset_yx=(-1.0, -1.5),
        end_offset_yx=(1.0, 1.5),
        sample_count=8,
    )
    center = sample_padded_translation(
        source,
        output_shape=(6, 8),
        padding_yx=(2, 2),
        offset_yx=(0.0, 0.0),
        interpolation="bilinear",
    )
    assert np.allclose(integrated, center, rtol=0.0, atol=2e-16)


@pytest.mark.parametrize("sample_count", [0, -1, 4097, 1.5])
def test_shutter_integration_rejects_invalid_sample_count(sample_count: object) -> None:
    with pytest.raises(GateWeaveSamplingError):
        integrate_padded_translation(
            np.zeros((8, 10), dtype=np.float64),
            output_shape=(4, 6),
            padding_yx=(2, 2),
            start_offset_yx=(0.0, 0.0),
            end_offset_yx=(1.0, 1.0),
            sample_count=sample_count,  # type: ignore[arg-type]
        )


def test_formal_gate_weave_sampling_audit_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["bilinear_rmse_improvement_over_integer"] > 0.4
