from __future__ import annotations

import json

import numpy as np
import pytest

from src.filmfx.boolean_grain import (
    BooleanGrainContext,
    build_boolean_grain_context,
    render_boolean_grain,
    render_boolean_grain_region,
)


PARAMETERS = {
    "radius_input_pixels": 0.4,
    "monte_carlo_samples": 8,
    "gaussian_filter_sigma_output_pixels": 0.6,
    "maximum_input_intensity": 255.0 / 256.0,
    "epsilon": 1e-12,
    "seed": 41,
}


def test_zero_field_has_no_grains_and_renders_exact_zero() -> None:
    context = build_boolean_grain_context(np.zeros((5, 7)), **PARAMETERS)
    assert context.grain_count == 0
    output = render_boolean_grain(context, output_zoom=2)
    assert output.shape == (10, 14)
    assert output.dtype == np.float32
    assert not output.flags.writeable
    assert np.array_equal(output, np.zeros_like(output))


def test_context_repeat_serialization_and_read_only_arrays() -> None:
    intensity = np.full((7, 6), 0.5, dtype=np.float64)
    first = build_boolean_grain_context(intensity, **PARAMETERS)
    second = build_boolean_grain_context(intensity, **PARAMETERS)
    assert first.grain_count > 0
    assert first.fingerprint() == second.fingerprint()
    assert np.array_equal(first.grain_centers_yx, second.grain_centers_yx)
    assert not first.input_intensity.flags.writeable
    assert not first.grain_centers_yx.flags.writeable
    replay = BooleanGrainContext.from_dict(
        json.loads(json.dumps(first.to_dict(), sort_keys=True))
    )
    assert replay.fingerprint() == first.fingerprint()
    assert np.array_equal(
        render_boolean_grain(replay, output_zoom=2),
        render_boolean_grain(first, output_zoom=2),
    )


def test_region_partition_is_bit_exact() -> None:
    context = build_boolean_grain_context(
        np.full((8, 8), 0.55, dtype=np.float64),
        **PARAMETERS,
    )
    full = render_boolean_grain(context, output_zoom=2)
    pieces = []
    starts = (0, 5, 11)
    ends = (5, 11, 16)
    for start, end in zip(starts, ends, strict=True):
        pieces.append(
            render_boolean_grain_region(
                context,
                output_zoom=2,
                output_origin_yx=(start, 0),
                output_shape=(end - start, 16),
            )
        )
    assert np.array_equal(np.concatenate(pieces), full)


def test_flat_field_mean_is_bounded_and_tracks_input() -> None:
    context = build_boolean_grain_context(
        np.full((12, 12), 0.5, dtype=np.float64),
        **PARAMETERS,
    )
    output = render_boolean_grain(context, output_zoom=2)
    interior = output[4:-4, 4:-4]
    assert np.min(interior) >= 0.0
    assert np.max(interior) <= 1.0
    assert abs(float(np.mean(interior)) - 0.5) < 0.15
    assert float(np.var(interior)) > 0.0


def test_invalid_inputs_and_regions_fail_closed() -> None:
    with pytest.raises(ValueError):
        build_boolean_grain_context(np.array([[1.0]]), **PARAMETERS)
    with pytest.raises(ValueError):
        build_boolean_grain_context(
            np.zeros((2, 2)),
            **{**PARAMETERS, "radius_input_pixels": 0.0},
        )
    context = build_boolean_grain_context(np.zeros((2, 2)), **PARAMETERS)
    with pytest.raises(ValueError):
        render_boolean_grain(context, output_zoom=0)
    with pytest.raises(ValueError):
        render_boolean_grain_region(
            context,
            output_zoom=2,
            output_origin_yx=(0, 0),
            output_shape=(5, 4),
        )
