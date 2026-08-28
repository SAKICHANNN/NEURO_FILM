from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import src.filmfx.staged_colour as staged_colour_module
from src.filmfx import composite_layers, physical_halation_layer
from src.filmfx.staged_colour import (
    STAGED_PHYSICAL_COLOUR_VERSION,
    execute_staged_physical_colour_halation_default,
    materialized_physical_colour_halation_v2_reference,
)


def _colour_highlight_field(shape=(129, 181, 3), seed=641) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = rng.uniform(0.01, 0.30, size=shape).astype(np.float32)
    white = np.exp(
        -((y - shape[0] * 0.35) ** 2 + (x - shape[1] * 0.31) ** 2) / 150.0
    )
    warm = np.exp(
        -((y - shape[0] * 0.64) ** 2 + (x - shape[1] * 0.70) ** 2) / 210.0
    )
    base += white[..., None].astype(np.float32) * np.array([0.88, 0.85, 0.82], np.float32)
    base += warm[..., None].astype(np.float32) * np.array([0.92, 0.58, 0.25], np.float32)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _seam_max(difference: np.ndarray, tile_size: int) -> float:
    values = []
    for y in range(tile_size, difference.shape[0], tile_size):
        values.append(float(difference[max(0, y - 1) : min(difference.shape[0], y + 1)].max()))
    for x in range(tile_size, difference.shape[1], tile_size):
        values.append(float(difference[:, max(0, x - 1) : min(difference.shape[1], x + 1)].max()))
    return max(values, default=0.0)


@pytest.mark.parametrize("tile_size", [37, 59])
def test_staged_colour_tracks_materialized_v2_and_srgb8(tile_size: int) -> None:
    base = _colour_highlight_field()
    reference = materialized_physical_colour_halation_v2_reference(
        base, source_row_chunk=31
    )
    staged, metadata = execute_staged_physical_colour_halation_default(
        base,
        tile_size=tile_size,
        source_row_chunk=31,
        coarse_row_chunk=3,
    )
    rgb_difference = np.abs(staged.rgb - reference.rgb)
    alpha_difference = np.abs(staged.alpha - reference.alpha)
    staged_output = composite_layers(base, [staged])
    reference_output = composite_layers(base, [reference])
    output_difference = np.abs(staged_output - reference_output)
    assert float(rgb_difference.max()) <= 2e-6
    assert float(alpha_difference.max()) <= 2e-6
    assert float(output_difference.max()) <= 2e-6
    assert _seam_max(output_difference, tile_size) <= 2e-6
    assert np.rint(staged_output * 255).astype(np.uint8).tobytes() == np.rint(
        reference_output * 255
    ).astype(np.uint8).tobytes()
    assert metadata.version == STAGED_PHYSICAL_COLOUR_VERSION
    assert tuple(context.name for context in metadata.contexts) == (
        "local_mean",
        "red_tail",
        "red_glare",
    )
    assert len(metadata.percentiles) == 2
    assert metadata.global_context_bytes == sum(
        context.coarse_bytes for context in metadata.contexts
    )
    assert metadata.max_source_window_shape[0] < base.shape[0]
    assert all(
        context.row_stage.max_row_span < base.shape[0]
        for context in metadata.contexts
    )
    assert metadata.persistent_derived_full_scalar_bytes == 0
    assert metadata.scratch_disk_bytes == 0


def test_staged_colour_is_repeat_identical() -> None:
    base = _colour_highlight_field(seed=643)
    first, first_metadata = execute_staged_physical_colour_halation_default(
        base, tile_size=47, source_row_chunk=29, coarse_row_chunk=3
    )
    second, second_metadata = execute_staged_physical_colour_halation_default(
        base, tile_size=47, source_row_chunk=29, coarse_row_chunk=3
    )
    assert first.rgb.tobytes() == second.rgb.tobytes()
    assert first.alpha.tobytes() == second.alpha.tobytes()
    assert first_metadata == second_metadata


def test_materialized_v2_stays_within_frozen_legacy_compatibility() -> None:
    base = _colour_highlight_field(seed=644)
    legacy = physical_halation_layer(base)
    reference = materialized_physical_colour_halation_v2_reference(
        base, source_row_chunk=31
    )
    rgb_difference = np.abs(legacy.rgb - reference.rgb)
    alpha_difference = np.abs(legacy.alpha - reference.alpha)
    legacy_output = composite_layers(base, [legacy])
    reference_output = composite_layers(base, [reference])
    output_difference = np.abs(legacy_output - reference_output)
    code_difference = np.abs(
        np.rint(legacy_output * 255).astype(np.int16)
        - np.rint(reference_output * 255).astype(np.int16)
    )
    assert float(rgb_difference.max()) <= 7e-4
    assert float(alpha_difference.max()) <= 7e-4
    assert float(alpha_difference.mean()) <= 3e-5
    assert float(output_difference.max()) <= 7e-4
    assert float(output_difference.mean()) <= 3e-5
    assert np.count_nonzero(code_difference) / code_difference.size <= 0.005
    assert int(code_difference.max()) <= 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: value.astype(np.float64),
        lambda value: np.full_like(value, np.nan),
        lambda value: value * np.float32(2.0),
        lambda value: value[:, :, :2],
        lambda value: value[:1],
    ],
)
def test_staged_colour_rejects_invalid_input(factory) -> None:
    base = _colour_highlight_field()
    with pytest.raises(ValueError):
        execute_staged_physical_colour_halation_default(
            factory(base), tile_size=37, source_row_chunk=31, coarse_row_chunk=3
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": True},
        {"tile_size": 80},
        {"source_row_chunk": 129},
        {"coarse_row_chunk": 0},
    ],
)
def test_staged_colour_rejects_invalid_execution_geometry(kwargs) -> None:
    parameters = {"tile_size": 37, "source_row_chunk": 31, "coarse_row_chunk": 3}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        execute_staged_physical_colour_halation_default(
            _colour_highlight_field(), **parameters
        )


def test_colour_metadata_is_frozen_and_output_name_is_explicit() -> None:
    layer, metadata = execute_staged_physical_colour_halation_default(
        _colour_highlight_field(),
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        name="research_colour",
    )
    assert layer.name == "research_colour"
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.tile_count = 0


def test_injected_colour_global_provider_failure_returns_no_layer(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("injected global provider failure")

    monkeypatch.setattr(
        staged_colour_module,
        "build_chunk_invariant_global_stage_from_rows",
        fail,
    )
    with pytest.raises(RuntimeError, match="injected"):
        execute_staged_physical_colour_halation_default(
            _colour_highlight_field(),
            tile_size=37,
            source_row_chunk=31,
            coarse_row_chunk=3,
        )


def test_injected_colour_nonfinite_context_fails_closed(monkeypatch) -> None:
    original = staged_colour_module._background_window

    def nonfinite(base, bounds, audit):
        result = original(base, bounds, audit).copy()
        result[0, 0] = np.nan
        return result

    monkeypatch.setattr(staged_colour_module, "_background_window", nonfinite)
    with pytest.raises(ValueError, match="finite"):
        execute_staged_physical_colour_halation_default(
            _colour_highlight_field(),
            tile_size=37,
            source_row_chunk=31,
            coarse_row_chunk=3,
        )
