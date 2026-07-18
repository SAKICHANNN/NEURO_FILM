from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import src.filmfx as filmfx_package
import src.filmfx.staged_density_adapter as adapter_module
from src.filmfx import composite_layers, execute_staged_density_halation_default
from src.filmfx.staged_density_adapter import (
    STAGED_DENSITY_ADAPTER_VERSION,
    render_staged_density_halation_research,
)


def _highlight_field(shape=(131, 197, 3), seed=451) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = rng.uniform(0.01, 0.28, size=shape).astype(np.float32)
    hot = np.exp(
        -((y - shape[0] * 0.43) ** 2 + (x - shape[1] * 0.61) ** 2) / 190.0
    ).astype(np.float32)
    base += hot[..., None] * np.asarray([0.94, 0.76, 0.44], np.float32)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _contains_mutable_or_array(value: object) -> bool:
    if isinstance(value, (np.ndarray, list, dict, set)):
        return True
    if dataclasses.is_dataclass(value):
        return any(
            _contains_mutable_or_array(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple):
        return any(_contains_mutable_or_array(item) for item in value)
    return False


@pytest.mark.parametrize("row_chunk", [17, 64])
@pytest.mark.parametrize("output_margin", [0, 4])
def test_adapter_is_float_and_srgb8_identical_to_current_compositor(
    row_chunk: int, output_margin: int
) -> None:
    base = _highlight_field()
    layer, _ = execute_staged_density_halation_default(
        base, tile_size=37, source_row_chunk=31, coarse_row_chunk=3
    )
    reference = composite_layers(base, [layer], output_margin=output_margin)
    actual, metadata = render_staged_density_halation_research(
        base,
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        composite_row_chunk=row_chunk,
        output_margin=output_margin,
    )
    assert actual.tobytes() == reference.tobytes()
    assert np.rint(actual * 255).astype(np.uint8).tobytes() == np.rint(
        reference * 255
    ).astype(np.uint8).tobytes()
    assert metadata.version == STAGED_DENSITY_ADAPTER_VERSION
    assert metadata.public_ndarray_count == 1
    assert metadata.scratch_bytes == 0
    assert not metadata.renderer_integration_allowed


def test_adapter_preserves_input_and_returns_array_free_metadata() -> None:
    base = _highlight_field(seed=453)
    before = base.tobytes()
    writeable = base.flags.writeable
    output, metadata = render_staged_density_halation_research(
        base,
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        composite_row_chunk=17,
    )
    assert base.tobytes() == before
    assert base.flags.writeable == writeable
    assert isinstance(output, np.ndarray)
    assert not _contains_mutable_or_array(metadata)
    assert metadata.returned_composite_bytes == output.nbytes
    assert metadata.private_layer_bytes == base.shape[0] * base.shape[1] * 4 * 4


def test_adapter_repeats_output_and_metadata() -> None:
    base = _highlight_field(seed=454)
    parameters = dict(
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        composite_row_chunk=19,
        output_margin=4,
    )
    first, first_metadata = render_staged_density_halation_research(base, **parameters)
    second, second_metadata = render_staged_density_halation_research(base, **parameters)
    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"composite_row_chunk": 0},
        {"composite_row_chunk": True},
        {"output_margin": -1},
        {"output_margin": 33},
    ],
)
def test_adapter_rejects_invalid_orchestration_controls(kwargs) -> None:
    parameters = dict(
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        composite_row_chunk=17,
        output_margin=0,
    )
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        render_staged_density_halation_research(_highlight_field(), **parameters)


def test_adapter_propagates_executor_failure_without_result(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("injected executor failure")

    monkeypatch.setattr(adapter_module, "execute_staged_density_halation_default", fail)
    with pytest.raises(RuntimeError, match="injected executor"):
        render_staged_density_halation_research(
            _highlight_field(),
            tile_size=37,
            source_row_chunk=31,
            coarse_row_chunk=3,
            composite_row_chunk=17,
        )


def test_adapter_propagates_second_row_failure_without_result(monkeypatch) -> None:
    original = adapter_module.composite_layers
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-row failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "composite_layers", fail_second)
    with pytest.raises(RuntimeError, match="injected second-row"):
        render_staged_density_halation_research(
            _highlight_field(),
            tile_size=37,
            source_row_chunk=31,
            coarse_row_chunk=3,
            composite_row_chunk=17,
        )
    assert calls == 2


def test_adapter_rejects_nonfinite_compositor_result(monkeypatch) -> None:
    original = adapter_module.composite_layers

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0, 0, 0] = np.nan
        return result

    monkeypatch.setattr(adapter_module, "composite_layers", nonfinite)
    with pytest.raises(ValueError, match="invalid float32 RGB"):
        render_staged_density_halation_research(
            _highlight_field(),
            tile_size=37,
            source_row_chunk=31,
            coarse_row_chunk=3,
            composite_row_chunk=17,
        )


def test_adapter_is_not_exported_from_package_root() -> None:
    assert not hasattr(filmfx_package, "render_staged_density_halation_research")
    assert not hasattr(filmfx_package, "StagedDensityAdapterMetadata")
