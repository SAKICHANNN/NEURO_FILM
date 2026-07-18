from __future__ import annotations

import numpy as np
import pytest

from src.filmfx import (
    GLOBAL_RESAMPLE_VERSION,
    GlobalResamplePlan,
    StagedGlobalField,
    build_shape_stable_global_stage,
    plan_shape_stable_global_resample,
    reconstruct_shape_stable_full,
    reconstruct_shape_stable_tiled,
    reconstruct_shape_stable_window,
)


def _field(shape: tuple[int, ...], seed: int = 71) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


@pytest.mark.parametrize("tile_size", [37, 64])
def test_u1_6g0_counterexample_is_now_byte_identical(tile_size: int) -> None:
    field = _field((257, 389))
    plan = plan_shape_stable_global_resample(field.shape, 52.0)
    stage = build_shape_stable_global_stage(field, plan)
    full = reconstruct_shape_stable_full(stage)
    tiled, metadata = reconstruct_shape_stable_tiled(stage, tile_size=tile_size)
    assert full.tobytes() == tiled.tobytes()
    assert float(np.abs(full - tiled).max()) == 0.0
    assert metadata.version == GLOBAL_RESAMPLE_VERSION
    assert metadata.coarse_bytes == stage.coarse.nbytes
    assert metadata.output_bytes == tiled.nbytes
    assert metadata.max_window_shape == (tile_size, tile_size)


@pytest.mark.parametrize(
    ("shape", "sigma", "tile_size"),
    [
        ((19, 31), 0.0, 7),
        ((37, 53), (2.0, 11.0), 13),
        ((83, 107, 3), 38.0, 29),
        ((9, 13, 2), (80.0, 7.0), 4),
    ],
)
def test_irregular_fields_are_tile_invariant(shape, sigma, tile_size) -> None:
    field = _field(shape)
    plan = plan_shape_stable_global_resample(field.shape[:2], sigma)
    stage = build_shape_stable_global_stage(field, plan)
    full = reconstruct_shape_stable_full(stage)
    tiled, metadata = reconstruct_shape_stable_tiled(stage, tile_size=tile_size)
    assert full.tobytes() == tiled.tobytes()
    assert stage.coarse.flags.writeable is False
    assert metadata.coarse_shape == stage.coarse.shape
    second, second_metadata = reconstruct_shape_stable_tiled(stage, tile_size=tile_size)
    assert second.tobytes() == tiled.tobytes()
    assert second_metadata == metadata


def test_window_uses_global_coordinates() -> None:
    field = _field((41, 67, 2))
    stage = build_shape_stable_global_stage(
        field,
        plan_shape_stable_global_resample(field.shape[:2], 24.0),
    )
    full = reconstruct_shape_stable_full(stage)
    window = reconstruct_shape_stable_window(stage, y0=7, y1=29, x0=11, x1=61)
    assert window.tobytes() == full[7:29, 11:61].tobytes()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: plan_shape_stable_global_resample((0, 7), 2.0),
        lambda: plan_shape_stable_global_resample((7, 9), -1.0),
        lambda: plan_shape_stable_global_resample((7, 9), float("nan")),
        lambda: plan_shape_stable_global_resample((7, 9), (1.0, 2.0, 3.0)),
        lambda: plan_shape_stable_global_resample((7, 9), 2.0, max_direct_radius=True),
    ],
)
def test_plan_rejects_invalid_contracts(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_stage_and_windows_fail_closed() -> None:
    field = _field((17, 23))
    plan = plan_shape_stable_global_resample(field.shape, 18.0)
    with pytest.raises(ValueError):
        build_shape_stable_global_stage(field.astype(np.float64), plan)
    with pytest.raises(ValueError):
        build_shape_stable_global_stage(field[:-1], plan)

    stage = build_shape_stable_global_stage(field, plan)
    with pytest.raises(ValueError):
        reconstruct_shape_stable_window(stage, y0=0, y1=18, x0=0, x1=23)
    with pytest.raises(ValueError):
        reconstruct_shape_stable_window(stage, y0=3, y1=3, x0=0, x1=4)
    with pytest.raises(ValueError):
        reconstruct_shape_stable_tiled(stage, tile_size=0)

    writable = stage.coarse.copy()
    invalid = StagedGlobalField(plan=plan, coarse=writable)
    with pytest.raises(ValueError, match="read-only"):
        reconstruct_shape_stable_full(invalid)

    forged_plan = GlobalResamplePlan(
        version=GLOBAL_RESAMPLE_VERSION,
        source_shape=plan.source_shape,
        coarse_shape=plan.coarse_shape,
        spatial_sigma=plan.spatial_sigma,
        coarse_sigma=plan.coarse_sigma,
        scale_factors=(99.0, 99.0),
        truncate=plan.truncate,
    )
    with pytest.raises(ValueError, match="scale_factors"):
        build_shape_stable_global_stage(field, forged_plan)
