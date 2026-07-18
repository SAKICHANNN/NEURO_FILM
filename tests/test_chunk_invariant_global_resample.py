from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.filmfx import (
    CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION,
    GLOBAL_RESAMPLE_VERSION,
    GlobalResamplePlan,
    StagedGlobalField,
    build_chunk_invariant_global_stage,
    build_chunk_invariant_global_stage_from_rows,
    build_shape_stable_global_stage,
    plan_chunk_invariant_global_resample,
    plan_shape_stable_global_resample,
    reconstruct_chunk_invariant_full,
    reconstruct_chunk_invariant_tiled,
    reconstruct_chunk_invariant_window,
)


def _field(shape: tuple[int, ...], seed: int = 401) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


@pytest.mark.parametrize(
    ("shape", "sigma", "chunks", "tile_sizes"),
    [
        ((263, 397), 52.0, (1, 5, 19), (37, 64)),
        ((197, 281), (41.0, 69.0), (1, 3, 13), (31, 71)),
        ((89, 113, 3), 39.0, (1, 4, 9), (23, 47)),
    ],
)
def test_full_row_and_tiled_v2_are_byte_identical(shape, sigma, chunks, tile_sizes) -> None:
    field = _field(shape)
    plan = plan_chunk_invariant_global_resample(field.shape[:2], sigma)
    full_stage = build_chunk_invariant_global_stage(field, plan)
    full = reconstruct_chunk_invariant_full(full_stage)
    assert full_stage.plan.version == CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION
    assert full_stage.coarse.flags.writeable is False

    for chunk in chunks:
        calls: list[tuple[int, int]] = []

        def reader(y0: int, y1: int) -> np.ndarray:
            calls.append((y0, y1))
            return field[y0:y1]

        row_stage, metadata = build_chunk_invariant_global_stage_from_rows(
            field.shape,
            plan,
            coarse_row_chunk=chunk,
            reader=reader,
        )
        assert row_stage.coarse.tobytes() == full_stage.coarse.tobytes()
        assert reconstruct_chunk_invariant_full(row_stage).tobytes() == full.tobytes()
        assert metadata.version == CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION
        assert metadata.call_count == len(calls) == len(metadata.calls)
        assert [(record.y0, record.y1) for record in metadata.calls] == calls
        assert metadata.max_row_span < field.shape[0]
        assert metadata.total_read_bytes == sum(record.bytes_read for record in metadata.calls)
        assert metadata.logical_source_bytes == field.nbytes
        assert metadata.coarse_bytes == row_stage.coarse.nbytes

        repeated_stage, repeated_metadata = build_chunk_invariant_global_stage_from_rows(
            field.shape,
            plan,
            coarse_row_chunk=chunk,
            reader=lambda y0, y1: field[y0:y1],
        )
        assert repeated_stage.coarse.tobytes() == row_stage.coarse.tobytes()
        assert repeated_metadata == metadata

    for tile_size in tile_sizes:
        tiled, metadata = reconstruct_chunk_invariant_tiled(full_stage, tile_size=tile_size)
        assert tiled.tobytes() == full.tobytes()
        assert metadata.version == CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION


def test_v2_window_uses_global_coordinates() -> None:
    field = _field((91, 127, 2))
    stage = build_chunk_invariant_global_stage(
        field,
        plan_chunk_invariant_global_resample(field.shape[:2], 38.0),
    )
    full = reconstruct_chunk_invariant_full(stage)
    window = reconstruct_chunk_invariant_window(stage, y0=7, y1=83, x0=11, x1=119)
    assert window.tobytes() == full[7:83, 11:119].tobytes()


def test_v2_closes_last_cell_overrun_without_rewriting_v1() -> None:
    field = _field((27, 27))
    v1_plan = plan_shape_stable_global_resample(field.shape, 12.0)
    assert v1_plan.version == GLOBAL_RESAMPLE_VERSION
    with pytest.raises(ValueError, match="shape-mismatch"):
        build_shape_stable_global_stage(field, v1_plan)

    v2_plan = plan_chunk_invariant_global_resample(field.shape, 12.0)
    stage = build_chunk_invariant_global_stage(field, v2_plan)
    row_stage, _ = build_chunk_invariant_global_stage_from_rows(
        field.shape,
        v2_plan,
        coarse_row_chunk=1,
        reader=lambda y0, y1: field[y0:y1],
    )
    assert stage.coarse.tobytes() == row_stage.coarse.tobytes()


def test_versions_are_strictly_separated() -> None:
    field = _field((83, 107))
    v1 = plan_shape_stable_global_resample(field.shape, 38.0)
    v2 = plan_chunk_invariant_global_resample(field.shape, 38.0)
    with pytest.raises(ValueError, match="supported GlobalResamplePlan"):
        build_shape_stable_global_stage(field, v2)
    with pytest.raises(ValueError, match="chunk-invariant"):
        build_chunk_invariant_global_stage(field, v1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.astype(np.float64),
        lambda rows: rows[:-1],
        lambda rows: np.full_like(rows, np.nan),
    ],
)
def test_row_reader_contract_fails_closed(mutation) -> None:
    field = _field((83, 107))
    plan = plan_chunk_invariant_global_resample(field.shape, 38.0)

    def reader(y0: int, y1: int) -> np.ndarray:
        return mutation(field[y0:y1])

    with pytest.raises(ValueError):
        build_chunk_invariant_global_stage_from_rows(
            field.shape,
            plan,
            coarse_row_chunk=1,
            reader=reader,
        )


def test_row_builder_rejects_invalid_contracts_and_reader_failure() -> None:
    field = _field((83, 107))
    plan = plan_chunk_invariant_global_resample(field.shape, 38.0)
    finite_plan = plan_chunk_invariant_global_resample((19, 31), 0.0)

    with pytest.raises(ValueError, match="finite-halo"):
        build_chunk_invariant_global_stage_from_rows(
            (19, 31),
            finite_plan,
            coarse_row_chunk=1,
            reader=lambda y0, y1: _field((19, 31))[y0:y1],
        )
    for chunk in (0, True, plan.coarse_shape[0]):
        with pytest.raises(ValueError, match="coarse_row_chunk"):
            build_chunk_invariant_global_stage_from_rows(
                field.shape,
                plan,
                coarse_row_chunk=chunk,
                reader=lambda y0, y1: field[y0:y1],
            )
    with pytest.raises(ValueError, match="field_shape"):
        build_chunk_invariant_global_stage_from_rows(
            (82, 107),
            plan,
            coarse_row_chunk=1,
            reader=lambda y0, y1: field[y0:y1],
        )

    calls = 0

    def failing_reader(y0: int, y1: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected reader failure")
        return field[y0:y1]

    with pytest.raises(RuntimeError, match="injected"):
        build_chunk_invariant_global_stage_from_rows(
            field.shape,
            plan,
            coarse_row_chunk=1,
            reader=failing_reader,
        )


def test_v2_stage_validation_rejects_writable_and_forged_geometry() -> None:
    field = _field((83, 107))
    plan = plan_chunk_invariant_global_resample(field.shape, 38.0)
    stage = build_chunk_invariant_global_stage(field, plan)
    writable = StagedGlobalField(plan=plan, coarse=stage.coarse.copy())
    with pytest.raises(ValueError, match="read-only"):
        reconstruct_chunk_invariant_full(writable)

    forged = dataclasses.replace(plan, scale_factors=(99.0, 99.0))
    with pytest.raises(ValueError, match="scale_factors"):
        build_chunk_invariant_global_stage(field, forged)


def test_v2_plan_dataclass_remains_the_versioned_public_contract() -> None:
    plan = plan_chunk_invariant_global_resample((83, 107), 38.0)
    assert isinstance(plan, GlobalResamplePlan)
    assert plan.version == CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION
