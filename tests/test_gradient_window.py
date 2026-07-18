from __future__ import annotations

import numpy as np
import pytest

from src.filmfx import (
    GRADIENT_WINDOW_VERSION,
    array_scalar_window_reader,
    coordinate_gradient_window,
    coordinate_gradient_window_from_array,
)


def _field(shape=(37, 53), seed=211) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _full(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradient_y, gradient_x = np.gradient(field)
    return gradient_y.astype(np.float32, copy=False), gradient_x.astype(np.float32, copy=False)


def test_full_window_is_byte_identical_to_current_numpy_gradient() -> None:
    field = _field()
    expected_y, expected_x = _full(field)
    actual_y, actual_x, metadata = coordinate_gradient_window_from_array(
        field, y0=0, y1=field.shape[0], x0=0, x1=field.shape[1]
    )
    assert actual_y.tobytes() == expected_y.tobytes()
    assert actual_x.tobytes() == expected_x.tobytes()
    assert metadata.version == GRADIENT_WINDOW_VERSION
    assert metadata.expanded_bounds == (0, field.shape[0], 0, field.shape[1])
    assert metadata.expanded_input_bytes == field.nbytes
    assert metadata.output_bytes == actual_y.nbytes + actual_x.nbytes


@pytest.mark.parametrize(
    "bounds",
    [
        (7, 19, 11, 31),
        (0, 9, 13, 29),
        (29, 37, 17, 41),
        (9, 27, 0, 8),
        (5, 23, 45, 53),
        (0, 1, 0, 1),
        (36, 37, 52, 53),
        (12, 13, 7, 46),
        (4, 31, 22, 23),
    ],
)
def test_interior_edges_corners_and_thin_cores_match_full(bounds) -> None:
    field = _field()
    expected_y, expected_x = _full(field)
    y0, y1, x0, x1 = bounds
    actual_y, actual_x, metadata = coordinate_gradient_window_from_array(
        field, y0=y0, y1=y1, x0=x0, x1=x1
    )
    assert actual_y.tobytes() == expected_y[y0:y1, x0:x1].tobytes()
    assert actual_x.tobytes() == expected_x[y0:y1, x0:x1].tobytes()
    assert metadata.expanded_bounds == (
        max(0, y0 - 1),
        min(field.shape[0], y1 + 1),
        max(0, x0 - 1),
        min(field.shape[1], x1 + 1),
    )
    assert metadata.expanded_shape[0] <= metadata.core_shape[0] + 2
    assert metadata.expanded_shape[1] <= metadata.core_shape[1] + 2
    assert actual_y.flags.writeable is False
    assert actual_x.flags.writeable is False


@pytest.mark.parametrize("tile_size", [7, 16, 29])
def test_irregular_2d_tile_assembly_is_byte_identical(tile_size: int) -> None:
    field = _field((41, 67))
    expected_y, expected_x = _full(field)
    actual_y = np.empty_like(field)
    actual_x = np.empty_like(field)
    reader = array_scalar_window_reader(field)
    for y0 in range(0, field.shape[0], tile_size):
        y1 = min(field.shape[0], y0 + tile_size)
        for x0 in range(0, field.shape[1], tile_size):
            x1 = min(field.shape[1], x0 + tile_size)
            gy, gx, _ = coordinate_gradient_window(
                reader, field.shape, y0=y0, y1=y1, x0=x0, x1=x1
            )
            actual_y[y0:y1, x0:x1] = gy
            actual_x[y0:y1, x0:x1] = gx
    assert actual_y.tobytes() == expected_y.tobytes()
    assert actual_x.tobytes() == expected_x.tobytes()


@pytest.mark.parametrize("row_chunk", [1, 8, 19])
def test_row_chunk_assembly_is_byte_identical(row_chunk: int) -> None:
    field = _field((43, 59))
    expected_y, expected_x = _full(field)
    actual_y = np.empty_like(field)
    actual_x = np.empty_like(field)
    reader = array_scalar_window_reader(field)
    for y0 in range(0, field.shape[0], row_chunk):
        y1 = min(field.shape[0], y0 + row_chunk)
        gy, gx, _ = coordinate_gradient_window(
            reader, field.shape, y0=y0, y1=y1, x0=0, x1=field.shape[1]
        )
        actual_y[y0:y1] = gy
        actual_x[y0:y1] = gx
    assert actual_y.tobytes() == expected_y.tobytes()
    assert actual_x.tobytes() == expected_x.tobytes()


def test_reader_receives_only_the_exact_expanded_window() -> None:
    field = _field()
    calls = []

    def reader(y0, y1, x0, x1):
        calls.append((y0, y1, x0, x1))
        return field[y0:y1, x0:x1]

    coordinate_gradient_window(reader, field.shape, y0=8, y1=13, x0=17, x1=23)
    assert calls == [(7, 14, 16, 24)]


def test_repeated_execution_and_metadata_are_identical() -> None:
    field = _field()
    kwargs = dict(y0=3, y1=29, x0=5, x1=47)
    first_y, first_x, first_metadata = coordinate_gradient_window_from_array(field, **kwargs)
    second_y, second_x, second_metadata = coordinate_gradient_window_from_array(field, **kwargs)
    assert first_y.tobytes() == second_y.tobytes()
    assert first_x.tobytes() == second_x.tobytes()
    assert first_metadata == second_metadata


@pytest.mark.parametrize(
    "field",
    [
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 4), dtype=np.float64),
        np.full((3, 4), np.nan, dtype=np.float32),
        np.zeros((3, 4, 1), dtype=np.float32),
    ],
)
def test_array_reader_rejects_invalid_fields(field) -> None:
    with pytest.raises(ValueError):
        array_scalar_window_reader(field)


@pytest.mark.parametrize(
    "bounds",
    [
        (-1, 2, 0, 2),
        (2, 2, 0, 2),
        (0, 38, 0, 2),
        (0, 2, 8, 8),
        (0, 2, 0, 54),
        (False, 2, 0, 2),
    ],
)
def test_invalid_bounds_fail_closed(bounds) -> None:
    field = _field()
    with pytest.raises(ValueError):
        coordinate_gradient_window_from_array(
            field, y0=bounds[0], y1=bounds[1], x0=bounds[2], x1=bounds[3]
        )


@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (lambda *_: np.zeros((4, 4), dtype=np.float64), "float32"),
        (lambda *_: np.full((4, 4), np.nan, dtype=np.float32), "finite"),
        (lambda *_: np.zeros((3, 4), dtype=np.float32), "shape"),
        (lambda *_: np.zeros((4, 4, 1), dtype=np.float32), "2-D"),
        (lambda *_: (_ for _ in ()).throw(RuntimeError("boom")), "reader failed"),
    ],
)
def test_reader_contract_failures_are_rejected(reader, message) -> None:
    with pytest.raises(ValueError, match=message):
        coordinate_gradient_window(reader, (4, 4), y0=0, y1=4, x0=0, x1=4)
