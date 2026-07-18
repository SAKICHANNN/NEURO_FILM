from __future__ import annotations

import numpy as np
import pytest

from src.filmfx import STREAMING_PERCENTILE_VERSION, exact_streaming_percentiles


PERCENTILES = (0.0, 1.0, 50.0, 99.7, 99.8, 100.0)


def _factory(array: np.ndarray, cuts: tuple[int, ...]):
    flat = array.reshape(-1)

    def produce():
        start = 0
        for stop in cuts:
            yield flat[start:stop]
            start = stop
        yield flat[start:]

    return produce


def _assert_numpy_parity(array: np.ndarray, cuts: tuple[int, ...]) -> None:
    result = exact_streaming_percentiles(
        _factory(array, cuts), count=array.size, percentiles=PERCENTILES
    )
    expected = np.percentile(array, PERCENTILES, method="linear")
    assert np.asarray(result.values, dtype=np.float64).tobytes() == expected.astype(np.float64).tobytes()
    assert result.version == STREAMING_PERCENTILE_VERSION
    assert result.count == array.size
    assert result.passes == 2
    assert result.histogram_bytes <= (1 + 2 * len(PERCENTILES)) * 65536 * 8


def test_random_irregular_chunks_match_numpy_exactly() -> None:
    array = np.random.default_rng(81).normal(size=(101, 103)).astype(np.float32)
    _assert_numpy_parity(array, (1, 777, 4099, 10000))


def test_duplicates_extremes_negatives_and_signed_zero_match_numpy() -> None:
    info = np.finfo(np.float32)
    array = np.asarray(
        [-0.0, 0.0, -7.0, -7.0, 2.5, 2.5, info.min, info.max, info.tiny, -info.tiny] * 37,
        dtype=np.float32,
    )
    _assert_numpy_parity(array, (3, 19, 113, 211))


def test_random_finite_float32_bit_patterns_match_numpy_exactly() -> None:
    bits = np.random.default_rng(123).integers(0, 2**32, size=50000, dtype=np.uint32)
    array = bits.view(np.float32)
    array = array[np.isfinite(array)]
    percentiles = tuple(np.linspace(0.0, 100.0, 101)) + (0.01, 0.1, 13.37, 99.7, 99.8, 99.99)
    result = exact_streaming_percentiles(
        _factory(array, tuple(range(997, array.size, 997))),
        count=array.size,
        percentiles=percentiles,
    )
    expected = np.percentile(array, percentiles, method="linear")
    assert np.asarray(result.values, dtype=np.float64).tobytes() == expected.astype(np.float64).tobytes()


def test_chunk_boundaries_do_not_change_result() -> None:
    array = np.random.default_rng(19).uniform(-4.0, 9.0, size=2003).astype(np.float32)
    first = exact_streaming_percentiles(_factory(array, (17, 1000)), count=array.size, percentiles=(13.2, 99.7))
    second = exact_streaming_percentiles(_factory(array, (1, 2, 3, 2002)), count=array.size, percentiles=(13.2, 99.7))
    assert first.values == second.values
    assert first.stream_sha256 == second.stream_sha256
    assert first.rank_pairs == second.rank_pairs


def test_changed_second_pass_fails_closed() -> None:
    array = np.arange(100, dtype=np.float32)
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        current = array.copy()
        if calls == 2:
            current[50] += 1.0
        yield current

    with pytest.raises(ValueError, match="changed"):
        exact_streaming_percentiles(changed, count=array.size, percentiles=(50.0,))


@pytest.mark.parametrize(
    ("factory", "count", "percentiles", "message"),
    [
        (lambda: [np.arange(4, dtype=np.float64)], 4, (50.0,), "float32"),
        (lambda: [np.asarray([0.0, np.nan], dtype=np.float32)], 2, (50.0,), "finite"),
        (lambda: [np.arange(3, dtype=np.float32)], 4, (50.0,), "first pass"),
        (lambda: [np.arange(3, dtype=np.float32)], 3, (), "percentiles"),
        (lambda: [np.arange(3, dtype=np.float32)], 3, (101.0,), "percentiles"),
    ],
)
def test_invalid_contracts_fail_closed(factory, count, percentiles, message) -> None:
    with pytest.raises(ValueError, match=message):
        exact_streaming_percentiles(factory, count=count, percentiles=percentiles)


def test_second_pass_count_mismatch_fails_closed() -> None:
    array = np.arange(10, dtype=np.float32)
    calls = 0

    def shortened():
        nonlocal calls
        calls += 1
        yield array if calls == 1 else array[:-1]

    with pytest.raises(ValueError, match="second pass"):
        exact_streaming_percentiles(shortened, count=10, percentiles=(50.0,))
