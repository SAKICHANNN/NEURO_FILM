import numpy as np
import pytest

from src.film_physics.compact_log_scanner_compiler import (
    CompactLogScannerCompiler,
    apply_compact_log_scanner,
    interpret_negative_scan_relative,
    invert_compact_log_scanner,
)

COMPILER = CompactLogScannerCompiler(
    compiler_id="u6-p4id-synthetic-log-scanner-v1",
    matrix_density_to_log10_rgb=(
        (-0.01914318034784818, -0.12430920920289017, -0.5426234628121562),
        (-0.34150689554450087, -0.5473259673318212, -0.323929450559223),
        (-0.348671066082305, -0.142761433560209, -0.03650384363698002),
    ),
    bias_log10_rgb=(
        0.012682750789860522,
        0.01518857683993641,
        0.03158356724978537,
    ),
)


def test_float32_execution_matches_float64_compiler_tightly() -> None:
    levels = np.linspace(0.0, 1.0, 65, dtype=np.float32)
    density = np.asarray(
        [(a, b, c) for a in levels for b in levels[::8] for c in levels[::8]],
        dtype=np.float32,
    )
    output = apply_compact_log_scanner(density, COMPILER)
    matrix64 = np.asarray(COMPILER.matrix_density_to_log10_rgb, dtype=np.float64)
    bias64 = np.asarray(COMPILER.bias_log10_rgb, dtype=np.float64)
    oracle = np.power(10.0, density.astype(np.float64) @ matrix64 + bias64)
    assert output.dtype == np.float32
    assert np.max(np.abs(output.astype(np.float64) - oracle)) <= 2e-7
    assert np.array_equal(output, apply_compact_log_scanner(density, COMPILER))


def test_each_dye_density_is_nonincreasing_in_every_scanner_channel() -> None:
    levels = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    for channel in range(3):
        density = np.zeros((len(levels), 3), dtype=np.float32)
        density[:, channel] = levels
        output = apply_compact_log_scanner(density, COMPILER)
        assert np.all(np.diff(output, axis=0) <= 0.0)


@pytest.mark.parametrize("bad", [-0.01, np.nan, np.inf])
def test_invalid_density_fails_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        apply_compact_log_scanner(np.full((2, 3), bad, dtype=np.float32), COMPILER)


def test_dtype_and_shape_fail_closed() -> None:
    with pytest.raises(TypeError):
        apply_compact_log_scanner(np.zeros((2, 3), dtype=np.float64), COMPILER)
    with pytest.raises(ValueError):
        apply_compact_log_scanner(np.zeros((2, 2), dtype=np.float32), COMPILER)


def test_negative_interpretation_binds_clear_and_maximum_density_endpoints() -> None:
    endpoints = apply_compact_log_scanner(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float32),
        COMPILER,
    )
    values = interpret_negative_scan_relative(
        endpoints,
        clear_scan_rgb=endpoints[0],
        maximum_density_scan_rgb=endpoints[1],
    )
    assert np.max(np.abs(values[0])) <= 2e-7
    assert np.max(np.abs(values[1] - 1.0)) <= 2e-7


def test_negative_interpretation_rejects_outside_endpoint_values() -> None:
    clear = np.ones(3, dtype=np.float32)
    maximum = np.full(3, 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        interpret_negative_scan_relative(
            np.full((1, 3), 1.1, dtype=np.float32),
            clear_scan_rgb=clear,
            maximum_density_scan_rgb=maximum,
        )


def test_compact_scanner_inverse_recovers_mixed_dye_amounts() -> None:
    levels = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    density = np.asarray(
        [(a, b, c) for a in levels for b in levels for c in levels],
        dtype=np.float32,
    )
    recovered = invert_compact_log_scanner(
        apply_compact_log_scanner(density, COMPILER), COMPILER
    )
    assert np.max(np.abs(recovered - density)) <= 4e-6


def test_compact_scanner_inverse_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        invert_compact_log_scanner(np.zeros((1, 3), dtype=np.float32), COMPILER)
