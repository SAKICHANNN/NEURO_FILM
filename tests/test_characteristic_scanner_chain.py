import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_characteristic_prior import compile_prior
from src.film_physics.characteristic_scanner_chain import (
    render_characteristic_scanner_positive,
)
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler
from src.film_physics.relative_display_characteristic_ingress import (
    relative_display_to_finite_density_transmittance,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixtures():
    trace_path = ROOT / "configs/data/kodak_250d_characteristic_curve_pixels_v1.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    prior, _ = compile_prior(trace, source_evidence_id="0" * 64)
    config = json.loads(
        (ROOT / "configs/u6_p4if_negative_scanner_inverse_d0_v2.json").read_text(
            encoding="utf-8"
        )
    )
    row = config["compiler"]
    compiler = CompactLogScannerCompiler(
        row["compiler_id"],
        tuple(tuple(values) for values in row["matrix_density_to_log10_rgb"]),
        tuple(row["bias_log10_rgb"]),
    )
    return prior, compiler


def test_base_characteristic_transmittance_returns_normalized_density_response() -> None:
    prior, compiler = _fixtures()
    source = np.linspace(0.0, 1.0, 99, dtype=np.float32).reshape(3, 11, 3)
    density, transmittance, _ = relative_display_to_finite_density_transmittance(
        source, prior
    )
    expected = np.empty_like(density)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.density_bounds
        expected[..., channel] = (density[..., channel] - lower) / (upper - lower)
    output, receipt = render_characteristic_scanner_positive(
        source, transmittance, prior=prior, compiler=compiler
    )
    np.testing.assert_allclose(output, expected, atol=7e-7, rtol=0.0)
    assert receipt["minimum_shared_scale"] == 1.0


def test_out_of_envelope_structure_is_direction_bounded() -> None:
    prior, compiler = _fixtures()
    source = np.full((5, 7, 3), 0.5, dtype=np.float32)
    _, transmittance, _ = relative_display_to_finite_density_transmittance(source, prior)
    direction = np.array([0.7, -0.5, 0.2], dtype=np.float64)
    structured = np.ascontiguousarray(
        transmittance.astype(np.float64) * np.power(10.0, -direction), dtype=np.float32
    )
    output, receipt = render_characteristic_scanner_positive(
        source, structured, prior=prior, compiler=compiler
    )
    assert receipt["limited_fraction"] == 1.0
    assert np.all(output >= -4e-6) and np.all(output <= 1.0 + 4e-6)


def test_zero_transmittance_fails_closed() -> None:
    prior, compiler = _fixtures()
    source = np.full((1, 1, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="structured transmittance"):
        render_characteristic_scanner_positive(
            source, np.zeros_like(source), prior=prior, compiler=compiler
        )


def test_material_density_overshoot_still_fails_closed() -> None:
    prior, compiler = _fixtures()
    source = np.ones((1, 1, 3), dtype=np.float32)
    _, transmittance, _ = relative_display_to_finite_density_transmittance(source, prior)
    outside = np.ascontiguousarray(transmittance * np.float32(0.9))
    with pytest.raises(ValueError, match="observed bounds"):
        render_characteristic_scanner_positive(
            source, outside, prior=prior, compiler=compiler
        )
