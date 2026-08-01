from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_multispectral_scanner_band_compiler import (
    evaluate_multispectral_scanner_band_compiler,
    load_contract,
    write_report,
)
from src.film_physics.multispectral_scanner import (
    IdealBandProfile,
    apply_xyz_compiler,
    fit_nonnegative_xyz_compiler,
    sample_ideal_bands,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p6x_multispectral_scanner_band_compiler_v1.json"
)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return evaluate_multispectral_scanner_band_compiler(
        ROOT, load_contract(CONTRACT)
    )


def test_ideal_band_sampling_is_explicit_linear_interpolation() -> None:
    wavelength = np.asarray([400.0, 410.0, 420.0], dtype=np.float64)
    spectra = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float64)
    profile = IdealBandProfile("test", (405.0, 410.0, 415.0))
    actual = sample_ideal_bands(spectra, wavelength, profile)
    np.testing.assert_array_equal(
        actual, np.asarray([[0.25, 0.5, 0.75]], dtype=np.float64)
    )


def test_nonnegative_compiler_preserves_clear_response() -> None:
    bands = np.asarray(
        [[0.1, 0.4, 0.8], [0.7, 0.2, 0.3], [0.9, 0.8, 0.1]],
        dtype=np.float64,
    )
    target = np.asarray(
        [[0.2, 0.3, 0.4], [0.5, 0.3, 0.2], [0.7, 0.8, 0.1]],
        dtype=np.float64,
    )
    clear = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
    matrix = fit_nonnegative_xyz_compiler(bands, target, clear)
    assert np.all(matrix >= 0.0)
    np.testing.assert_allclose(np.sum(matrix, axis=0), clear, rtol=0.0, atol=1e-15)
    output = apply_xyz_compiler(np.ones((1, 3), dtype=np.float64), matrix)
    np.testing.assert_allclose(output[0], clear, rtol=0.0, atol=1e-15)


def test_full_evaluator_keeps_development_and_fresh_stress_separate(
    report: dict[str, object],
) -> None:
    populations = report["populations"]
    assert isinstance(populations, dict)
    assert populations["measured_rows"] == 8640
    assert populations["measured_split_rows"] == {
        "development": 3456,
        "held-set": 1728,
        "held-slide": 2304,
        "joint-held": 1152,
    }
    assert populations["cave_representative_scene_count"] == 31
    assert populations["cave_representative_rows"] > 4096
    assert report["checks"]["measured_split_coverage"] is True
    assert report["checks"]["cave_audit"] is True


def test_compilers_are_nonnegative_bounded_and_clear_normalized(
    report: dict[str, object],
) -> None:
    for row in report["profiles"].values():
        matrix = np.asarray(row["compiler_matrix"], dtype=np.float64)
        assert np.all(matrix >= 0.0)
    assert report["checks"]["clear_response"] is True
    assert report["checks"]["candidate_bounded"] is True


def test_report_encoding_is_repeatable(
    report: dict[str, object], tmp_path: Path
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert write_report(report, first) == write_report(report, second)
    assert first.read_bytes() == second.read_bytes()


def test_parent_hash_drift_fails_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["cave_official_tail_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="cave_official_tail_path hash mismatch"):
        evaluate_multispectral_scanner_band_compiler(ROOT, contract)


def test_invalid_or_reordered_profiles_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        IdealBandProfile("bad", (500.0, 490.0, 510.0))
    profile = IdealBandProfile("outside", (390.0, 500.0, 600.0))
    with pytest.raises(ValueError, match="inside the wavelength grid"):
        sample_ideal_bands(
            np.ones((1, 3), dtype=np.float64),
            np.asarray([400.0, 500.0, 600.0], dtype=np.float64),
            profile,
        )
