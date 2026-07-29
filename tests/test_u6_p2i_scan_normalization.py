from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    ScanSignalNormalization,
    derive_scan_signal_normalization,
    scan_signal_normalization_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _normalization() -> ScanSignalNormalization:
    p2f = json.loads(
        (ROOT / "configs/u6_p2f_generic_reversal_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    template = build_operator(
        json.loads(
            (
                ROOT / p2f["parents"]["negative_sensitometry_path"]
            ).read_text(encoding="utf-8")
        )
    )
    reversal = GenericReversalDevelopment(
        template, float(p2f["candidate"]["maximum_relative_layer_exposure"])
    )
    scanner_payload = json.loads(
        (
            ROOT / "configs/u6_p6a_scanner_profile_boundary_v1.json"
        ).read_text(encoding="utf-8")
    )
    scanner = _profile(scanner_payload["profiles"]["scanner_a"])
    return derive_scan_signal_normalization(
        reversal,
        scanner,
        flat_field_shape=(64, 64, 3),
        scanner_stages=("spectral", "flare", "dmax", "mtf"),
    )


def _scan(values: np.ndarray) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.asarray(values),
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("red", "green", "blue"),
    )


def test_endpoints_map_exactly_to_display_black_and_white() -> None:
    normalization = _normalization()
    endpoints = np.asarray(
        [normalization.black_scan_rgb, normalization.white_scan_rgb],
        dtype=np.float64,
    )
    output = normalization.apply(_scan(endpoints)).values
    np.testing.assert_array_equal(
        output, np.asarray([[0.0] * 3, [1.0] * 3])
    )


def test_roundtrip_serialization_and_identity_are_exact() -> None:
    normalization = _normalization()
    replay = ScanSignalNormalization.from_dict(
        json.loads(json.dumps(normalization.to_dict(), sort_keys=True))
    )
    rng = np.random.default_rng(20260729)
    display_values = rng.uniform(0.0, 1.0, size=(97, 11, 3))
    display = PhysicalDomainArray(
        display_values,
        PhysicalDomain.DISPLAY_LINEAR,
        PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
        ("red", "green", "blue"),
    )
    restored = replay.apply(replay.inverse(display)).values
    np.testing.assert_allclose(restored, display_values, rtol=0.0, atol=1e-12)
    assert replay == normalization
    assert scan_signal_normalization_identity(
        replay
    ) == scan_signal_normalization_identity(normalization)


def test_wrong_domain_outside_endpoint_and_noise_derivation_fail() -> None:
    normalization = _normalization()
    outside = np.asarray(normalization.black_scan_rgb)[None, :] - 1e-6
    with pytest.raises(ValueError, match="clipping forbidden"):
        normalization.apply(_scan(outside))
    wrong = PhysicalDomainArray(
        np.full((1, 3), 0.5, np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        normalization.apply(wrong)
    with pytest.raises(ValueError, match="noise"):
        replace(
            normalization,
            scanner_stages=normalization.scanner_stages + ("noise",),
        )
