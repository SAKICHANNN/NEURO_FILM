from __future__ import annotations

import copy

import numpy as np
import pytest

from src.film_physics.scanner_safe_residual_profile import (
    ScannerSafeResidualProfile,
    apply_scanner_safe_residual_profile,
    identity_scanner_safe_residual_profile,
)

EVIDENCE = "1" * 64


def test_profile_canonical_roundtrip_and_identity_execution() -> None:
    profile = identity_scanner_safe_residual_profile(evidence_sha256=EVIDENCE)
    payload = profile.to_payload()
    assert ScannerSafeResidualProfile.from_payload(payload) == profile
    source = np.linspace(0.0, 1.0, 45, dtype=np.float64).reshape(3, 5, 3)
    output, receipt = apply_scanner_safe_residual_profile(source, profile)
    np.testing.assert_array_equal(output, source)
    assert receipt.limited_pixel_fraction == 0.0


def test_nonidentity_profile_is_bounded_without_clipping() -> None:
    profile = ScannerSafeResidualProfile(
        profile_id="stress",
        matrix=((1.4, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.0, 1.3)),
        bias=(0.2, -0.1, 0.0),
        evidence_sha256=EVIDENCE,
    )
    source = np.asarray([[[0.8, 0.2, 0.9], [0.1, 0.5, 0.4]]], dtype=np.float64)
    output, receipt = apply_scanner_safe_residual_profile(source, profile)
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert receipt.limited_pixel_fraction > 0.0


def test_profile_tamper_is_rejected() -> None:
    payload = identity_scanner_safe_residual_profile(evidence_sha256=EVIDENCE).to_payload()
    tampered = copy.deepcopy(payload)
    tampered["bias"][0] = 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        ScannerSafeResidualProfile.from_payload(tampered)


def test_profile_rejects_noncanonical_or_invalid_fields() -> None:
    with pytest.raises(ValueError, match="evidence hash"):
        identity_scanner_safe_residual_profile(evidence_sha256="A" * 64)
    payload = identity_scanner_safe_residual_profile(evidence_sha256=EVIDENCE).to_payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields drift"):
        ScannerSafeResidualProfile.from_payload(payload)
