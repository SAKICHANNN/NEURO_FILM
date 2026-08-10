from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.local_intensity_reciprocity import load_contract, run_audit
from src.film_physics.reciprocity import (
    IntensityConditionedReciprocityProfile,
    ReciprocityDomainError,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ab_local_intensity_reciprocity_v1.json"


def _profile() -> IntensityConditionedReciprocityProfile:
    return IntensityConditionedReciprocityProfile(
        "test", 1.20, 1.43, -3.0, 1.5
    )


def test_local_profile_is_identity_at_reference_and_changes_long_contrast() -> None:
    profile = _profile()
    rates = np.exp2(np.linspace(-10.0, 4.0, 257))
    assert np.array_equal(profile.effective_exposure(rates, 1.0), rates)
    output = profile.effective_exposure(rates, 600.0)
    contrast_drift = np.log(output[-1] / output[0]) - np.log(rates[-1] / rates[0])
    assert contrast_drift > 0.8
    assert np.all(np.diff(output) > 0.0)


def test_local_profile_rejects_invalid_parameters_and_rates() -> None:
    with pytest.raises(ReciprocityDomainError):
        IntensityConditionedReciprocityProfile("bad", 0.9, 1.4, -3.0, 1.5)
    with pytest.raises(ReciprocityDomainError):
        IntensityConditionedReciprocityProfile("bad", 1.2, 1.4, -3.0, 0.0)
    with pytest.raises(ReciprocityDomainError):
        _profile().effective_exposure([0.0], 10.0)


def test_formal_local_intensity_audit_passes_frozen_gates() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["incident_rate_count"] == 257
    assert report["measurements"]["longest_time_global_shape_rmse"] > 0.30
    assert report["measurements"]["maximum_any_global_log_contrast_drift"] < 1e-12
