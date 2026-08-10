from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.reciprocity_compiler import load_contract, run_audit
from src.film_physics.reciprocity import (
    DocumentedIdentityInterval,
    PowerReciprocityProfile,
    ReciprocityDomainError,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2aa_reciprocity_compiler_v1.json"


def test_power_profile_is_identity_then_exactly_invertible() -> None:
    profile = PowerReciprocityProfile("hp5", 1.31)
    metered = np.asarray([0.001, 0.1, 1.0, 10.0, 600.0])
    corrected = profile.corrected_time_seconds(metered)
    assert np.array_equal(corrected[:3], metered[:3])
    assert corrected[3] == pytest.approx(10.0**1.31)
    assert np.allclose(profile.effective_time_seconds(corrected), metered, rtol=1e-14)


def test_reciprocity_domains_fail_closed() -> None:
    with pytest.raises(ReciprocityDomainError):
        PowerReciprocityProfile("bad", 0.99)
    profile = PowerReciprocityProfile("valid", 1.25)
    for invalid in ([0.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ReciprocityDomainError):
            profile.corrected_time_seconds(invalid)
    with pytest.raises(ReciprocityDomainError):
        profile.effective_exposure([-0.1], [10.0])


def test_documented_identity_interval_rejects_extrapolation() -> None:
    profile = DocumentedIdentityInterval("vision3-50d", 0.001, 1.0)
    times = np.asarray([0.001, 0.01, 1.0])
    assert np.array_equal(profile.corrected_time_seconds(times), times)
    with pytest.raises(ReciprocityDomainError, match="outside"):
        profile.corrected_time_seconds([1.0001])


def test_formal_audit_obeys_frozen_contract() -> None:
    contract = load_contract(CONTRACT)
    if any(
        not (ROOT / source["local_research_copy"]).is_file()
        for source in contract["sources"].values()
    ):
        pytest.skip("official source PDFs are local research inputs")
    report = run_audit(root=ROOT, contract=contract)
    assert report["automatic_pass"] is True
    assert report["measurements"]["ilford_profile_count"] == 11
    assert report["measurements"]["ilford_distinct_exponent_group_count"] == 7
    assert report["gate_results"]["kodak_outside_interval_rejected"] is True
