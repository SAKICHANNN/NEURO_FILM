from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.native_gamma_density import apply_native_gamma_density

ROOT = Path(__file__).resolve().parents[1]


def test_p4ho_contract_binds_p4hn_and_profile() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4ho_native_gamma_density_inverse_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["parents"]["p4hn_evidence"]["required_decision"] == (
        "retain_source_observable_calibrated_native_transport_for_gamma_density_stage"
    )
    assert contract["candidate"]["scipy_gamma_ppf_is_oracle_only"] is True
    assert min(contract["candidate"]["shape_values"]) < 1.0
    assert max(contract["candidate"]["shape_values"]) >= 100000.0


def test_p4ho_wrapper_rejects_shape_mismatch_before_native_call() -> None:
    with pytest.raises(ValueError, match="invalid native gamma"):
        apply_native_gamma_density(
            object(),  # type: ignore[arg-type]
            np.array([0.5]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            inverse_iterations=80,
        )
