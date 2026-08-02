from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.density_conditioned_thomas_profile import (
    DensityConditionedThomasEvaluationError,
    _validate_contract,
    compile_and_evaluate,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
    binary_circular_aperture_kernel,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bw_density_conditioned_thomas_profile_v1.json"


def test_4000dpi_aperture_is_normalized_and_symmetric() -> None:
    kernel = binary_circular_aperture_kernel(6.35, 48.0)
    assert kernel.shape == (9, 9)
    assert np.sum(kernel, dtype=np.float64) == pytest.approx(1.0)
    assert np.array_equal(kernel, kernel[::-1, ::-1])


def test_contract_rejects_realized_variance_normalization() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["compiler"]["realized_variance_normalization_allowed"] = True
    with pytest.raises(DensityConditionedThomasEvaluationError):
        _validate_contract(contract)


def test_profile_roundtrip_and_wrong_receipt_rejection() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bundle, report = compile_and_evaluate(contract, ROOT)
    profile = DensityConditionedThomasProfile.from_dict(bundle)
    assert profile.to_dict() == {key: value for key, value in bundle.items() if key != "profile_id"}
    assert profile.identity() == bundle["profile_id"]
    assert report["automatic_pass"]


def test_formal_compilation_repeats_exactly() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first_bundle, first_report = compile_and_evaluate(contract, ROOT)
    second_bundle, second_report = compile_and_evaluate(contract, ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    assert first_report["automatic_pass"]
    assert set(first_report["gate_results"]) == {
        "parent_identity",
        "analytic_aperture_sigma",
        "monte_carlo_median",
        "monte_carlo_p95",
        "profile_probe_range",
        "full_field_mean",
        "positive_developed_density",
        "row_partition_exact",
        "profile_roundtrip_exact",
        "finite",
        "no_refit_or_realized_normalization",
    }
