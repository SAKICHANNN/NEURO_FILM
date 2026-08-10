from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.boolean_gaussian_lod_boundary import (
    BooleanGaussianLodError,
    _feature_distance_summary,
    evaluate_boolean_gaussian_lod_boundary,
    higher_order_features,
    synthesize_exact_spectrum_gaussian,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cg_boolean_gaussian_lod_boundary_v1.json"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_rejects_radius_or_parent_drift(tmp_path: Path) -> None:
    contract = load_contract()
    contract["reference"]["radius_input_pixels"][-1] = 0.71
    with pytest.raises(BooleanGaussianLodError, match="contract drift"):
        validate_contract(contract, ROOT)

    contract = load_contract()
    parent = contract["parents"]["boolean_representation_decision"]
    payload = json.loads((ROOT / parent["path"]).read_text(encoding="utf-8"))
    payload["decision"] = "forged"
    forged = tmp_path / "parent.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")
    parent["path"] = (
        str(forged.relative_to(ROOT)) if forged.is_relative_to(ROOT) else str(forged)
    )
    with pytest.raises((BooleanGaussianLodError, FileNotFoundError)):
        validate_contract(contract, ROOT)


def test_exact_spectrum_gaussian_is_repeatable_and_matches_target() -> None:
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-(xx * xx + yy * yy) / 8.0)
    target = np.abs(np.fft.fft2(kernel)) ** 2
    target[0, 0] = 0.0
    target /= np.mean(target)
    first, first_error = synthesize_exact_spectrum_gaussian(target, seed=1234)
    repeated, repeated_error = synthesize_exact_spectrum_gaussian(target, seed=1234)
    different, _ = synthesize_exact_spectrum_gaussian(target, seed=1235)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert first_error < 1e-12
    assert repeated_error == first_error


def test_higher_order_features_are_finite_and_shape_stable() -> None:
    contract = load_contract()
    field = np.random.default_rng(7).normal(size=(72, 72))
    features = higher_order_features(field, contract)
    assert {key: value.shape for key, value in features.items()} == {
        "marginal_quantiles": (15,),
        "local_rms_quantiles": (5,),
        "excursion_topology": (16,),
    }
    assert all(np.all(np.isfinite(value)) for value in features.values())


def test_zero_spread_topology_remains_a_factual_comparison() -> None:
    base = {
        "marginal_quantiles": np.asarray([0.0, 1.0]),
        "local_rms_quantiles": np.asarray([1.0, 2.0]),
        "excursion_topology": np.asarray([0.0, 0.0]),
    }
    development = [base, {key: value.copy() for key, value in base.items()}]
    boolean = [{key: value.copy() for key, value in base.items()}]
    gaussian = [{key: value.copy() for key, value in base.items()}]
    result = _feature_distance_summary(development, boolean, gaussian)
    topology = result["families"]["excursion_topology"]
    assert topology["development_iqr_rms_scale_was_zero"] is True
    assert topology["distance_scale_used"] == 1.0
    assert topology["gaussian_to_boolean_distance_ratio"] == 0.0


def test_frozen_boolean_gaussian_lod_evaluation_is_exact() -> None:
    first = evaluate_boolean_gaussian_lod_boundary(load_contract(), ROOT)
    second = evaluate_boolean_gaussian_lod_boundary(load_contract(), ROOT)
    assert first == second
    assert first["maximum_gaussian_periodogram_relative_error"] < 1e-12
    assert len(first["radius_results"]) == 3
