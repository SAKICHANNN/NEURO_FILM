from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.clustered_nps_canonical_field import (
    ClusteredNPSCanonicalFieldError,
    _validate_contract,
    evaluate_clustered_nps_canonical_field,
)
from src.film_physics.clustered_nps_field import (
    discrete_periodogram,
    normalized_thomas_spectrum_grid,
    synthesize_clustered_nps_field,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bt_clustered_nps_canonical_field_v1.json"


def test_small_field_reproduces_target_and_repeats() -> None:
    kwargs = {
        "shape": (64, 80),
        "particle_sigma_pixels": 1.3,
        "cluster_sigma_pixels": 1.1,
        "mean_offspring": 20.0,
        "seed": 17,
    }
    first = synthesize_clustered_nps_field(**kwargs)
    second = synthesize_clustered_nps_field(**kwargs)
    assert np.array_equal(first.values, second.values)
    assert np.allclose(discrete_periodogram(first.values), first.target_spectrum)
    assert np.mean(first.values) == pytest.approx(0.0, abs=1e-12)
    assert np.mean(np.square(first.values)) == pytest.approx(1.0, abs=1e-12)


def test_target_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        normalized_thomas_spectrum_grid(
            (32, 32),
            particle_sigma_pixels=0.0,
            cluster_sigma_pixels=1.0,
            mean_offspring=1.0,
        )


def test_contract_rejects_shape_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["synthesis"]["field_shape"] = [256, 256]
    with pytest.raises(ClusteredNPSCanonicalFieldError):
        _validate_contract(contract)


def test_formal_canonical_field_is_repeatable_and_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_clustered_nps_canonical_field(contract, ROOT)
    second = evaluate_clustered_nps_canonical_field(contract, ROOT)
    assert first == second
    assert first["row_count"] == 3
    assert first["checks"]["repeat_exact"] is True
    assert set(first["checks"]) == {
        "seed_count",
        "row_count",
        "periodogram",
        "covariance",
        "variance",
        "sample_mean",
        "ifft_imaginary",
        "repeat_exact",
        "seed_distinct_fields",
    }
