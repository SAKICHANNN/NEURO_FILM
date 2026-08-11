from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import compound_thomas_archival_confirmation as p4cl

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/u6_p4cl_compound_thomas_archival_confirmation_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_and_parent_are_exact() -> None:
    p4cl._validate_contract(ROOT, _contract())


def test_base_and_scale_fields_are_exact_and_finite() -> None:
    first = p4cl._base_and_scale_fields(_contract(), 3, seed_offset=9)
    second = p4cl._base_and_scale_fields(_contract(), 3, seed_offset=9)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
        assert np.isfinite(left).all()
        assert np.allclose(np.mean(left, axis=(1, 2)), 0.0, atol=1e-6)
        assert np.allclose(np.sqrt(np.mean(np.square(left), axis=(1, 2))), 1.0)


def test_compound_field_is_bounded_finite_and_non_gaussian() -> None:
    contract = _contract()
    bases, scales = p4cl._base_and_scale_fields(contract, 4, seed_offset=21)
    zero = p4cl._compound_fields(bases, scales, 0.0, [0.25, 4.0])
    candidate = p4cl._compound_fields(bases, scales, 0.7, [0.25, 4.0])
    assert np.allclose(zero, bases)
    assert np.isfinite(candidate).all()
    assert not np.array_equal(candidate, bases)
    assert np.median(p4cl._feature_matrix(candidate)[:, 2]) > np.median(
        p4cl._feature_matrix(bases)[:, 2]
    )


def test_feature_scoring_prefers_exact_model_median() -> None:
    rng = np.random.default_rng(5)
    fields = np.asarray(
        [p4cl._normalized(rng.standard_normal((96, 96))) for _ in range(8)]
    )
    features = p4cl._feature_matrix(fields)
    median = np.median(features, axis=0)
    rows = p4cl._score_sources(
        {"source": list(fields)}, median, np.ones(4), median
    )
    assert rows[0]["gaussian_feature_distance"] == 0.0
    assert rows[0]["compound_feature_distance"] == 0.0
