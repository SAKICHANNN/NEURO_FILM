from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.real_film.ntire_paired_shaped_lut_candidate import (
    NTIREPairedCandidateError,
    _aggregate,
    _fit_affine,
    _load_geometry,
    _sample_indices,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a0y_ntire_paired_shaped_lut_candidate2_v1.json"


def test_contract_loads_and_roles_are_disjoint() -> None:
    contract = load_contract(CONFIG)
    roles = contract["roles"]
    assert (
        len(set(roles["fit"] + roles["calibration"] + roles["sealed_confirmation"]))
        == 56
    )


def test_parent_geometry_evidence_binding_loads() -> None:
    geometry = _load_geometry(load_contract(CONFIG), ROOT)
    assert geometry["official_baseline"]["commit"] == (
        "3478fbb39449f5cba29483d4cbfe019013f8261f"
    )


def test_contract_rejects_parameter_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["controls"]["bounded_logit_affine"]["ridge_alpha"] = 0.02
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NTIREPairedCandidateError):
        load_contract(path)


def test_sample_indexes_are_stable_and_unique() -> None:
    first = _sample_indices(92, 32, 1024)
    second = _sample_indices(92, 32, 1024)
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 32


def test_bounded_affine_is_finite_and_positive() -> None:
    rng = np.random.default_rng(4)
    source = rng.uniform(0.01, 0.99, (1024, 3))
    target = np.clip(
        source
        @ np.asarray([[1.03, 0.01, 0.0], [0.0, 0.98, 0.01], [0.01, 0.0, 1.02]]).T,
        0.001,
        0.999,
    )
    operator = _fit_affine(
        source, target, load_contract(CONFIG)["controls"]["bounded_logit_affine"]
    )
    assert np.isfinite(operator.matrix).all()
    assert np.linalg.det(operator.matrix) >= 0.01


def test_aggregate_fails_tail_even_with_median_gain() -> None:
    rows = []
    for reduction in (0.5, 0.4, -0.2):
        rows.append(
            {
                "relative_reduction_vs_strongest_legitimate": reduction,
                "relative_reduction_vs_cyclic_wrong_target": 0.5,
                "mean_oklab_error": {"candidate": 0.05},
                "gradient_p999_ratio": 1.0,
                "new_exact_boundary_fraction": 0.0,
            }
        )
    gates = load_contract(CONFIG)["gates"]["calibration"]
    _, checks = _aggregate(rows, gates)
    assert checks["median_reduction_vs_strongest_legitimate"]
    assert not checks["worst_tail_vs_strongest_legitimate"]


def test_cache_override_rejects_nonproject_d_path(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    with pytest.raises(NTIREPairedCandidateError, match="project-owned D fallback"):
        evaluate(
            contract,
            ROOT,
            model_lock_path=tmp_path / "lock.json",
            cache_root_override=tmp_path,
            range_reader=lambda *_: b"",
        )
