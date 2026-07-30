from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.response_bounded_backing_return import load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3n_response_bounded_backing_return_v1.json"


def test_p3n_contract_is_development_only_and_nonfitting() -> None:
    contract = load_contract(CONTRACT)
    assert contract["node"] == "U6.P3N"
    assert "development only" in contract["epistemic_scope"]
    assert not contract["candidate"]["profile_fitting_allowed"]
    assert not contract["candidate"]["per_image_parameter_selection_allowed"]
    assert not contract["candidate"]["hard_clipping_allowed"]
    assert not contract["training_allowed"]
    assert not contract["production_integration_allowed"]


def test_p3n_contract_preserves_shared_direction_and_parent_failure() -> None:
    contract = load_contract(CONTRACT)
    candidate = contract["candidate"]
    assert candidate["maximum_transmittance_delta"] == 0.009
    assert "identically" in candidate["shared_scale"]
    gates = contract["automatic_gates"]
    assert gates["parent_unbounded_maximum_abs_minimum"] == 0.08
    assert gates["parent_unbounded_isolated_excursion_minimum"] == 1
    assert gates["maximum_isolated_candidate_excursion_count"] == 0
    assert gates["bounded_population_p95_abs_minimum"] > 0.0


def test_p3n_contract_rejects_schema_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
