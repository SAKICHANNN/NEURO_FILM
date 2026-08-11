from __future__ import annotations

from pathlib import Path

from src.eval.feasibility_bounded_transport_confirmation import load_contract
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file

ROOT = Path(__file__).resolve().parents[1]


def test_cb24_contract_binds_disjoint_confirmation_and_exact_operator() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb24_feasibility_bounded_transport_confirmation_v1.json"
    )
    population = contract["population"]
    operator = contract["operator"]
    assert contract["experiment_id"] == "U5.R2CB24"
    assert population["source_count_exact"] == 12
    assert population["camera_make_count_exact"] == 12
    assert "disjoint_from_cb23_development" in population["role"]
    assert (
        hash_file(ROOT / population["decision_path"]) == population["decision_sha256"]
    )
    assert (
        hash_file(ROOT / population["manifest_path"]) == population["manifest_sha256"]
    )
    assert (
        hash_file(ROOT / operator["implementation_path"])
        == operator["implementation_sha256"]
    )
    assert (
        hash_file(ROOT / operator["shared_evaluator_path"])
        == operator["shared_evaluator_sha256"]
    )
    assert contract["blind_protocol"]["seed"] == 20263211
