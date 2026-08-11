from __future__ import annotations

from pathlib import Path

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.global_chroma_procrustes_confirmation import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cb21_contract_binds_disjoint_confirmation_population() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb21_global_chroma_procrustes_confirmation_v1.json"
    )
    population = contract["population"]
    assert contract["experiment_id"] == "U5.R2CB21"
    assert population["source_count_exact"] == 12
    assert population["camera_make_count_exact"] == 12
    assert "disjoint_from_cb20_development" in population["role"]
    assert hash_file(ROOT / population["decision_path"]) == population["decision_sha256"]
    assert hash_file(ROOT / population["manifest_path"]) == population["manifest_sha256"]
    assert contract["blind_protocol"]["seed"] == 20262911
