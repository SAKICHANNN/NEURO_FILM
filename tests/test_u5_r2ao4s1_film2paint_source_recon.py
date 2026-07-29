from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2ao4s1_film2paint_exact_source_recon_v1.json"
DECISION = (
    ROOT
    / "configs/u5_r2ao4s1_film2paint_exact_source_recon_decision_v1.json"
)
CONTRACT_SHA256 = "7de1310df861585c5b61fcb34a146b83ea4d59fa1b03011d99e1a024fd375ea3"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_identity_and_no_execution_authority() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256
    contract = _load(CONTRACT)
    assert contract["bounded_recon"]["download_dataset_payloads"] is False
    assert contract["operator_fitting_allowed"] is False
    assert contract["training_allowed"] is False
    assert contract["pixel_payload_acquisition_allowed"] is False


def test_decision_closes_publication_only_source() -> None:
    decision = _load(DECISION)
    assert decision["contract"]["sha256"] == CONTRACT_SHA256
    assert decision["availability_audit"]["publication_supplies_patch_table"] is False
    assert decision["availability_audit"]["source_thesis_repository"]["dataset_files"] == 0
    assert decision["decision"]["separate_bounded_data_acquisition_contract_allowed"] is False
    assert decision["decision"]["operator_fitting_allowed"] is False
    assert decision["decision"]["training_allowed"] is False


def test_device_space_and_stock_claim_boundaries_remain_explicit() -> None:
    decision = _load(DECISION)
    limits = decision["scientific_limits"]
    assert limits["digital_to_film_pairs"] is False
    assert limits["film_recorder_device_space_is_scene_linear_rgb"] is False
    assert limits["digital_sg_stock_signal_separable_from_exposure"] is False
    assert decision["decision"]["production_integration_allowed"] is False
