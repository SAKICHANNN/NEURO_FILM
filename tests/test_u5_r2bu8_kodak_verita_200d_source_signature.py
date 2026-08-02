from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.eval.kodak_verita_source_signature import (
    REPORT_SCHEMA,
    VeritaSourceSignatureError,
    audit_source_signature,
    canonical_json,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu8_kodak_verita_200d_source_signature_v1.json"


def test_verita_source_signature_is_exact_and_evidence_bounded(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = audit_source_signature(contract, ROOT, overlay_path=tmp_path / "first.png")
    second = audit_source_signature(
        contract, ROOT, overlay_path=tmp_path / "second.png"
    )

    assert canonical_json(first) == canonical_json(second)
    assert first["schema"] == REPORT_SCHEMA
    assert len(first["verita_samples"]) == 3
    assert all(len(rows) == 3 for rows in first["verita_samples"].values())
    assert len(first["stable_evidence_id"]) == 64
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()
    encoded = canonical_json(first).decode("utf-8").lower()
    assert "output_rgb" not in encoded
    assert "rendered_pixels" not in encoded
    assert "product_operator" not in first
    assert first["signature_pass"] is False
    assert first["material_domains_by_pair"] == {
        "kodak_verita_200d_5206_7206__vs__kodak_vision3_50d_5203_7203": ["granularity"],
        "kodak_verita_200d_5206_7206__vs__kodak_vision3_250d_5207_7207": [
            "mtf",
            "granularity",
        ],
        "kodak_verita_200d_5206_7206__vs__kodak_vision3_500t_5219_7219": [
            "mtf",
            "granularity",
        ],
    }


def test_verita_source_signature_rejects_vector_operation_drift(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_contract(CONFIG))
    contract["vector_source"]["mtf"]["channels"]["blue"]["stroke_operation_index"] = (
        2711
    )
    with pytest.raises(VeritaSourceSignatureError):
        audit_source_signature(contract, ROOT, overlay_path=tmp_path / "bad.png")


def test_verita_source_signature_rejects_parent_hash_drift(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_contract(CONFIG))
    contract["vision3_evidence"]["mtf_trace"]["sha256"] = "0" * 64
    with pytest.raises(VeritaSourceSignatureError):
        audit_source_signature(contract, ROOT, overlay_path=tmp_path / "bad.png")


def test_verita_contract_rejects_gate_relaxation(tmp_path: Path) -> None:
    payload = copy.deepcopy(load_contract(CONFIG))
    payload["gates"]["minimum_material_domains_per_vision3_pair"] = 1
    path = tmp_path / "relaxed.json"
    path.write_bytes(canonical_json(payload))
    with pytest.raises(VeritaSourceSignatureError):
        load_contract(path)
