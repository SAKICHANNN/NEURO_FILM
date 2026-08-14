from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.prophoto_icc_semantic_ingress import (
    ProPhotoSemanticIngressError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c9_embedded_prophoto_icc_semantic_ingress_v1.json"


def test_contract_is_frozen_and_scoped() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U1.4C9"
    assert contract["source"]["expected_rows"] == 12
    assert contract["gates"]["maximum_product_vs_littlecms_absolute_error"] == 2e-6
    assert contract["gates"]["require_exact_two_process_replay"] is True
    assert "not gamut mapping" in contract["claim_ceiling"]


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(CONTRACT.read_bytes() + b" ")
    with pytest.raises(ProPhotoSemanticIngressError, match="contract hash drift"):
        load_contract(changed)
