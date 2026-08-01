from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_halation_mechanism_source import (
    HalationMechanismSourceError,
    audit_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3q0_nist_halation_mechanism_source_v1.json"
SOURCE = ROOT / "data/physical_halation/nist_sensitometry_1922_v1/nbsscientificpaper439vol18p1_A2b.pdf"


def test_contract_rejects_modern_stock_parameter_inference(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["source"]["pdf_pages"] = 119
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HalationMechanismSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not SOURCE.is_file(), reason="exact NIST source PDF unavailable")
def test_exact_source_and_review_pages_are_repeatable() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_u6_p3q1_analytical_base_return_geometry"
    assert all(first["checks"].values())
    assert [row["pdf_page_one_based"] for row in first["reviewed_pages"]] == [30, 31]


@pytest.mark.skipif(not SOURCE.is_file(), reason="exact NIST source PDF unavailable")
def test_review_page_drift_fails_before_source_decision(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    relative_paths = [contract["source"]["path"]] + [
        row["path"] for row in contract["review_pages"]
    ]
    for relative in relative_paths:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    page = project / contract["review_pages"][0]["path"]
    page.write_bytes(page.read_bytes() + b"drift")
    with pytest.raises(HalationMechanismSourceError, match="reviewed-page integrity"):
        audit_source(contract, project)
