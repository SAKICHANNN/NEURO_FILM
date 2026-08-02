from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_aperture_table_source import (
    KodakApertureTableSourceError,
    audit_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4be_kodak_aperture_table_source_v1.json"
SOURCE = (
    ROOT
    / "data/physical_grain/kodak_aperture_table_1959_v1/official_page.html"
)


def test_contract_rejects_subscription_pdf_access(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["acquisition"]["paid_pdf_or_subscription_access_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KodakApertureTableSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not SOURCE.is_file(), reason="P4BE official HTML unavailable")
def test_official_source_audit_repeats_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_sha256"] == (
        "7830415e8679f8f96766000cc1b6d4fba163fd3abfd931f4b18a39486744ffc3"
    )
    assert first["source_bytes"] == 156232
    assert first["source_pass"]
    assert first["table_count"] == 4
    assert first["density_group_count"] == 23
    assert first["numeric_measurement_count"] == 161
    assert first["decision"] == "open_fixed_kodak_aperture_law_confirmation"
    assert all(len(group["measurements"]) == 7 for group in first["density_groups"])
