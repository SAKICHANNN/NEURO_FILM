from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from src.eval.physical_callier_measurement_source import (
    CallierMeasurementSourceError,
    audit_source,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6o_callier_measurement_source_v1.json"
PDF = ROOT / "data/physical_scanner/callier_thesis2013_v1/PhD-GiorgioTrumpy.pdf"


def test_contract_freezes_four_rows_without_fitting() -> None:
    contract = load_contract(CONTRACT)
    assert [row["sample_id"] for row in contract["frozen_measurement_table"]] == [
        "silverHD",
        "silverLD",
        "dyeHD",
        "dyeLD",
    ]
    assert contract["source_gate"]["no_figure_digitization"]
    assert contract["source_gate"]["no_parameter_fit"]


def test_contract_rejects_measurement_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["frozen_measurement_table"][0]["sample_id"] = "other"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CallierMeasurementSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not PDF.is_file()
    or shutil.which("pdftotext.exe") is None
    or shutil.which("pdfinfo.exe") is None,
    reason="P6O source or PDF tools unavailable",
)
def test_real_measurement_source_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_pass"]
    assert first["page_count"] == 145
    assert first["source_sha256"] == (
        "961d100976140d7bfae7da58319961ab2f9cb566a8b128ac02ab13c833be291f"
    )
    assert first["maximum_reported_vs_raw_ratio_difference"] == pytest.approx(0.05)
