from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from src.eval.physical_density_grain_source import (
    DensityGrainSourceError,
    audit_source,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4aa_nasa_density_grain_source_v1.json"
PDF = ROOT / "data/physical_grain/nasa_cr_124393_v1/19730022682.pdf"


def test_contract_forbids_figure_digitization_and_fit() -> None:
    contract = load_contract(CONTRACT)
    assert contract["source_gate"]["no_figure_digitization"]
    assert contract["source_gate"]["no_parameter_fit"]
    assert contract["source_gate"][
        "minimum_machine_extractable_numeric_density_granularity_rows"
    ] == 3


def test_contract_rejects_row_gate_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["source_gate"][
        "minimum_machine_extractable_numeric_density_granularity_rows"
    ] = 0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DensityGrainSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not PDF.is_file()
    or shutil.which("pdftotext.exe") is None
    or shutil.which("pdfinfo.exe") is None,
    reason="P4AA source or PDF tools unavailable",
)
def test_report_exposes_machine_extractable_granularity_tables() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_sha256"] == (
        "8773bb57946c38573d2feb42a08ddf3b0d5001745bc46ac23d825d6cdad90814"
    )
    assert first["page_count"] == 85
    assert first["source_pass"]
    assert first["numeric_density_granularity_row_count"] == 14
    assert {row["table"] for row in first["numeric_density_granularity_rows"]} == {
        4,
        5,
        6,
    }
    assert first["decision"] == "open_fixed_density_granularity_compatibility"
