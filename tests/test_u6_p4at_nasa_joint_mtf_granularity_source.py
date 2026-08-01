from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.eval.nasa_joint_mtf_granularity_source import (
    JointTableSourceError,
    audit_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4at_nasa_joint_mtf_granularity_source_v1.json"
PDF = ROOT / "data/physical_grain/nasa_cr_124393_v1/19730022682.pdf"


def test_contract_is_no_fit_and_no_digitization() -> None:
    contract = load_contract(CONTRACT)
    assert contract["table_gate"]["no_figure_digitization"]
    assert contract["table_gate"]["no_parameter_fit"]
    assert contract["table_gate"]["minimum_machine_readable_mtf_values"] == 240


def test_contract_rejects_value_gate_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["table_gate"]["minimum_machine_readable_mtf_values"] = 0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JointTableSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not PDF.is_file()
    or shutil.which("pdftotext.exe") is None
    or shutil.which("pdfinfo.exe") is None,
    reason="P4AT source or PDF tools unavailable",
)
def test_audit_binds_numeric_tables_and_nuisance_boundaries() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_pass"]
    assert first["stable_evidence_id"] == (
        "656897fd4816a46ee184736dd626aaf618e21d53acb8b2397a25d2db6c90d0c8"
    )
    assert first["decision"] == "open_nuisance_aware_measured_compatibility"
    assert len(first["numeric_density_granularity_rows"]) == 14
    assert len(first["mtf_blocks"]) == 9
    assert first["mtf_numeric_value_count"] == 243
    assert first["printed_mtf_series_count"] == 27
    assert first["prose_mtf_series_count"] == 26
    assert first["printed_prose_series_count_contradiction"]
    assert all(first["factual_statements"].values())
    assert first["mtf_blocks"][0]["table_block"] == "25a"
    assert first["mtf_blocks"][0]["exposure_id"] == "80-3-2"
    assert first["mtf_blocks"][-1]["rows"][-1] == {
        "frequency_cycles_per_mm": 16,
        "mtf_by_edge": {"3": 0.06, "5": 0.11, "7": 0.05},
    }
