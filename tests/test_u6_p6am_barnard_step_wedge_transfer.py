from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.scanner_step_wedge_transfer import (
    ScannerOecfTransferError,
    evaluate,
    load_contract,
    stable_id,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6am_barnard_step_wedge_transfer_v1.json"


def test_barnard_fixed_parameter_diagnostic() -> None:
    report = evaluate(load_contract(CONTRACT, ROOT), ROOT)
    assert report["status"] == "pass-local-equation-reject-p6al-parameter-transfer"
    assert report["barnard_paper_blue_equation"][
        "passes_local_positive_control"
    ]
    assert not report["p6al_exact_profile"]["passes_transfer_gate"]
    assert report["controls"] == {
        "valid_range_codes_strictly_decreasing": True,
        "highest_density_scanner_reversal_observed": True,
        "parameters_fit_on_10b_161": False,
    }


def test_barnard_report_identity_is_exact() -> None:
    config = load_contract(CONTRACT, ROOT)
    first = evaluate(config, ROOT)
    second = evaluate(config, ROOT)
    assert first == second
    assert stable_id(first) == stable_id(second)


def test_contract_rejects_source_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["source"]["tiff_sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScannerOecfTransferError, match="source binding mismatch"):
        load_contract(path, ROOT)


def test_contract_rejects_valid_range_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["measurement"]["paper_stated_valid_indices_zero_based"].append(15)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScannerOecfTransferError, match="measurement drift"):
        load_contract(path, ROOT)
