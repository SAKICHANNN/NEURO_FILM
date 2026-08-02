from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fuji_noise_spectrum_source import (
    FujiNoiseSpectrumSourceError,
    evaluate_fuji_noise_spectrum_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bk_fuji_noise_spectrum_source_v1.json"


def test_p4bk_contract_is_frozen_before_audit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bk_fuji_noise_spectrum_source_contract.v1"
    )
    assert payload["source"]["pdf_sha256"] == (
        "f03541cacdd0deacc0475f8a5ea3aa8bf2761e87d8183037e27a51283f4d92bf"
    )
    assert payload["measurement"]["scanning_aperture_diameter_micrometres"] == 1.0
    assert len(payload["table_1"]) == 5
    assert sum(len(row["k"]) for row in payload["table_1"]) == 11
    assert payload["audit"]["require_no_graph_digitization"] is True


def test_p4bk_contract_rejects_coefficient_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["table_1"][0]["k"][0] = 141.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FujiNoiseSpectrumSourceError, match="contract drift"):
        load_contract(path)


def test_p4bk_source_audit_passes() -> None:
    report = evaluate_fuji_noise_spectrum_source(load_contract(CONTRACT), ROOT)
    assert report["automatic_pass"] is True
    assert report["decision"] == "open_historical_bw_measured_nps_compiler"
    assert report["table_row_count"] == 5
    assert report["gaussian_component_count"] == 11
    assert report["material_process_context_count"] == 2
    assert all(report["gate_results"].values())


def test_p4bk_source_audit_rejects_wrong_pdf(tmp_path: Path) -> None:
    payload = load_contract(CONTRACT)
    source = ROOT / payload["source"]["pdf_path"]
    wrong = tmp_path / "wrong.pdf"
    wrong.write_bytes(source.read_bytes()[:-1])
    payload = json.loads(json.dumps(payload))
    payload["source"]["pdf_path"] = str(wrong.relative_to(tmp_path))
    with pytest.raises(FujiNoiseSpectrumSourceError, match="identity mismatch"):
        evaluate_fuji_noise_spectrum_source(payload, tmp_path)
