from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_characteristic_source import (
    CharacteristicSourceError,
    audit_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2p_kodak_250d_characteristic_source_v1.json"
GRAPH = ROOT / "outputs/u5_r2aa0_source_audit/extracted/250d_characteristic.png"


def test_contract_rejects_relaxed_monotonic_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["maximum_monotonic_reversal_density"] = 0.1
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak characteristic graph unavailable")
def test_exact_kodak_characteristic_source_is_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT, overlay_path=tmp_path / "first.png")
    second = audit_source(contract, ROOT, overlay_path=tmp_path / "second.png")
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_u6_p2q_characteristic_prior_compiler"
    assert first["source_sha256"] == contract["source"]["sha256"]
    assert first["graph_sha256"] == contract["graph"]["sha256"]
    assert all(first["gate_results"].values())
    assert first["channels"]["blue"]["status_m_density"][-1] > 2.9
    assert first["channels"]["red"]["status_m_density"][0] < 0.25


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak characteristic graph unavailable")
def test_trace_hash_drift_fails_before_evaluation(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    for relative in (
        contract["source"]["path"],
        contract["graph"]["path"],
        contract["trace"]["path"],
        "configs/data/kodak_250d_2383_curve_pixels_v1.json",
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    trace_path = project / contract["trace"]["path"]
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    payload["curves"]["red"][0] = [65, 300]
    trace_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicSourceError, match="trace integrity"):
        audit_source(contract, project, overlay_path=tmp_path / "forged.png")
