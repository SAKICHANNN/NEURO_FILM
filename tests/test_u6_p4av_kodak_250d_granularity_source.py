from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_granularity_source import (
    GranularitySourceError,
    audit_source,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4av_kodak_250d_granularity_source_v1.json"
GRAPH = ROOT / "outputs/u5_r2aa0_source_audit/extracted/250d_characteristic.png"


def test_contract_rejects_relaxed_trace_distance(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["trace_max_ink_distance_px"] = 20.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GranularitySourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak graph is unavailable")
def test_exact_granularity_source_is_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT, overlay_path=tmp_path / "first.png")
    second = audit_source(contract, ROOT, overlay_path=tmp_path / "second.png")
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_joint_mtf_granularity_compatibility"
    assert all(first["gate_results"].values())
    assert first["channels"]["blue"]["sigma_d_max"] > first["channels"]["red"]["sigma_d_max"]


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak graph is unavailable")
def test_trace_hash_tamper_fails(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    for relative in (
        contract["source"]["path"],
        contract["graph"]["path"],
        contract["trace"],
    ):
        source = ROOT / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    trace_path = project / contract["trace"]
    trace_path.write_bytes(trace_path.read_bytes() + b" ")
    with pytest.raises(GranularitySourceError, match="trace integrity"):
        audit_source(contract, project, overlay_path=tmp_path / "unused.png")
