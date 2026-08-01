from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_mtf_source import MtfSourceError, audit_source, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p5i_kodak_250d_mtf_source_v1.json"
GRAPH = ROOT / "outputs/u5_r2aa0_source_audit/extracted/250d_mtf.png"


def test_contract_rejects_relaxed_trace_distance(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["trace_max_ink_distance_px"] = 20.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MtfSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak MTF graph is unavailable")
def test_exact_kodak_mtf_source_is_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT, overlay_path=tmp_path / "first.png")
    second = audit_source(contract, ROOT, overlay_path=tmp_path / "second.png")
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_u6_p5j_positive_psf_compiler"
    assert first["source_sha256"] == contract["source"]["sha256"]
    assert first["graph_sha256"] == contract["graph"]["sha256"]
    assert first["overlay_sha256"]
    assert all(first["gate_results"].values())
    assert first["channels"]["blue"]["responses_percent"][0] > 90.0
    assert first["channels"]["red"]["responses_percent"][-1] < 25.0


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak MTF graph is unavailable")
def test_trace_point_off_source_ink_fails(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    trace_path = ROOT / contract["trace"]
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    payload["curves"]["red"][0] = [392, 120]
    # Exercise the fact through a temporary project root so the contract itself
    # remains exact while the traced source pixels are adversarially changed.
    project = tmp_path / "project"
    project.mkdir()
    (project / "data/physics/kodak_vision3_250d").mkdir(parents=True)
    (project / "outputs/u5_r2aa0_source_audit/extracted").mkdir(parents=True)
    (project / "configs/data").mkdir(parents=True)
    source = ROOT / contract["source"]["path"]
    graph = ROOT / contract["graph"]["path"]
    (project / contract["source"]["path"]).write_bytes(source.read_bytes())
    (project / contract["graph"]["path"]).write_bytes(graph.read_bytes())
    (project / contract["trace"]).write_text(json.dumps(payload), encoding="utf-8")
    report = audit_source(contract, project, overlay_path=tmp_path / "forged.png")
    assert not report["source_pass"]
    assert not report["gate_results"]["source_ink_proximity"]
