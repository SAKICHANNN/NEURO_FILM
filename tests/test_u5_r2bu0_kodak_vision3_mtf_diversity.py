from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_vision3_mtf_diversity import (
    MtfDiversityError,
    audit_diversity,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu0_kodak_vision3_mtf_diversity_v1.json"
GRAPH = (
    ROOT
    / "outputs/u6_p2t_kodak_characteristic_diversity/extracted/50d-002.png"
)


def test_contract_rejects_relaxed_shape_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_pair_log_shape_rmse_per_channel"] = 0.01
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MtfDiversityError, match="contract drift"):
        load_contract(path)


def test_contract_rejects_unbounded_trace_path(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["trace"]["path"] = "../foreign.json"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MtfDiversityError, match="repository-relative"):
        load_contract(path)


@pytest.mark.skipif(not GRAPH.is_file(), reason="exact Kodak MTF graphs unavailable")
def test_exact_kodak_mtf_diversity_is_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = audit_diversity(contract, ROOT, overlay_dir=tmp_path / "first")
    second = audit_diversity(contract, ROOT, overlay_dir=tmp_path / "second")
    assert first == second
    assert first["gate_results"]["parent_characteristic_remains_closed"]
    assert first["material_pair_count"] in range(4)
    assert first["decision"] in {
        "retain_source_domain_mtf_signature_and_open_joint_physical_prior",
        "close_exact_vision3_mtf_diversity_without_rescue",
    }
