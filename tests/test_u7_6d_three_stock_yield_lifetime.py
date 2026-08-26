from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6d_three_stock_yield_lifetime_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6d_three_stock_yield_lifetime.py"
EVIDENCE = ROOT / "docs/evidence/U7_6D_THREE_STOCK_YIELD_LIFETIME_RESULT.json"


def _audit_module():
    spec = importlib.util.spec_from_file_location("u7_6d_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_generator_lifetime_and_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6D"
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 134217728
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.95


def test_evaluator_requires_exact_outputs_and_memory_reduction(tmp_path: Path) -> None:
    module = _audit_module()
    module.SCRATCH = tmp_path / "scratch"
    module.SCRATCH.mkdir()
    stock_ids = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    output_rows = [
        {
            "style_id": style,
            "output_sha256": f"output-{style}",
            "decoded_rgb16_sha256": f"decoded-{style}",
            "normalized_recipe_sha256": f"recipe-{style}",
        }
        for style in ("velvia_50", "portra_400", "ektar_100")
    ]
    rows = [
        {
            "mode": mode,
            "wall_seconds": 10.0 if mode == "baseline" else 9.9,
            "peak_process_tree_rss_bytes": 1_000_000_000
            if mode == "baseline"
            else 850_000_000,
            "rows": output_rows,
            "manifest_stock_ids": stock_ids,
        }
        for mode in ("baseline", "candidate", "candidate", "baseline")
    ]
    gates = json.loads(CONFIG.read_text(encoding="utf-8"))["gates"]
    result = module.evaluate_rows(rows, gates)
    assert result["decision"] == "PASS"
    assert result["peak_rss_reduction_bytes"] == 150_000_000

    rows[2]["rows"] = [{**output_rows[0], "output_sha256": "drift"}, *output_rows[1:]]
    assert module.evaluate_rows(rows, gates)["decision"] == "FAIL_CLOSED"


def test_formal_evidence_closes_unproductive_lifetime_change() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "FAIL_CLOSED_REVERT_YIELD_LIFETIME_CHANGE"
    assert evidence["scientific_identity"] == (
        "b47c3edc29b31ca36e2a5adc817003c28633ad65f51cc76ffd906d9ccd53049b"
    )
    observations = evidence["observations"]
    assert observations["three_encoded_outputs_byte_exact"] is True
    assert observations["peak_rss_reduction_bytes"] < 134217728
    assert observations["peak_rss_ratio"] > 0.95
    assert observations["candidate_residue_count"] == 0
