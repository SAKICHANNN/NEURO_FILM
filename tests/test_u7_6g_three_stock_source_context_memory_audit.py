from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6g_three_stock_source_context_memory_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6g_three_stock_source_context_memory.py"
EVIDENCE = ROOT / "docs/evidence/U7_6G_THREE_STOCK_SOURCE_CONTEXT_MEMORY_RESULT.json"


def _module():
    spec = importlib.util.spec_from_file_location("u7_6g_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_context_memory_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6G"
    assert payload["execution"]["lab_row_chunk"] == 128
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 201326592
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.90


def test_evaluator_requires_context_output_and_resource_parity() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_rows = [
        {
            "style_id": style,
            "output_sha256": f"output-{style}",
            "decoded_rgb16_sha256": f"decoded-{style}",
            "icc_fingerprint_sha256": "icc",
            "normalized_recipe_sha256": f"recipe-{style}",
        }
        for style in ("velvia_50", "portra_400", "ektar_100")
    ]
    rows = []
    for mode in ("baseline", "candidate", "candidate", "baseline"):
        rows.append(
            {
                "mode": mode,
                "wall_seconds": 10.0 if mode == "baseline" else 10.5,
                "peak_process_tree_rss_bytes": (
                    2_000_000_000 if mode == "baseline" else 1_750_000_000
                ),
                "source_context": {"lab_mean": [1.0, 2.0, 3.0]},
                "context_residue_count": 0,
                "rows": [dict(row) for row in output_rows],
                "manifest_stock_ids": list(module.STOCK_IDS),
            }
        )
    result = module.evaluate_rows(rows, config["gates"])
    assert result["decision"] == "PASS"
    assert result["peak_rss_reduction_bytes"] == 250_000_000

    rows[1]["source_context"]["lab_mean"][0] = 4.0
    assert module.evaluate_rows(rows, config["gates"])["decision"] == "FAIL_CLOSED"


def test_formal_evidence_closes_nonreducing_context_candidate() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_DO_NOT_INTEGRATE_SCRATCH_BACKED_CONTEXT"
    )
    assert evidence["scientific_identity"] == (
        "374ae9c3f8ae0c2b879bcde81635e51d333e0c8b0ea7bf4e6328fc8ed2ac8c74"
    )
    observations = evidence["observations"]
    assert observations["source_context_cross_execution_exact"] is True
    assert observations["three_encoded_outputs_cross_execution_exact"] is True
    assert observations["peak_rss_reduction_bytes"] < 201326592
    assert observations["peak_rss_ratio"] > 0.90
    assert observations["candidate_residue_count"] == 0
