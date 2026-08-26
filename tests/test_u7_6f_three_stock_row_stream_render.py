from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6f_three_stock_row_stream_render_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6f_three_stock_row_stream_render.py"
EVIDENCE = ROOT / "docs/evidence/U7_6F_THREE_STOCK_ROW_STREAM_RENDER_RESULT.json"


def _module():
    spec = importlib.util.spec_from_file_location("u7_6f_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_row_stream_and_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6F"
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["execution"]["tile_size"] == 512
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 201326592
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.90


def test_evaluator_requires_cross_execution_parity_and_memory(tmp_path: Path) -> None:
    module = _module()
    module.SCRATCH = tmp_path / "scratch"
    module.SCRATCH.mkdir()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    controls = config["controls"]
    output_rows = [
        {
            "style_id": style,
            "output_sha256": f"MODE-{style}",
            "decoded_rgb16_sha256": f"decoded-{style}",
            "icc_fingerprint_sha256": "icc",
            "normalized_recipe_sha256": f"recipe-{style}",
        }
        for style in ("velvia_50", "portra_400", "ektar_100")
    ]
    rows = []
    for mode in ("baseline", "candidate", "candidate", "baseline"):
        current = [
            {**row, "output_sha256": row["output_sha256"].replace("MODE", mode)}
            for row in output_rows
        ]
        rows.append(
            {
                "mode": mode,
                "wall_seconds": 10.0 if mode == "baseline" else 9.0,
                "peak_process_tree_rss_bytes": (
                    2_000_000_000 if mode == "baseline" else 1_750_000_000
                ),
                "rows": current,
                "manifest_stock_ids": [
                    "fujifilm_velvia_50",
                    "kodak_portra_400",
                    "kodak_ektar_100",
                ],
                "render_execution_id": controls[f"{mode}_execution_id"],
            }
        )
    result = module.evaluate_rows(rows, config["gates"], controls)
    assert result["decision"] == "PASS"
    assert result["peak_rss_reduction_bytes"] == 250_000_000

    rows[1]["rows"][0]["decoded_rgb16_sha256"] = "drift"
    assert module.evaluate_rows(rows, config["gates"], controls)["decision"] == "FAIL_CLOSED"


def test_formal_evidence_reverts_nonreducing_row_stream() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "FAIL_CLOSED_REVERT_ROW_STREAM_RENDER"
    assert evidence["scientific_identity"] == (
        "59e9564735b5f550570f045a07b296a6faa42c0d1000167bc824f3efeaea1406"
    )
    observations = evidence["observations"]
    assert observations["three_decoded_sample_arrays_cross_execution_exact"] is True
    assert observations["three_icc_fingerprints_cross_execution_exact"] is True
    assert observations["peak_rss_reduction_bytes"] < 201326592
    assert observations["peak_rss_ratio"] > 0.90
    assert observations["candidate_residue_count"] == 0
