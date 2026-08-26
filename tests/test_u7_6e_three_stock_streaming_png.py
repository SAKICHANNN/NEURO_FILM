from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6e_three_stock_streaming_png_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6e_three_stock_streaming_png.py"


def _module():
    spec = importlib.util.spec_from_file_location("u7_6e_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_encoder_and_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6E"
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["execution"]["png_row_count"] == 128
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 134217728
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.92


def test_evaluator_requires_cross_encoder_samples_and_memory(tmp_path: Path) -> None:
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
        current = [{**row, "output_sha256": row["output_sha256"].replace("MODE", mode)} for row in output_rows]
        rows.append(
            {
                "mode": mode,
                "wall_seconds": 10.0 if mode == "baseline" else 9.0,
                "peak_process_tree_rss_bytes": 1_000_000_000 if mode == "baseline" else 850_000_000,
                "rows": current,
                "manifest_stock_ids": [
                    "fujifilm_velvia_50",
                    "kodak_portra_400",
                    "kodak_ektar_100",
                ],
                "png_encoder_id": controls[f"{mode}_encoder_id"],
                "png_row_count": 128,
            }
        )
    result = module.evaluate_rows(rows, config["gates"], controls)
    assert result["decision"] == "PASS"
    assert result["peak_rss_reduction_bytes"] == 150_000_000

    rows[1]["rows"][0]["decoded_rgb16_sha256"] = "drift"
    assert module.evaluate_rows(rows, config["gates"], controls)["decision"] == "FAIL_CLOSED"
