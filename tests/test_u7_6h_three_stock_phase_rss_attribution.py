from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6h_three_stock_phase_rss_attribution_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6h_three_stock_phase_rss_attribution.py"


def _module():
    spec = importlib.util.spec_from_file_location("u7_6h_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_attribution_without_product_change() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6H"
    assert payload["execution"]["sample_interval_seconds"] == 0.005
    assert payload["execution"]["fresh_process_count"] == 2
    assert payload["gates"]["dominant_phase_repeat_exact"] is True
    assert payload["gates"]["peak_process_tree_rss_repeat_ratio_max"] == 1.10


def test_evaluator_requires_repeat_exact_dominant_phase(tmp_path: Path) -> None:
    module = _module()
    module.SCRATCH = tmp_path / "scratch"
    module.SCRATCH.mkdir()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    outputs = [
        {
            "style_id": style,
            "output_sha256": f"output-{style}",
            "decoded_rgb16_sha256": f"decoded-{style}",
            "icc_fingerprint_sha256": "icc",
            "normalized_recipe_sha256": f"recipe-{style}",
        }
        for style in module.STYLES
    ]
    rows = [
        {
            "peak_process_tree_rss_bytes": peak,
            "dominant_phase": "render_portra_400",
            "rows": [dict(row) for row in outputs],
            "manifest_stock_ids": list(module.STOCK_IDS),
        }
        for peak in (1_900_000_000, 1_920_000_000)
    ]
    assert module.evaluate_rows(rows, config["gates"])["decision"] == "PASS"
    rows[1]["dominant_phase"] = "encode_png16"
    assert module.evaluate_rows(rows, config["gates"])["decision"] == "FAIL_CLOSED"
