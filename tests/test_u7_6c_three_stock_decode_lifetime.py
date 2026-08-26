from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6c_three_stock_decode_lifetime_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6c_three_stock_decode_lifetime.py"


def _module():
    spec = importlib.util.spec_from_file_location("u7_6c_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_lifecycle_and_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6C"
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 134217728
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.95
    assert "non-calibrated Look Approximations" in payload["claim_ceiling"]


def test_evaluator_requires_exact_outputs_and_memory_reduction(tmp_path: Path) -> None:
    module = _module()
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
            "wall_seconds": 10.0 if mode == "baseline" else 9.8,
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

    rows[1]["rows"] = [{**output_rows[0], "output_sha256": "drift"}, *output_rows[1:]]
    assert module.evaluate_rows(rows, gates)["decision"] == "FAIL_CLOSED"
