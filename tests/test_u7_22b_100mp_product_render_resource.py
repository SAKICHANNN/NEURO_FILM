from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_22b_100mp_product_render_resource_v1.json"
CONTRACT = ROOT / "docs/planning/U7_22B_100MP_PRODUCT_RENDER_RESOURCE_CONTRACT.md"


def test_contract_freezes_exact_100mp_product_route() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["schema"] == "kmcfm.u7-22b-100mp-product-render-resource-contract.v1"
    assert payload["source"]["width"] * payload["source"]["height"] == 100_000_000
    assert payload["source"]["sha256"] == "17fcc4f2073de7490f205184f8fdc5bb85a810e8283b6559a3cbf16f3894b5f9"
    assert payload["candidate"] == {
        "entrypoint": "scripts/render_film.py",
        "product_look": "ektar_100",
        "look_amount": 0.65,
        "output_bit_depth": 8,
        "tile_size": 512,
        "tile_workers": 8,
        "write_recipe": True,
        "effects": {"grain": 0.0, "halation": 0.0, "dust": 0.0},
        "runs_per_report": 2,
    }
    assert payload["gates"]["maximum_worker_wall_seconds"] == 600.0
    assert payload["gates"]["maximum_process_tree_rss_bytes"] == 16 * 1024**3
    assert payload["decision_if_fail"].startswith("FAIL_CLOSED")


def test_contract_forbids_rescue_and_calibrated_claims() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "without parameter" in text
    assert "calibrated Ektar response" in text
    assert "public release" in text
