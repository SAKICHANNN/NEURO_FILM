from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u4_5g_preview_png_compression_binding import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5g_preview_png_compression_binding_v1.json"


def test_u4_5g_contract_has_exact_source_and_frozen_levels() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source"]["sha256"] == (
        "7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6"
    )
    assert config["render"]["compression_levels"] == [0, 6, 9]
    assert [row["style_id"] for row in config["historical_default_level_6"]["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]


def test_u4_5g_execution_passes_all_frozen_gates() -> None:
    report = execute(CONFIG)
    assert report["status"] == "PASS_PRIVATE_U4_5G_PREVIEW_PNG_COMPRESSION_BINDING"
    assert all(report["gates"].values())
    assert report["owned_residue_empty"] is True
