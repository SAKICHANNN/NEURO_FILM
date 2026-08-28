from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p313_canon_sraw_working_image_compatibility import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p313_canon_sraw_working_image_compatibility_v1.json"


def test_p313_contract_freezes_six_unique_cc0_models() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = config["rows"]
    assert len(rows) == 6
    assert len({row["model"] for row in rows}) == 6
    assert len({row["sha256"] for row in rows}) == 6
    assert config["decode"] == {
        "gamma": [1, 1],
        "no_auto_bright": True,
        "output_bps": 16,
        "output_color": "sRGB",
        "use_camera_wb": True,
        "user_flip": None,
    }
    claim = config["claim_ceiling"].casefold()
    assert "no vendor-exact colour" in claim
    assert "candidate 3" in claim


def test_p313_real_sraw_cohort_passes_frozen_working_image_gates() -> None:
    report = execute(CONFIG)
    assert report["decision"] == "PASS_PRIVATE_CANON_SRAW_WORKING_IMAGE_COMPATIBILITY"
    assert all(report["gates"].values())
    assert len(report["records"]) == 6
