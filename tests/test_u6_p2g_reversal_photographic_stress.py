from __future__ import annotations

import json
from pathlib import Path

from src.eval.reversal_photographic_stress import SCHEMA, evaluate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2g_reversal_photographic_stress_v1.json"


def test_contract_is_fixed_and_excludes_fitting() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["pipeline"]["pseudo_exposure_scale"] == 1.0
    assert contract["automatic_gates"]["near_black_threshold"] == 0.02
    assert contract["automatic_gates"]["near_white_threshold"] == 0.98
    assert any("fit reversal" in item for item in contract["forbidden"])


def test_one_image_smoke_is_repeat_exact(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / contract["input"]["manifest"]).read_text(encoding="utf-8")
    )
    selected = [manifest[0]]
    smoke = json.loads(json.dumps(contract))
    smoke["input"]["expected_rows"] = 1
    smoke["input"]["expected_camera_makes"] = 1
    smoke["visual_protocol"]["fixed_ids"] = [selected[0]["id"]]
    first = evaluate(
        smoke,
        selected,
        root=ROOT,
        contact_sheet_path=tmp_path / "a.png",
    )
    second = evaluate(
        smoke,
        selected,
        root=ROOT,
        contact_sheet_path=tmp_path / "b.png",
    )
    assert first == second
    assert first["contact_sheet_sha256"] == second["contact_sheet_sha256"]
    assert first["decisions"]["finite_bounded"] is True
    assert first["decisions"]["repeat"] is True
