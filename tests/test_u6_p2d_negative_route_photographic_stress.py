from __future__ import annotations

import json
from pathlib import Path

from src.eval.negative_route_photographic_stress import SCHEMA, evaluate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2d_negative_route_photographic_stress_v1.json"


def test_contract_excludes_closed_slide_and_unproved_bw() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["routes"] == [
        "color_negative_neutral_scan",
        "color_negative_print",
    ]
    assert "slide_direct_scan" not in contract["routes"]
    assert "bw_developer_scan" not in contract["routes"]


def test_two_row_smoke_is_repeat_exact_and_bounded(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / contract["input"]["manifest"]).read_text(encoding="utf-8")
    )
    selected_ids = set(contract["visual_protocol"]["fixed_ids"])
    # Keep all fixed rows for the contact-sheet invariant but reduce neither
    # their resolution nor the frozen arithmetic.
    selected = [row for row in manifest if row["id"] in selected_ids]
    smoke = json.loads(json.dumps(contract))
    smoke["input"]["expected_rows"] = len(selected)
    smoke["input"]["expected_camera_makes"] = len(
        {row["make"] for row in selected}
    )
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
