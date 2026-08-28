from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u4_5d_jpeg_12mp_draft_feasibility import evaluate_geometry
from src.inference.three_stock_preview import preview_dimensions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5d_jpeg_12mp_draft_feasibility_v1.json"


def test_frozen_12mp_target_uses_existing_preview_geometry() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    target = config["target"]
    assert preview_dimensions(4032, 6048, 12_000_000) == (
        target["width"],
        target["height"],
    )
    assert target["width"] * target["height"] == target["pixels"]


def test_geometry_requires_smaller_nonundershooting_native_draft() -> None:
    assert all(
        evaluate_geometry(
            source=(4032, 6048),
            target=(2828, 4242),
            draft=(3000, 4500),
            ceiling=13_500_000,
        ).values()
    )
    full = evaluate_geometry(
        source=(4032, 6048),
        target=(2828, 4242),
        draft=(4032, 6048),
        ceiling=12_000_000,
    )
    assert full["draft_not_below_target"]
    assert full["draft_not_above_source"]
    assert not full["draft_strictly_smaller_than_source"]
    assert not full["draft_pixels_at_most_target_ceiling"]


def test_undershoot_and_upsample_is_not_eligible() -> None:
    gates = evaluate_geometry(
        source=(4032, 6048),
        target=(2828, 4242),
        draft=(2016, 3024),
        ceiling=12_000_000,
    )
    assert not gates["draft_not_below_target"]
    assert gates["draft_strictly_smaller_than_source"]
    assert gates["draft_pixels_at_most_target_ceiling"]
