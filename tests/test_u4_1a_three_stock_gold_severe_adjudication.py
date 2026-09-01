from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.three_stock_gold_severe_adjudication import (
    _exact_review_boxes,
    _gold_rows,
    _review_sheet_bytes,
    _verify_candidate_bindings,
    normalized_recipe_semantic_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u4_1a_three_stock_gold_severe_adjudication_v1.json").read_text(
        "utf-8"
    )
)


def test_contract_binds_exact_gold_inputs_and_candidate_assets() -> None:
    bindings = _verify_candidate_bindings(CONFIG, ROOT)
    rows = _gold_rows(CONFIG, ROOT)
    assert list(bindings) == [
        "configs/render_profiles/safe_rich_product_v1.json",
        "configs/film_color_stats.json",
        "configs/color_guardrails.json",
        "scripts/render_film.py",
        "src/inference/product_desktop.py",
        "src/inference/three_stock_look.py",
    ]
    assert [row["id"] for row in rows] == [
        "01",
        "05",
        "08",
        "09",
        "11",
        "18",
        "21",
        "29",
        "FS_FACE_01",
    ]
    assert all((ROOT / row["source_path"]).is_file() for row in rows)


def test_recipe_semantic_identity_redacts_only_absolute_paths() -> None:
    recipe = {
        "input": {"path": "C:/run-a/source.png", "sha256": "a" * 64},
        "output": {"path": "C:/run-a/output.png", "sha256": "b" * 64},
        "render": {"style": "portra_400", "look_amount": 1.0},
    }
    moved = json.loads(json.dumps(recipe))
    moved["input"]["path"] = "P:/run-b/source.png"
    moved["output"]["path"] = "P:/run-b/output.png"
    assert normalized_recipe_semantic_identity(recipe) == normalized_recipe_semantic_identity(
        moved
    )
    moved["render"]["style"] = "ektar_100"
    assert normalized_recipe_semantic_identity(recipe) != normalized_recipe_semantic_identity(
        moved
    )


def test_four_review_crops_and_sheet_are_deterministic() -> None:
    yy, xx = np.mgrid[:320, :352]
    source = np.stack(
        ((xx * 3 + yy) % 251, (xx + yy * 5) % 251, (xx * 7 + yy * 2) % 251),
        axis=-1,
    ).astype(np.uint8)
    output = np.ascontiguousarray(np.roll(source, 1, axis=2))
    boxes_a = _exact_review_boxes(source, output, 256)
    boxes_b = _exact_review_boxes(source, output, 256)
    assert boxes_a == boxes_b
    assert list(boxes_a) == CONFIG["review"]["exact_crop_strategies"]
    assert all(right - left == bottom - top == 256 for left, top, right, bottom in boxes_a.values())
    rows = [(style, output, boxes_a) for style in CONFIG["candidate"]["style_ids"]]
    assert _review_sheet_bytes("synthetic", source, rows) == _review_sheet_bytes(
        "synthetic", source, rows
    )
