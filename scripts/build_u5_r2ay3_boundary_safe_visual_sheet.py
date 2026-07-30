#!/usr/bin/env python
"""Build the frozen AY3 worst-case visual sheet after automatic passage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.boundary_safe_neutral_base import (  # noqa: E402
    apply_boundary_safe_neutral_base,
    validate_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (  # noqa: E402
    build_fixed_ao6_renderer,
)
from src.eval.fivek_neutral_base_parameter_pilot import (  # noqa: E402
    apply_neutral_base,
)
from src.eval.fivek_unseen_content_confirmation import (  # noqa: E402
    load_srgb16,
)


def _select(rows: list[dict], count: int) -> list[dict]:
    worst = sorted(
        rows,
        key=lambda row: (
            -float(row["ridge"]["look_rmse"]),
            row["pair_id"],
        ),
    )
    limited = sorted(
        rows,
        key=lambda row: (
            -float(row["ridge"]["limited_fraction"]),
            row["pair_id"],
        ),
    )
    selected: list[dict] = []
    seen: set[str] = set()
    for row in [*worst[:6], *limited[:6], *worst]:
        if row["pair_id"] not in seen:
            selected.append(row)
            seen.add(row["pair_id"])
        if len(selected) == count:
            break
    return selected


def _tile(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return ImageOps.contain(Image.fromarray(pixels, mode="RGB"), size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay3_boundary_safe_neutral_base_development_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("automatic_pass") is not True:
        raise SystemExit("visual review is forbidden before automatic pass")
    validated = validate_contract(ROOT, config)
    renderer = build_fixed_ao6_renderer(
        validated["fixed_config"], validated["fixed_validated"]
    )
    evidence = {
        row["pair_id"]: row for row in validated["source_manifest"]["rows"]
    }
    original = {
        row["pair_id"]: row for row in validated["ay2_report"]["rows"]
    }
    epsilon = float(config["neutral_operator"]["boundary_epsilon"])
    maximum_side = int(
        validated["ay2_config"]["confirmation"]["maximum_side"]
    )
    selected = _select(report["rows"], int(args.rows))
    columns = [
        "source",
        "neutral_target",
        "unsafe_ridge",
        "safe_ridge",
        "direct_AO6",
        "safe_ridge_AO6",
        "target_AO6",
    ]
    tile_size = (250, 170)
    header = 28
    sheet = Image.new(
        "RGB",
        (
            tile_size[0] * len(columns),
            (tile_size[1] + header) * len(selected),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    selection = []
    for row_index, result in enumerate(selected):
        pair_id = str(result["pair_id"])
        paths = evidence[pair_id]
        source = load_srgb16(ROOT / paths["source_path"], maximum_side)
        target = load_srgb16(ROOT / paths["target_path"], maximum_side)
        parameters = np.asarray(original[pair_id]["ridge"]["parameters"])
        unsafe = apply_neutral_base(source, parameters)
        safe, scale = apply_boundary_safe_neutral_base(
            source, parameters, boundary_epsilon=epsilon
        )
        direct_look, _ = renderer(source)
        safe_look, _ = renderer(safe)
        target_look, _ = renderer(target)
        arrays = [
            source,
            target,
            unsafe,
            safe,
            direct_look,
            safe_look,
            target_look,
        ]
        for column_index, (column, array) in enumerate(
            zip(columns, arrays, strict=True)
        ):
            tile = _tile(array, tile_size)
            left = column_index * tile_size[0]
            top = row_index * (tile_size[1] + header)
            sheet.paste(
                tile,
                (
                    left + (tile_size[0] - tile.width) // 2,
                    top + header + (tile_size[1] - tile.height) // 2,
                ),
            )
            draw.text(
                (left + 4, top + 4),
                f"{column} | {pair_id}" if column_index == 0 else column,
                fill="black",
            )
        selection.append(
            {
                "pair_id": pair_id,
                "look_rmse": result["ridge"]["look_rmse"],
                "limited_fraction": result["ridge"]["limited_fraction"],
                "minimum_scale": float(np.min(scale)),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "visual_stress_sheet.png"
    sheet.save(sheet_path, format="PNG", compress_level=6)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
