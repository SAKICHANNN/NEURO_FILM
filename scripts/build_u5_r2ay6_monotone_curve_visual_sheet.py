#!/usr/bin/env python
"""Build the permitted AY6 fresh worst-tail visual review sheet."""

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
    apply_boundary_safe_residual,
)
from src.eval.fivek_monotone_channel_curve_development import (  # noqa: E402
    apply_curve_operator,
    validate_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (  # noqa: E402
    build_fixed_ao6_renderer,
)
from src.eval.fivek_unseen_content_confirmation import (  # noqa: E402
    load_srgb16,
)


def _tile(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return ImageOps.contain(Image.fromarray(pixels, mode="RGB"), size)


def _select(rows: list[dict], count: int) -> list[dict]:
    worst = sorted(
        rows,
        key=lambda row: (
            -float(row["ridge"]["neutral_rmse"]),
            row["pair_id"],
        ),
    )
    harmed = sorted(
        rows,
        key=lambda row: (
            float(row["global"]["neutral_rmse"])
            - float(row["ridge"]["neutral_rmse"]),
            row["pair_id"],
        ),
    )
    selected = []
    seen = set()
    for row in [*worst[:8], *harmed[:8], *worst]:
        if row["pair_id"] not in seen:
            selected.append(row)
            seen.add(row["pair_id"])
        if len(selected) == count:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay6_monotone_channel_curve_development_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional exact normalized manifest for a later confirmation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("automatic_pass") is not True:
        raise SystemExit("visual review is forbidden before automatic pass")
    validated = validate_contract(ROOT, config)
    renderer = build_fixed_ao6_renderer(
        validated["safe_validated"]["fixed_config"],
        validated["safe_validated"]["fixed_validated"],
    )
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest is not None
        else validated["fresh_manifest"]
    )
    evidence = {row["pair_id"]: row for row in manifest["rows"]}
    rows = (
        report["populations"]["fresh_63"]["rows"]
        if "populations" in report
        else report["population"]["rows"]
    )
    selected = _select(rows, int(args.rows))
    knots = np.asarray(config["operator"]["knot_inputs"])
    epsilon = float(config["parents"]["ay3"]["boundary_epsilon"])
    maximum_side = int(validated["ay0_config"]["decode"]["maximum_side"])
    columns = [
        "source",
        "global_curve",
        "ridge_curve",
        "neutral_target",
        "global+AO6",
        "ridge+AO6",
        "target+AO6",
    ]
    tile_size = (240, 164)
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
    for row_index, row in enumerate(selected):
        pair_id = str(row["pair_id"])
        item = evidence[pair_id]
        source = load_srgb16(
            Path(item["source_path"]), maximum_side
        )
        target = load_srgb16(
            Path(item["target_path"]), maximum_side
        )
        neutral = {}
        looks = {}
        for method in ("global", "ridge"):
            candidate = apply_curve_operator(
                source,
                np.asarray(row[method]["parameters"]),
                knots=knots,
            )
            neutral[method], _ = apply_boundary_safe_residual(
                source, candidate, boundary_epsilon=epsilon
            )
            looks[method], _ = renderer(neutral[method])
        target_look, _ = renderer(target)
        arrays = [
            source,
            neutral["global"],
            neutral["ridge"],
            target,
            looks["global"],
            looks["ridge"],
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
                "global_neutral_rmse": row["global"]["neutral_rmse"],
                "ridge_neutral_rmse": row["ridge"]["neutral_rmse"],
                "ridge_limited_fraction": row["ridge"][
                    "limited_fraction"
                ],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "fresh_visual_stress_sheet.png"
    sheet.save(sheet_path, format="PNG", compress_level=6)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
