#!/usr/bin/env python
"""Build a worst-case-heavy visual sheet after the frozen AY1 gates pass."""

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

from scripts.build_fivek_freeze_pack import (  # noqa: E402
    filtered_target,
    load_expert_icc_srgb,
    load_raw_default,
    resize_to_shape,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (  # noqa: E402
    build_fixed_ao6_renderer,
    validate_contract,
)
from src.eval.fivek_neutral_base_parameter_pilot import (  # noqa: E402
    apply_neutral_base,
)


def _select_rows(rows: list[dict], count: int) -> list[dict]:
    by_error = sorted(
        rows,
        key=lambda row: (
            -float(row["ridge"]["rmse_to_fixed_look_target"]),
            row["pair_id"],
        ),
    )
    by_gain = sorted(
        rows,
        key=lambda row: (
            float(row["global"]["rmse_to_fixed_look_target"])
            - float(row["ridge"]["rmse_to_fixed_look_target"]),
            row["pair_id"],
        ),
    )
    candidates = [
        *by_error[:4],
        *by_gain[:4],
        *list(reversed(by_gain[-2:])),
    ]
    selected: list[dict] = []
    seen: set[str] = set()
    for row in [*candidates, *by_error]:
        pair_id = str(row["pair_id"])
        if pair_id not in seen:
            selected.append(row)
            seen.add(pair_id)
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
        / "configs/u5_r2ay1_fivek_neutral_base_fixed_ao6_ablation_v1.json",
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
    renderer = build_fixed_ao6_renderer(config, validated)
    selected = _select_rows(report["rows"], int(args.rows))
    source_rows = {
        row["pair_id"]: row
        for row in json.loads(
            (
                ROOT
                / validated["neutral_config"]["source_evidence"]["manifest"]
            ).read_text(encoding="utf-8")
        )["rows"]
    }
    parameter_rows = {
        row["pair_id"]: row
        for row in validated["neutral_report"]["rows"]
    }
    neutral_config = validated["neutral_config"]
    target_policy = type(
        "TargetPolicy",
        (),
        {
            "luma_strength": neutral_config["neutral_target"][
                "luma_strength"
            ],
            "chroma_strength": neutral_config["neutral_target"][
                "chroma_strength"
            ],
            "chroma_headroom": neutral_config["neutral_target"][
                "chroma_headroom"
            ],
            "wb_anchor_strength": neutral_config["neutral_target"][
                "white_balance_anchor_strength"
            ],
        },
    )()
    columns = [
        "raw",
        "neutral_target",
        "ridge_neutral",
        "direct_AO6",
        "global_AO6",
        "ridge_AO6",
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
    selection_records = []
    for row_index, result_row in enumerate(selected):
        pair_id = str(result_row["pair_id"])
        source_row = source_rows[pair_id]
        parameters = parameter_rows[pair_id]
        raw, _ = load_raw_default(
            ROOT / source_row["raw_path"], maximum_side=512
        )
        raw = np.clip(raw, 0.0, 1.0)
        expert = load_expert_icc_srgb(
            ROOT / source_row["expert_path"], maximum_side=512
        )
        if expert.shape != raw.shape:
            expert = resize_to_shape(expert, raw.shape[:2])
        expert = np.clip(expert, 0.0, 1.0)
        target = filtered_target(raw, expert, target_policy)
        global_neutral = apply_neutral_base(
            raw, np.asarray(parameters["global_parameters"])
        )
        ridge_neutral = apply_neutral_base(
            raw, np.asarray(parameters["ridge_parameters"])
        )
        direct_look, _ = renderer(raw)
        global_look, _ = renderer(global_neutral)
        ridge_look, _ = renderer(ridge_neutral)
        target_look, _ = renderer(target)
        arrays = [
            raw,
            target,
            ridge_neutral,
            direct_look,
            global_look,
            ridge_look,
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
        selection_records.append(
            {
                "pair_id": pair_id,
                "camera_group_id": result_row["camera_group_id"],
                "ridge_rmse": result_row["ridge"][
                    "rmse_to_fixed_look_target"
                ],
                "global_minus_ridge_rmse": result_row["global"][
                    "rmse_to_fixed_look_target"
                ]
                - result_row["ridge"]["rmse_to_fixed_look_target"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "visual_stress_sheet.png"
    sheet.save(sheet_path, format="PNG", compress_level=6)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
