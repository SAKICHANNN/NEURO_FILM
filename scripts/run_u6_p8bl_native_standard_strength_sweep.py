#!/usr/bin/env python3
"""Run the frozen P8BL display-domain strength comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
from src.color_engine.srgb_transfer import (  # noqa: E402
    linear_srgb_to_encoded,
)
from src.film_physics.native_standard_strength import (  # noqa: E402
    apply_native_standard_display_strength,
)
from src.preprocess import load_working_image  # noqa: E402
from src.preprocess.output_encode import save_srgb16_png  # noqa: E402


SCHEMA = "neuro_film.u6_p8bl_native_standard_strength_sweep_contract.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported P8BL contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    source_contract = p8aq._load_exact_json(
        ROOT / config["source_contract"],
        config["source_contract_sha256"],
    )
    source_report = p8aq._load_exact_json(
        ROOT / config["source_report"],
        config["source_report_sha256"],
    )
    if (
        parent["result"]["status"] != "pass_limited"
        or not str(parent["next_leaf"]).startswith("U6.P8BL")
        or not source_report["automatic_gate_pass"]
    ):
        raise ValueError("P8BL parent drift")
    strengths = [float(value) for value in config["strengths"]]
    if strengths != [0.65, 0.8, 1.0]:
        raise ValueError("P8BL strength grid drift")
    stride = int(config["style_diagnostic"]["pixel_stride"])
    if stride != 8:
        raise ValueError("P8BL style diagnostic drift")
    report_by_id = {row["id"]: row for row in source_report["rows"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    previews: list[tuple[str, list[Image.Image]]] = []
    for row in source_contract["rows"]:
        source_path = ROOT / row["path"]
        if _sha256(source_path) != row["sha256"]:
            raise ValueError("P8BL source identity drift")
        working = load_working_image(source_path)
        source_encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(working.pixels).astype(np.float32)
        )
        full_path = (
            ROOT
            / "outputs/u6_p8bk_native_standard_content_stress_v1"
            / row["id"]
            / "render.png"
        )
        full_code = cv2.imread(
            str(full_path), cv2.IMREAD_UNCHANGED
        )[..., ::-1]
        if (
            full_code.dtype != np.uint16
            or _sha256(full_path)
            != report_by_id[row["id"]]["png_output_sha256"]
        ):
            raise ValueError("P8BL full-strength output drift")
        full = np.ascontiguousarray(
            full_code.astype(np.float32) / np.float32(65535.0)
        )
        row_previews: list[Image.Image] = []
        for strength in strengths:
            output = apply_native_standard_display_strength(
                source_encoded,
                full,
                strength=strength,
            )
            source_lab = rgb2lab(
                source_encoded[::stride, ::stride].astype(np.float64)
            )
            output_lab = rgb2lab(
                output[::stride, ::stride].astype(np.float64)
            )
            style_delta = np.linalg.norm(
                output_lab - source_lab, axis=2
            )
            label = f"s{int(round(strength * 100)):03d}"
            path = output_dir / row["id"] / f"{label}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_srgb16_png(output, path)
            code = cv2.imread(
                str(path), cv2.IMREAD_UNCHANGED
            )[..., ::-1]
            boundary = float(
                np.mean(np.any((code == 0) | (code == 65535), axis=2))
            )
            full_boundary = float(
                np.mean(
                    np.any(
                        (full_code == 0) | (full_code == 65535),
                        axis=2,
                    )
                )
            )
            preview = Image.fromarray(
                np.rint(
                    code.astype(np.float32) / np.float32(257.0)
                ).astype(np.uint8),
                mode="RGB",
            )
            preview.thumbnail((500, 420))
            row_previews.append(preview)
            results.append(
                {
                    "id": row["id"],
                    "strength": strength,
                    "output_sha256": _sha256(path),
                    "output_code_boundary_fraction": boundary,
                    "new_output_code_boundary_fraction": max(
                        0.0, boundary - full_boundary
                    ),
                    "full_strength_exact": (
                        strength != 1.0
                        or np.array_equal(code, full_code)
                    ),
                    "median_style_delta_e76": float(
                        np.median(style_delta)
                    ),
                }
            )
        previews.append((row["id"], row_previews))
    sheet = Image.new(
        "RGB", (1580, len(previews) * 470), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (row_id, images) in enumerate(previews):
        y = row_index * 470
        draw.text((10, y + 5), row_id, fill="black")
        for index, (strength, image) in enumerate(
            zip(strengths, images)
        ):
            x = 10 + index * 520
            draw.text(
                (x, y + 25),
                f"global display strength {strength:.2f}",
                fill="black",
            )
            sheet.paste(image, (x, y + 50))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    gates = config["automatic_gate"]
    automatic_pass = all(
        row["output_code_boundary_fraction"]
        <= gates["maximum_output_code_boundary_fraction"]
        and row["new_output_code_boundary_fraction"]
        <= gates["maximum_new_output_code_boundary_fraction"]
        and row["full_strength_exact"]
        for row in results
    )
    report = {
        "schema": (
            "neuro_film.u6_p8bl_native_standard_strength_sweep_result.v1"
        ),
        "rows": results,
        "automatic_gate_pass": automatic_pass,
        "contact_sheet": str(sheet_path.relative_to(ROOT)),
        "contact_sheet_sha256": _sha256(sheet_path),
        "visual_review_allowed": automatic_pass,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    p8aq.write_report(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8bl_native_standard_strength_sweep_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bl_native_standard_strength_sweep_v1",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = run(json.loads(config_path.read_text()), output_dir)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
