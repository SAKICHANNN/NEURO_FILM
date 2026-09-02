from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import (
    render_three_stock_previews_to_directory,
)
from src.preprocess import load_jpeg_preview_working_image, load_working_image

CONFIG = ROOT / "configs/u7_20g_exif_oriented_jpeg_preview_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
CANONICAL_PRODUCT_SHA256 = (
    "fc51547d1e00a0a4a36dce96847afe09d27b3b531b65172d3f32fa2a46d2087d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _oriented_jpeg(path: Path, orientation: int) -> None:
    height, width = 240, 320
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[: height // 2, : width // 2] = (235, 24, 18)
    rgb[: height // 2, width // 2 :] = (24, 220, 35)
    rgb[height // 2 :, : width // 2] = (20, 48, 230)
    rgb[height // 2 :, width // 2 :] = (225, 210, 20)
    image = Image.fromarray(rgb, mode="RGB")
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, quality=96, subsampling=0, exif=exif)


def _canonical_product_source(path: Path) -> None:
    y, x = np.mgrid[0:43, 0:61]
    source = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 11) % 256,
            (x * 13 + y * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(source, mode="RGB").save(path)


def _orientation_record(source: Path, orientation: int) -> dict[str, Any]:
    target_width, target_height = (
        (90, 120) if orientation in {5, 6, 7, 8} else (120, 90)
    )
    scaled = load_jpeg_preview_working_image(
        source,
        target_width=target_width,
        target_height=target_height,
    )
    full = load_working_image(source)
    scaled_common = cv2.resize(
        scaled.pixels,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    full_common = cv2.resize(
        full.pixels,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    rmse = float(np.sqrt(np.mean((scaled_common - full_common) ** 2)))
    return {
        "orientation": orientation,
        "source_sha256": _sha256(source),
        "target_width": target_width,
        "target_height": target_height,
        "scaled_width": int(scaled.pixels.shape[1]),
        "scaled_height": int(scaled.pixels.shape[0]),
        "full_width": int(full.pixels.shape[1]),
        "full_height": int(full.pixels.shape[0]),
        "scaled_float32_sha256": _array_sha256(scaled.pixels),
        "full_float32_sha256": _array_sha256(full.pixels),
        "common_rmse": rmse,
        "covers_target": (
            scaled.pixels.shape[1] >= target_width
            and scaled.pixels.shape[0] >= target_height
        ),
        "orientation_applied": scaled.orientation_applied,
    }


def _canonical_product_probe(scratch: Path) -> dict[str, Any]:
    source = scratch / "canonical-source.png"
    output = scratch / "canonical-ektar.png"
    _canonical_product_source(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--output",
            str(output),
            "--write-recipe",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    return {
        "returncode": completed.returncode,
        "output_sha256": _sha256(output),
        "expected_output_sha256": CANONICAL_PRODUCT_SHA256,
        "look_amount": recipe["render"]["look_amount"],
        "evidence_grade": recipe["claim"]["evidence_grade"],
        "calibrated_reference_allowed": recipe["claim"]["calibrated_reference_allowed"],
    }


def build_report(*, scratch: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if scratch.exists():
        raise FileExistsError("scratch path must be absent")
    scratch.mkdir(parents=True)
    try:
        orientations = list(range(1, 9))
        execution = orientations if order == "forward" else list(reversed(orientations))
        records: dict[int, dict[str, Any]] = {}
        for orientation in execution:
            source = scratch / f"orientation-{orientation}.jpg"
            _oriented_jpeg(source, orientation)
            records[orientation] = _orientation_record(source, orientation)

        portrait_source = scratch / "orientation-6.jpg"
        preview_directory = scratch / "three-look-preview"
        manifest = render_three_stock_previews_to_directory(
            portrait_source,
            preview_directory,
            root=ROOT,
            profile_path=PROFILE,
            statistics_path=STATISTICS,
            guardrails_path=GUARDRAILS,
            max_preview_pixels=12_000,
            tile_size=23,
            tile_workers=1,
            jpeg_scaled_decode=True,
            include_input_preview=True,
        )
        preview_rows = []
        for row in manifest["rows"]:
            path = Path(row["output_path"])
            with Image.open(path) as image:
                size = list(image.size)
            preview_rows.append(
                {
                    "style_id": row["style_id"],
                    "sha256": _sha256(path),
                    "size": size,
                }
            )
        input_preview = Path(manifest["input_preview"]["output_path"])
        with Image.open(input_preview) as image:
            input_preview_size = list(image.size)

        invalid = scratch / "orientation-9.jpg"
        _oriented_jpeg(invalid, 9)
        invalid_error = None
        try:
            load_jpeg_preview_working_image(invalid, target_width=120, target_height=90)
        except ValueError as exc:
            invalid_error = str(exc)

        product = _canonical_product_probe(scratch)
        rows = [records[index] for index in orientations]
        all_rmse = [float(row["common_rmse"]) for row in rows]
        gates = {
            "all_orientation_values_supported": len(rows) == 8,
            "oriented_geometry_exact": all(
                (row["full_width"], row["full_height"])
                == ((240, 320) if row["orientation"] in {5, 6, 7, 8} else (320, 240))
                for row in rows
            ),
            "scaled_decode_covers_requested_oriented_geometry": all(
                row["covers_target"] for row in rows
            ),
            "scaled_full_structure_rmse": max(all_rmse) <= 0.025,
            "portrait_preview_geometry": (
                manifest["source_width"] == 240
                and manifest["source_height"] == 320
                and manifest["preview_height"] > manifest["preview_width"]
                and manifest["decoded_height"] > manifest["decoded_width"]
                and input_preview_size[1] > input_preview_size[0]
                and all(row["size"][1] > row["size"][0] for row in preview_rows)
            ),
            "invalid_orientation_rejects": (
                invalid_error == "scaled preview decode has invalid EXIF orientation"
            ),
            "canonical_product_output_exact": (
                product["returncode"] == 0
                and product["output_sha256"] == CANONICAL_PRODUCT_SHA256
            ),
            "claim_ceiling_exact": (
                product["evidence_grade"] == "look-approximation"
                and product["calibrated_reference_allowed"] is False
            ),
        }
        return {
            "schema_version": "neuro-film.u7-20g-exif-oriented-jpeg-preview-result.v1",
            "source_commit": _git_head(),
            "config_sha256": _sha256(CONFIG),
            "status": (
                "PASS_PRIVATE_U7_20G_EXIF_ORIENTED_JPEG_PREVIEW"
                if all(gates.values())
                else "FAIL_CLOSED_U7_20G_EXIF_ORIENTED_JPEG_PREVIEW"
            ),
            "orientations": rows,
            "maximum_scaled_full_rmse": max(all_rmse),
            "portrait_preview": {
                "source_width": manifest["source_width"],
                "source_height": manifest["source_height"],
                "preview_width": manifest["preview_width"],
                "preview_height": manifest["preview_height"],
                "decoded_width": manifest["decoded_width"],
                "decoded_height": manifest["decoded_height"],
                "input_preview_sha256": _sha256(input_preview),
                "input_preview_size": input_preview_size,
                "rows": preview_rows,
            },
            "invalid_orientation_error": invalid_error,
            "canonical_product": product,
            "gates": gates,
            "claim_ceiling": json.loads(CONFIG.read_text(encoding="utf-8"))[
                "claim_ceiling"
            ],
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output path must be absent")
    report = build_report(scratch=args.scratch, order=args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if report["status"].startswith("PASS_PRIVATE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
