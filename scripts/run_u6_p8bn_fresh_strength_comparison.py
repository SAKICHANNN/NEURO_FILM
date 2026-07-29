#!/usr/bin/env python3
"""Render the frozen fresh P8BN population at strengths 0.80 and 1.00."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_strength_file import (  # noqa: E402
    render_native_standard_strength_file_to_png16,
)
from src.film_physics.native_standard_strength_output import (  # noqa: E402
    verify_native_standard_strength_png16,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)


CONFIG = ROOT / "configs/u6_p8bn_fresh_strength_population_v1.json"
OUTPUT = ROOT / "outputs/u6_p8bn_fresh_strength_population_v1"
COMPARISON = OUTPUT / "comparison_v1"
SCHEMA = "neuro_film.u6_p8bn_fresh_strength_comparison_result.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"]["path"],
        config["parent_decision"]["sha256"],
    )
    if (
        parent["result"]["status"]
        != "pass-local-versioned-strength-transaction"
        or config["comparison"]["strengths"] != [0.8, 1.0]
    ):
        raise ValueError("P8BN parent or strength grid drift")
    report = json.loads((OUTPUT / "automatic_report.json").read_text())
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if (
        not report["automatic_pass"]
        or report["decoded_row_count"] != len(manifest)
        or len(manifest) < config["preflight_gate"]["minimum_decoded_rows"]
    ):
        raise ValueError("P8BN source preflight did not pass")
    config_rows = {row["id"]: row for row in config["candidates"]}
    for row in manifest:
        source = config_rows[row["id"]]
        if (
            row["raw_sha256"] != source["sha256"]
            or _sha(ROOT / source["path"]) != source["sha256"]
        ):
            raise ValueError("P8BN source identity drift")
    review = json.loads((OUTPUT / "visual_source_review.json").read_text())
    if (
        review["confirmed_severe_source_artifact_count"] != 0
        or sorted(review["eligible_ids"])
        != sorted(row["id"] for row in manifest)
    ):
        raise ValueError("P8BN source visual gate did not pass")
    return parent, manifest


def _contact_sheet(
    rows: list[dict[str, Any]],
    *,
    round_index: int,
    output_path: Path,
    mapping_path: Path,
) -> None:
    font = ImageFont.load_default()
    mapping: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []
    for row in rows:
        rng = random.Random(
            hashlib.sha256(
                f"u6-p8bn-round-{round_index}:{row['id']}".encode()
            ).digest()
        )
        order = [0.8, 1.0]
        rng.shuffle(order)
        mapping.append(
            {
                "id": row["id"],
                "A": order[0],
                "B": order[1],
            }
        )
        pair = []
        for strength in order:
            label = f"s{int(round(strength * 100)):03d}"
            with Image.open(
                COMPARISON / row["id"] / f"{label}.png"
            ) as opened:
                image = opened.convert("RGB")
                image.thumbnail((520, 360), Image.Resampling.LANCZOS)
            pair.append(image)
        tile = Image.new("RGB", (1080, 410), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((8, 5), row["id"], fill="black", font=font)
        for index, (label, image) in enumerate(zip(("A", "B"), pair)):
            x = 8 + index * 535
            draw.text((x, 24), label, fill="black", font=font)
            tile.paste(image, (x, 44))
        tiles.append(tile)
    sheet = Image.new("RGB", (1080, len(tiles) * 410 + 32), "white")
    ImageDraw.Draw(sheet).text(
        (8, 8),
        f"U6.P8BN blind strength round {round_index}",
        fill="black",
        font=font,
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, 32 + index * 410))
    sheet.save(output_path, "PNG")
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    _, manifest = _validate(config)
    if COMPARISON.exists():
        raise FileExistsError("P8BN comparison is create-only")
    COMPARISON.mkdir(parents=True)
    base = json.loads(
        (
            ROOT / "configs/u6_p8bb_native_standard_working_image_resources_v1.json"
        ).read_text()
    )
    p8aw._patch_runtime()
    builds = p8aq._build_components(base, COMPARISON / "binaries")
    package = json.loads((ROOT / base["package"]).read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / base["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths={
            name: Path(row["dll_path"]) for name, row in builds.items()
        },
    )
    results: list[dict[str, Any]] = []
    for row in manifest:
        row_dir = COMPARISON / row["id"]
        row_dir.mkdir()
        boundary_by_strength: dict[float, float] = {}
        for strength in config["comparison"]["strengths"]:
            label = f"s{int(round(strength * 100)):03d}"
            paths = {
                "raw_output_path": row_dir / f"{label}.f32",
                "raw_report_path": row_dir / f"{label}.raw.json",
                "png_output_path": row_dir / f"{label}.png",
                "png_report_path": row_dir / f"{label}.png.json",
            }
            result = render_native_standard_strength_file_to_png16(
                runtime,
                input_path=ROOT / row["raw_path"],
                strength=strength,
                **paths,
            )
            verified = verify_native_standard_strength_png16(
                report_path=paths["png_report_path"],
                expected_report_sha256=result["png_report_sha256"],
                expected_delivery_id=result["png_delivery_id"],
            )
            code = cv2.imread(
                str(paths["png_output_path"]), cv2.IMREAD_UNCHANGED
            )
            if code is None or code.dtype != np.uint16:
                raise RuntimeError("P8BN PNG decode drift")
            boundary = float(
                np.mean(np.any((code == 0) | (code == 65535), axis=2))
            )
            boundary_by_strength[float(strength)] = boundary
            results.append(
                {
                    "id": row["id"],
                    "strength": strength,
                    "png_output_sha256": result["png_output_sha256"],
                    "png_report_sha256": result["png_report_sha256"],
                    "verification_id": verified["verification_id"],
                    "output_code_boundary_fraction": boundary,
                }
            )
            paths["raw_report_path"].unlink()
            paths["raw_output_path"].unlink()
        results[-2]["new_boundary_fraction_vs_1_00"] = max(
            0.0,
            boundary_by_strength[0.8] - boundary_by_strength[1.0],
        )
        results[-1]["new_boundary_fraction_vs_1_00"] = 0.0

    for round_index in (1, 2, 3):
        _contact_sheet(
            manifest,
            round_index=round_index,
            output_path=COMPARISON / f"blind_round_{round_index}.png",
            mapping_path=COMPARISON / f"blind_round_{round_index}_mapping.json",
        )
    gates = config["comparison"]["automatic_gate"]
    automatic_pass = all(
        row["output_code_boundary_fraction"]
        <= gates["maximum_output_code_boundary_fraction"]
        and row["new_boundary_fraction_vs_1_00"]
        <= gates["maximum_new_boundary_fraction_at_0_80"]
        and len(row["verification_id"]) == 64
        for row in results
    )
    report = {
        "schema": SCHEMA,
        "config_sha256": _sha(CONFIG),
        "preflight_manifest_sha256": _sha(OUTPUT / "manifest.json"),
        "preflight_report_sha256": _sha(OUTPUT / "automatic_report.json"),
        "visual_source_review_sha256": _sha(
            OUTPUT / "visual_source_review.json"
        ),
        "factory_receipt_sha256": factory["receipt_sha256"],
        "rows": results,
        "automatic_gate_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    p8aq.write_report(COMPARISON / "report.json", report)
    return report


def main() -> None:
    report = run(json.loads(CONFIG.read_text()))
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "row_count": len(report["rows"]),
                "report_sha256": _sha(COMPARISON / "report.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
