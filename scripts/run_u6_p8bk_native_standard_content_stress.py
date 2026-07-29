#!/usr/bin/env python3
"""Render the bounded P8BK RAW content-stress cohort."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_file import (  # noqa: E402
    render_native_standard_file_to_png16,
)
from src.film_physics.native_standard_output import (  # noqa: E402
    verify_native_standard_png16,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)
from src.preprocess import (  # noqa: E402
    load_working_image,
    working_image_to_legacy_srgb8,
)


SCHEMA = "neuro_film.u6_p8bk_native_standard_content_stress_contract.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_list(path: Path, expected_sha256: str) -> list[Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"hash mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"JSON root must be a list: {path}")
    return value


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or len(config["rows"]) != 4:
        raise ValueError("unsupported P8BK contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    base = p8aq._load_exact_json(
        ROOT / config["working_image_contract"],
        config["working_image_contract_sha256"],
    )
    p8aq._load_exact_json(
        ROOT / config["prior_visual_review"],
        config["prior_visual_review_sha256"],
    )
    prior_rows = _load_exact_list(
        ROOT / config["prior_cc0_manifest"],
        config["prior_cc0_manifest_sha256"],
    )
    if (
        parent["result"]["status"] != "pass_limited"
        or not str(parent["next_leaf"]).startswith("U6.P8BK")
    ):
        raise ValueError("P8BK parent drift")
    prior_by_sha = {row["raw_sha256"]: row for row in prior_rows}
    for row in config["rows"]:
        if _sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("P8BK RAW identity drift")
        if "rawpixls_ai1_v1" in row["path"]:
            prior = prior_by_sha.get(row["sha256"])
            if (
                prior is None
                or prior["raw_path"] != row["path"]
                or not prior["license"].startswith("CC0/")
            ):
                raise ValueError("P8BK CC0 lineage drift")
    p8aw._patch_runtime()
    builds = p8aq._build_components(base, output_dir / "binaries")
    package = json.loads((ROOT / base["package"]).read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / base["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory_receipt = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths={
            name: Path(value["dll_path"])
            for name, value in builds.items()
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    sheet_rows: list[tuple[str, Image.Image, Image.Image]] = []
    for row in config["rows"]:
        row_dir = output_dir / row["id"]
        row_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / row["path"]
        working = load_working_image(source)
        input_preview = working_image_to_legacy_srgb8(working)
        input_preview.thumbnail((640, 500))
        paths = {
            "raw_output_path": row_dir / "render.f32",
            "raw_report_path": row_dir / "render.raw.json",
            "png_output_path": row_dir / "render.png",
            "png_report_path": row_dir / "render.png.json",
        }
        rendered = render_native_standard_file_to_png16(
            runtime,
            input_path=source,
            **paths,
        )
        verified = verify_native_standard_png16(
            report_path=paths["png_report_path"],
            expected_report_sha256=rendered["png_report_sha256"],
            expected_delivery_id=rendered["png_delivery_id"],
        )
        decoded = cv2.imread(
            str(paths["png_output_path"]), cv2.IMREAD_UNCHANGED
        )[..., ::-1]
        boundary = float(
            np.mean(np.any((decoded == 0) | (decoded == 65535), axis=2))
        )
        output_preview = Image.fromarray(
            np.rint(decoded.astype(np.float32) / 257.0).astype(np.uint8),
            mode="RGB",
        )
        output_preview.thumbnail((640, 500))
        sheet_rows.append((row["id"], input_preview, output_preview))
        results.append(
            {
                "id": row["id"],
                "content_stress": row["content_stress"],
                "shape": list(decoded.shape),
                "png_output_sha256": rendered["png_output_sha256"],
                "verification_id": verified["verification_id"],
                "output_code_boundary_fraction": boundary,
            }
        )
        paths["raw_output_path"].unlink()
        paths["raw_report_path"].unlink()
    sheet = Image.new("RGB", (1320, len(sheet_rows) * 550), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, before, after) in enumerate(sheet_rows):
        y = index * 550
        draw.text((10, y + 5), f"{name}: decoded input", fill="black")
        draw.text((670, y + 5), "native Standard output", fill="black")
        sheet.paste(before, (10, y + 35))
        sheet.paste(after, (670, y + 35))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    automatic_pass = all(
        row["output_code_boundary_fraction"]
        <= config["automatic_gate"][
            "maximum_output_code_boundary_fraction"
        ]
        for row in results
    )
    report = {
        "schema": (
            "neuro_film.u6_p8bk_native_standard_content_stress_result.v1"
        ),
        "factory_receipt_sha256": factory_receipt["receipt_sha256"],
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
        default="configs/u6_p8bk_native_standard_content_stress_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bk_native_standard_content_stress_v1",
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
