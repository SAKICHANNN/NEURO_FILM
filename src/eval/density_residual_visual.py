"""Blind and severe-review sheets for the fixed AZ0 density factorization."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.eval.global_frontier import load_frozen_samples, sha256_file


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> tuple[Path, dict[str, Any]]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"AZ0V hash mismatch: {path}")
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _candidate_paths(
    root: Path,
    config: dict[str, Any],
    gold_ids: set[str],
) -> dict[str, Path]:
    report_path, report = _load_hashed_json(
        root,
        config["candidate"]["report"],
        config["candidate"]["report_sha256"],
    )
    if (
        not report.get("automatic_pass")
        or report.get("next") != "fixed_full_resolution_blind_review"
    ):
        raise ValueError("AZ0 automatic gate forbids visual review")
    paths: dict[str, Path] = {}
    for row in report["records"]:
        if row["split"] != "gold":
            continue
        if row["candidate_id"] != config["candidate"]["candidate_id"]:
            raise ValueError("AZ0 candidate identity drift")
        sample_id = str(row["sample_id"])
        path = report_path.parent / row["output"]
        if sample_id not in gold_ids or sha256_file(path) != row["output_sha256"]:
            raise ValueError("AZ0 candidate output drift")
        paths[sample_id] = path
    if paths.keys() != gold_ids:
        raise ValueError("AZ0 candidate gold coverage drift")
    return paths


def _comparator_paths(
    root: Path,
    config: dict[str, Any],
    gold_ids: set[str],
) -> dict[str, Path]:
    manifest_path, manifest = _load_hashed_json(
        root,
        config["comparator"]["manifest"],
        config["comparator"]["manifest_sha256"],
    )
    paths: dict[str, Path] = {}
    for row in manifest["records"]:
        if row["candidate_id"] != config["comparator"]["candidate_id"]:
            continue
        sample_id = str(row["sample_id"])
        if sample_id not in gold_ids:
            continue
        path = manifest_path.parent / row["output"]
        if sha256_file(path) != row["output_sha256"]:
            raise ValueError("AO6 comparator output drift")
        paths[sample_id] = path
    if paths.keys() != gold_ids:
        raise ValueError("AO6 comparator gold coverage drift")
    return paths


def _sheet(
    rows: list[tuple[str, list[tuple[str, Path]]]],
    path: Path,
) -> None:
    tile_width, tile_height, header = 420, 280, 28
    columns = len(rows[0][1])
    canvas = Image.new(
        "RGB",
        (columns * tile_width, len(rows) * (tile_height + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, cells) in enumerate(rows):
        y = row_index * (tile_height + header)
        for column_index, (label, image_path) in enumerate(cells):
            with Image.open(image_path) as image:
                tile = ImageOps.fit(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    (tile_width, tile_height),
                    method=Image.Resampling.LANCZOS,
                )
            x = column_index * tile_width
            canvas.paste(tile, (x, y + header))
            draw.text((x + 5, y + 5), f"{label} {sample_id}", fill="black")
    canvas.save(path, format="PNG", compress_level=6)


def _portable_path(path: Path, root: Path) -> str:
    try:
        value = path.relative_to(root)
    except ValueError:
        value = path
    return str(value).replace("\\", "/")


def build_visual_evidence(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if (
        config["experiment_id"]
        != "u5.r2az0v-density-residual-visual-v1"
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or config["blind"].get("score_before_mapping_reveal") is not True
        or int(config["blind"]["round_count"]) != 3
    ):
        raise ValueError("AZ0V frozen boundary drift")
    samples = load_frozen_samples(root, config)
    gold_ids = [
        sample_id
        for sample_id, row in samples.items()
        if row["split"] == "gold"
    ]
    if len(gold_ids) != int(config["expected_gold_samples"]):
        raise ValueError("AZ0V gold population drift")
    gold_set = set(gold_ids)
    candidate = _candidate_paths(root, config, gold_set)
    comparator = _comparator_paths(root, config, gold_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_path = output_dir / "direct_severe_review.png"
    _sheet(
        [
            (
                sample_id,
                [
                    ("INPUT", root / samples[sample_id]["source_path"]),
                    ("AO6", comparator[sample_id]),
                    ("DENSITY", candidate[sample_id]),
                ],
            )
            for sample_id in gold_ids
        ],
        direct_path,
    )

    mappings: dict[str, Any] = {}
    blind_sheets = []
    for round_index in range(1, 4):
        round_mapping: dict[str, dict[str, str]] = {}
        rows = []
        for sample_index, sample_id in enumerate(gold_ids):
            roles = ["candidate", "comparator"]
            random.Random(
                int(config["blind"]["seed"])
                + round_index * 1009
                + sample_index * 9176
            ).shuffle(roles)
            round_mapping[sample_id] = {
                "A": roles[0],
                "B": roles[1],
            }
            paths = {"candidate": candidate[sample_id], "comparator": comparator[sample_id]}
            rows.append(
                (
                    sample_id,
                    [
                        ("INPUT", root / samples[sample_id]["source_path"]),
                        ("A", paths[roles[0]]),
                        ("B", paths[roles[1]]),
                    ],
                )
            )
        mappings[f"round_{round_index}"] = round_mapping
        sheet_path = output_dir / f"blind_round_{round_index}.png"
        _sheet(rows, sheet_path)
        blind_sheets.append(
            {
                "round": round_index,
                "path": _portable_path(sheet_path, root),
                "sha256": sha256_file(sheet_path),
            }
        )
    if len(
        {
            json.dumps(mapping, sort_keys=True)
            for mapping in mappings.values()
        }
    ) != 3:
        raise RuntimeError("AZ0V blind mappings are not distinct")
    mapping_raw = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_bytes(mapping_raw)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "direct_severe_review": {
            "path": _portable_path(direct_path, root),
            "sha256": sha256_file(direct_path),
        },
        "blind_sheets": blind_sheets,
        "private_mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "gold_sample_count": len(gold_ids),
        "claim_ceiling": config["claim_ceiling"],
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["build_visual_evidence"]
