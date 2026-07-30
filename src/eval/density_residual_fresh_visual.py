"""Build frozen AZ1 fresh-population severe and blind visual evidence."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.eval.density_residual_fresh_confirmation import validate_contract
from src.eval.global_frontier import sha256_file


def _tile(
    source: Path, left: Path, right: Path, labels: tuple[str, str, str]
) -> Image.Image:
    images = []
    for path in (source, left, right):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((420, 290), Image.Resampling.LANCZOS)
            images.append(image.copy())
    tile = Image.new("RGB", (1260, 322), "white")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(
        zip(labels, images, strict=True)
    ):
        x = index * 420
        draw.text((x + 5, 4), label, fill="black", font=font)
        tile.paste(image, (x, 24))
    return tile


def _write_sheet(tiles: list[Image.Image], path: Path) -> str:
    sheet = Image.new("RGB", (1260, len(tiles) * 322), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * 322))
    sheet.save(path, "PNG", compress_level=6)
    return sha256_file(path)


def _report_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_fresh_visual_evidence(
    *,
    root: Path,
    config: Mapping[str, Any],
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    report_path = candidate_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not report.get("automatic_pass")
        or report.get("stable_evidence_id")
        != "8a5835bba07b5af0c5094d1305f8e1fec5a9f1d56c59255ee722b6d4276ff7ff"
    ):
        raise ValueError("AZ1 automatic evidence is not eligible")
    rows = {row["sample_id"]: row for row in report["records"]}
    fixed_ids = list(config["visual_protocol"]["fixed_ids"])
    if set(fixed_ids) - set(rows):
        raise ValueError("AZ1 visual population drift")
    output_dir.mkdir(parents=True, exist_ok=False)

    direct_tiles = []
    for sample_id in fixed_ids:
        source = root / str(
            validated["source_rows"][sample_id]["decoded_path"]
        )
        comparator = validated["records"][
            (config["candidate"]["comparator_id"], sample_id)
        ]["absolute_path"]
        candidate = candidate_dir / rows[sample_id]["output"]
        direct_tiles.append(
            _tile(
                source,
                comparator,
                candidate,
                (f"INPUT {sample_id}", "AO6", "AZ1 DENSITY"),
            )
        )
    direct_path = output_dir / "direct_severe_review.png"
    direct_sha = _write_sheet(direct_tiles, direct_path)

    mapping: dict[str, Any] = {}
    blind_rows = []
    for round_index, seed in enumerate(
        config["visual_protocol"]["round_seeds"], start=1
    ):
        round_mapping = {}
        tiles = []
        for sample_id in fixed_ids:
            source = root / str(
                validated["source_rows"][sample_id]["decoded_path"]
            )
            paths = {
                "candidate": candidate_dir / rows[sample_id]["output"],
                "comparator": validated["records"][
                    (config["candidate"]["comparator_id"], sample_id)
                ]["absolute_path"],
            }
            order = ["candidate", "comparator"]
            random.Random(f"{seed}:{sample_id}").shuffle(order)
            round_mapping[sample_id] = {
                "A": order[0],
                "B": order[1],
            }
            tiles.append(
                _tile(
                    source,
                    paths[order[0]],
                    paths[order[1]],
                    (f"INPUT {sample_id}", "A", "B"),
                )
            )
        path = output_dir / f"blind_round_{round_index}.png"
        blind_rows.append(
            {
                "round": round_index,
                "path": _report_path(path, root),
                "sha256": _write_sheet(tiles, path),
            }
        )
        mapping[f"round_{round_index}"] = round_mapping
    mapping_path = output_dir / "private_mapping.json"
    mapping_raw = (
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    mapping_path.write_bytes(mapping_raw)
    build = {
        "schema": "neuro-film.u5-r2az1v-density-residual-fresh-visual.v1",
        "experiment_id": config["experiment_id"],
        "automatic_report_sha256": sha256_file(report_path),
        "automatic_stable_evidence_id": report["stable_evidence_id"],
        "direct_severe_review": {
            "path": _report_path(direct_path, root),
            "sha256": direct_sha,
        },
        "blind_sheets": blind_rows,
        "private_mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    build_path = output_dir / "build_report.json"
    build_path.write_text(
        json.dumps(build, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return build


__all__ = ["build_fresh_visual_evidence"]
