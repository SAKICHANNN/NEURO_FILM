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


def adjudicate_fresh_visual(
    *,
    config: Mapping[str, Any],
    observations: Mapping[str, Any],
    build_dir: Path,
) -> dict[str, Any]:
    if (
        observations.get("mapping_unread_when_recorded") is not True
        or observations.get("experiment_id") != "U5.R2AZ1V"
    ):
        raise ValueError("AZ1 observations were not recorded blind")
    build_path = build_dir / "build_report.json"
    mapping_path = build_dir / "private_mapping.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if (
        build.get("experiment_id") != config["experiment_id"]
        or sha256_file(mapping_path) != build["private_mapping_sha256"]
    ):
        raise ValueError("AZ1 visual evidence identity drift")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_sheets = {
        int(row["round"]): row["sha256"] for row in build["blind_sheets"]
    }
    rounds = []
    passing_rounds = 0
    for observed in observations["rounds"]:
        round_index = int(observed["round"])
        if expected_sheets.get(round_index) != observed["sheet_sha256"]:
            raise ValueError("AZ1 blind sheet identity drift")
        round_mapping = mapping[f"round_{round_index}"]
        if set(round_mapping) != set(observed["choices"]):
            raise ValueError("AZ1 blind population drift")
        decoded = {}
        candidate_preferences = 0
        for sample_id, label in observed["choices"].items():
            if label not in {"A", "B"}:
                raise ValueError("invalid AZ1 blind choice")
            arm = round_mapping[sample_id][label]
            decoded[sample_id] = arm
            candidate_preferences += arm == "candidate"
        passed = candidate_preferences >= int(
            config["visual_protocol"][
                "minimum_candidate_preferences_per_passing_round"
            ]
        )
        passing_rounds += passed
        rounds.append(
            {
                "round": round_index,
                "candidate_preferences": candidate_preferences,
                "comparator_preferences": len(decoded)
                - candidate_preferences,
                "decoded_choices": decoded,
                "pass": bool(passed),
            }
        )
    severe = int(
        observations["direct_severe_review"]["confirmed_severe_count"]
    )
    visual_pass = (
        passing_rounds
        >= int(config["visual_protocol"]["minimum_passing_rounds"])
        and severe
        <= int(
            config["visual_protocol"][
                "maximum_confirmed_severe_artifacts"
            ]
        )
    )
    core = {
        "schema": "neuro-film.u5-r2az1v-density-residual-fresh-adjudication.v1",
        "experiment_id": config["experiment_id"],
        "automatic_report_sha256": build["automatic_report_sha256"],
        "automatic_stable_evidence_id": build[
            "automatic_stable_evidence_id"
        ],
        "build_report_sha256": sha256_file(build_path),
        "private_mapping_sha256": sha256_file(mapping_path),
        "rounds": rounds,
        "passing_rounds": passing_rounds,
        "confirmed_severe_artifact_count": severe,
        "visual_gate_passed": visual_pass,
        "production_integration_opened": False,
        "branch": (
            config["branch_rules"]["complete_pass"]
            if visual_pass
            else config["branch_rules"]["visual_fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest(),
    }


__all__ = ["adjudicate_fresh_visual", "build_fresh_visual_evidence"]
