"""Deterministic three-way blind visual sheets for AO5V."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.eval.global_frontier import load_frozen_samples, sha256_file


class ThreeWayLookVisualError(ValueError):
    """Raised when the AO5V visual contract or evidence drifts."""


def blind_orders(seed: int, roles: list[str], count: int) -> list[list[str]]:
    if len(roles) != 3 or len(set(roles)) != 3 or count < 1 or count > 6:
        raise ValueError("AO5V requires three unique roles and 1..6 rounds")
    permutations = [list(values) for values in itertools.permutations(roles)]
    random.Random(seed).shuffle(permutations)
    return permutations[:count]


def _manifest_paths(
    root: Path,
    descriptor: dict[str, Any],
    sample_ids: set[str],
) -> dict[str, Path]:
    manifest_path = root / str(descriptor["manifest"])
    if sha256_file(manifest_path) != descriptor["manifest_sha256"]:
        raise ThreeWayLookVisualError("AO5V manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_id = str(descriptor["candidate_id"])
    paths: dict[str, Path] = {}
    for row in manifest["records"]:
        if row["candidate_id"] != candidate_id:
            continue
        sample_id = str(row["sample_id"])
        if sample_id not in sample_ids:
            continue
        if sample_id in paths:
            raise ThreeWayLookVisualError("AO5V manifest sample drift")
        output = manifest_path.parent / str(row["output"])
        if sha256_file(output) != row["output_sha256"]:
            raise ThreeWayLookVisualError("AO5V output hash mismatch")
        paths[sample_id] = output
    if paths.keys() != sample_ids:
        raise ThreeWayLookVisualError("AO5V manifest coverage incomplete")
    return paths


def build_three_way_blind_sheets(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if (
        config["experiment_id"] != "u5.r2ao5v-three-way-visual-v1"
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or int(config["blind"]["round_count"]) != 3
        or len(config["looks"]) != 3
    ):
        raise ThreeWayLookVisualError("AO5V frozen contract drift")
    roles = [str(value["role"]) for value in config["looks"]]
    orders = blind_orders(
        int(config["blind"]["seed"]),
        roles,
        int(config["blind"]["round_count"]),
    )
    if len({tuple(order) for order in orders}) != len(orders):
        raise ThreeWayLookVisualError("AO5V permutations are not unique")

    samples = load_frozen_samples(root, config)
    gold_ids = [
        sample_id
        for sample_id, value in samples.items()
        if value["split"] == "gold"
    ]
    if len(gold_ids) != int(config["expected_gold_samples"]):
        raise ThreeWayLookVisualError("AO5V gold coverage drift")
    sample_ids = set(gold_ids)
    paths_by_role = {
        descriptor["role"]: _manifest_paths(root, descriptor, sample_ids)
        for descriptor in config["looks"]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, dict[str, str]] = {}
    sheet_records = []
    tile_width, tile_height, header = 320, 220, 28
    for round_index, order in enumerate(orders, start=1):
        labels = {role: chr(ord("A") + index) for index, role in enumerate(order)}
        mappings[f"round_{round_index}"] = {
            label: role for role, label in labels.items()
        }
        canvas = Image.new(
            "RGB",
            (4 * tile_width, len(gold_ids) * (tile_height + header)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, sample_id in enumerate(gold_ids):
            y = row_index * (tile_height + header)
            paths = [
                root / str(samples[sample_id]["source_path"]),
                *(paths_by_role[role][sample_id] for role in order),
            ]
            names = [
                f"INPUT {sample_id}",
                *(labels[role] for role in order),
            ]
            for column_index, (path, name) in enumerate(zip(paths, names)):
                with Image.open(path) as image:
                    tile = ImageOps.fit(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (tile_width, tile_height),
                        method=Image.Resampling.LANCZOS,
                    )
                x = column_index * tile_width
                canvas.paste(tile, (x, y + header))
                draw.text((x + 5, y + 5), name, fill="black")
        sheet_path = output_dir / f"blind_round_{round_index}.png"
        canvas.save(sheet_path, format="PNG", compress_level=6)
        sheet_records.append(
            {
                "round": round_index,
                "path": str(sheet_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(sheet_path),
            }
        )
    mapping_path = output_dir / "private_mapping.json"
    mapping_encoded = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode()
    mapping_path.write_bytes(mapping_encoded)
    return {
        "sheets": sheet_records,
        "private_mapping": str(mapping_path.relative_to(root)).replace("\\", "/"),
        "private_mapping_sha256": hashlib.sha256(mapping_encoded).hexdigest(),
        "distinct_permutations": len({tuple(order) for order in orders}),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ThreeWayLookVisualError",
    "blind_orders",
    "build_three_way_blind_sheets",
]
