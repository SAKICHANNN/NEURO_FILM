"""Deterministic fixed-bank evaluation for the U5.R2B global frontier."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


class GlobalFrontierError(ValueError):
    """Raised when the frozen bank or its evidence is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_samples(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = root / str(config["frozen_set"])
    if sha256_file(path) != config["frozen_set_sha256"]:
        raise GlobalFrontierError("frozen set hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_set", payload)
    rows = frozen.get("samples")
    if not isinstance(rows, list):
        raise GlobalFrontierError("frozen set has no sample list")
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("availability") != "available" or row.get("split") not in {"gold", "stress"}:
            continue
        sample_id = str(row["id"])
        source = root / str(row["source_path"])
        if sha256_file(source) != row["source_sha256"]:
            raise GlobalFrontierError(f"source hash mismatch: {sample_id}")
        samples[sample_id] = dict(row)
    gold = sum(row["split"] == "gold" for row in samples.values())
    stress = sum(row["split"] == "stress" for row in samples.values())
    if gold != int(config["expected_gold_samples"]) or stress != int(config["expected_stress_samples"]):
        raise GlobalFrontierError("frozen gold/stress sample counts do not match")
    return samples


def new_hard_clipping_fraction(
    source: np.ndarray,
    output: np.ndarray,
    epsilon: float,
) -> float:
    source_values = np.asarray(source, dtype=np.float64)
    output_values = np.asarray(output, dtype=np.float64)
    if source_values.shape != output_values.shape:
        raise GlobalFrontierError("source/output sample grids differ")
    if not 0.0 <= epsilon < 0.5:
        raise GlobalFrontierError("clipping epsilon must be in [0, 0.5)")
    output_endpoint = (output_values <= epsilon) | (output_values >= 1.0 - epsilon)
    source_endpoint = (source_values <= epsilon) | (source_values >= 1.0 - epsilon)
    return float(np.mean(output_endpoint & ~source_endpoint))


def evaluate_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    renderer = root / str(config["renderer_script"])
    if sha256_file(renderer) != config["renderer_script_sha256"]:
        raise GlobalFrontierError("renderer script hash mismatch")
    samples = load_frozen_samples(root, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen_set_sha256") != config["frozen_set_sha256"]:
        raise GlobalFrontierError("render manifest frozen-set hash mismatch")

    candidates = tuple(str(value) for value in config["primary_candidate_ids"])
    expected = {(candidate, sample_id) for candidate in candidates for sample_id in samples}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected:
            continue
        if key in records:
            raise GlobalFrontierError(f"duplicate candidate/sample record: {key}")
        records[key] = dict(row)
    missing = sorted(expected - records.keys())
    if missing:
        raise GlobalFrontierError(f"frontier manifest is incomplete: {missing[:3]}")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    per_candidate: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        per_image: list[dict[str, Any]] = []
        for sample_id, sample in samples.items():
            record = records[(candidate, sample_id)]
            output_path = root / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise GlobalFrontierError(f"output hash mismatch: {candidate}/{sample_id}")
            source_path = root / str(sample["source_path"])
            with Image.open(source_path) as source_image, Image.open(output_path) as output_image:
                source_size = ImageOps.exif_transpose(source_image).size
                output_rgb = ImageOps.exif_transpose(output_image)
                if output_rgb.mode != "RGB" or output_rgb.size != source_size:
                    raise GlobalFrontierError(
                        f"output decode/dimension mismatch: {candidate}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            if not np.all(np.isfinite(output_pixels)):
                raise GlobalFrontierError(f"non-finite output: {candidate}/{sample_id}")
            style, residual = style_and_basic_residual(source_pixels, output_pixels)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels, output_pixels, epsilon
                    ),
                    "output_minimum": float(np.min(output_pixels)),
                    "output_maximum": float(np.max(output_pixels)),
                    "output_sha256": record["output_sha256"],
                }
            )
        gold_rows = [row for row in per_image if row["split"] == "gold"]
        stress_rows = [row for row in per_image if row["split"] == "stress"]
        summary = {
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in gold_rows])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median([row["median_non_basic_residual_delta_e76"] for row in gold_rows])
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in gold_rows)
            ),
            "worst_stress_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in stress_rows)
            ),
            "per_image": per_image,
        }
        gates = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary["gold_median_non_basic_residual_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_non_basic_residual_delta_e76"]),
            "gold_clipping": summary["worst_gold_new_hard_clipping_fraction"]
            <= float(config["metrics"]["maximum_worst_gold_new_hard_clipping_fraction"]),
        }
        summary["automatic_gates"] = gates
        summary["automatic_survivor"] = all(gates.values())
        per_candidate[candidate] = summary

    survivors = [
        candidate
        for candidate in candidates
        if per_candidate[candidate]["automatic_survivor"]
    ]
    return {
        "candidate_count": len(candidates),
        "sample_count": len(samples),
        "gold_sample_count": sum(row["split"] == "gold" for row in samples.values()),
        "stress_sample_count": sum(row["split"] == "stress" for row in samples.values()),
        "all_source_and_output_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "candidates": per_candidate,
        "automatic_survivors": survivors,
        "automatic_decision": (
            "visual_gate_required" if survivors else "no_automatic_survivor"
        ),
    }


def build_blind_contact_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    survivors: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    if not survivors:
        return {"rounds": [], "mapping": {}, "reason": "no_automatic_survivor"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = load_frozen_samples(root, config)
    gold_ids = [sample_id for sample_id, row in samples.items() if row["split"] == "gold"]
    by_key = {
        (str(row["candidate_id"]), str(row["sample_id"])): root / str(row["output"])
        for row in manifest["records"]
    }
    columns = list(dict.fromkeys(["bland_safe_rich_control", *survivors]))
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict[str, str]] = {}
    round_paths: list[str] = []
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = columns.copy()
        random.Random(int(config["render_seed"]) + round_index).shuffle(shuffled)
        labels = {candidate: chr(ord("A") + index) for index, candidate in enumerate(shuffled)}
        mapping[f"round_{round_index}"] = {
            label: candidate for candidate, label in labels.items()
        }
        sheet = _contact_sheet(root, samples, gold_ids, shuffled, labels, by_key)
        path = output_dir / f"blind_round_{round_index}.png"
        sheet.save(path, format="PNG")
        round_paths.append(str(path.relative_to(root)).replace("\\", "/"))
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "rounds": round_paths,
        "mapping": str(mapping_path.relative_to(root)).replace("\\", "/"),
    }


def _contact_sheet(
    root: Path,
    samples: Mapping[str, Mapping[str, Any]],
    gold_ids: Sequence[str],
    columns: Sequence[str],
    labels: Mapping[str, str],
    by_key: Mapping[tuple[str, str], Path],
) -> Image.Image:
    tile_width, tile_height, header = 300, 210, 26
    canvas = Image.new(
        "RGB",
        ((len(columns) + 1) * tile_width, len(gold_ids) * (tile_height + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(gold_ids):
        y = row_index * (tile_height + header)
        source = root / str(samples[sample_id]["source_path"])
        paths = [source, *(by_key[(candidate, sample_id)] for candidate in columns)]
        names = [f"INPUT {sample_id}", *(labels[candidate] for candidate in columns)]
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
    return canvas
