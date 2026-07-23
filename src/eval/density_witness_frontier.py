"""Deterministic rendering and evaluation for the U5.R2E1 density frontier."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.global_frontier import (
    GlobalFrontierError,
    load_frozen_samples,
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.density_domain import operator_from_config


class DensityFrontierError(ValueError):
    """Raised when the frozen E1 contract or evidence is invalid."""


def encoded_srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise DensityFrontierError("encoded sRGB must be finite HxWx3")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise DensityFrontierError("encoded sRGB must be in [0, 1]")
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def linear_srgb_to_encoded(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise DensityFrontierError("linear sRGB must be finite HxWx3")
    if np.any(value < -1e-12) or np.any(value > 1.0 + 1e-12):
        raise DensityFrontierError("linear sRGB escaped [0, 1]")
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


def candidate_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    witnesses = [str(value) for value in config["witness_ids"]]
    strengths = [float(value) for value in config["strengths"]]
    candidates = []
    for witness in witnesses:
        for strength in strengths:
            percent = int(round(strength * 100.0))
            candidate_id = str(config["candidate_id_format"]).format(
                witness=witness,
                strength_percent=percent,
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "witness_id": witness,
                    "strength": strength,
                }
            )
    if len(candidates) != int(config["candidate_count"]):
        raise DensityFrontierError("candidate count does not match frozen config")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise DensityFrontierError("candidate IDs are not unique")
    return candidates


def _load_parent_config(
    root: Path,
    config: Mapping[str, Any],
    path_key: str,
    hash_key: str,
) -> dict[str, Any]:
    path = root / str(config[path_key])
    if sha256_file(path) != str(config[hash_key]):
        raise DensityFrontierError(f"{path_key} hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    density = _load_parent_config(
        root,
        config,
        "density_operator_config",
        "density_operator_config_sha256",
    )
    inherited = _load_parent_config(
        root,
        config,
        "inherited_frontier_config",
        "inherited_frontier_config_sha256",
    )
    comparator_path = root / str(config["comparator_manifest"])
    if sha256_file(comparator_path) != str(config["comparator_manifest_sha256"]):
        raise DensityFrontierError("comparator manifest hash mismatch")
    for key, value in config["metrics"].items():
        if inherited["metrics"].get(key) != value:
            raise DensityFrontierError(f"inherited metric drift: {key}")
    samples = load_frozen_samples(root, config)
    witness_ids = set(str(value) for value in config["witness_ids"])
    if set(density["witnesses"]) != witness_ids:
        raise DensityFrontierError("density witness bank drift")
    return {
        "density": density,
        "inherited": inherited,
        "samples": samples,
        "candidates": candidate_bank(config),
        "comparator_manifest": json.loads(
            comparator_path.read_text(encoding="utf-8")
        ),
    }


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    density = validated["density"]
    samples = validated["samples"]
    operators = {
        witness: operator_from_config(
            payload,
            exposure_floor=float(density["exposure_floor"]),
            matrix_minimum_determinant=float(
                density["parameter_bounds"]["matrix_minimum_determinant"]
            ),
            minimum_endpoint_span=float(
                density["parameter_bounds"]["minimum_endpoint_span"]
            ),
        )
        for witness, payload in density["witnesses"].items()
    }
    source_arrays: dict[str, np.ndarray] = {}
    for sample_id, sample in samples.items():
        source_path = root / str(sample["source_path"])
        with Image.open(source_path) as image:
            source_arrays[sample_id] = (
                np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64)
                / 255.0
            )

    records = []
    for candidate in validated["candidates"]:
        operator = operators[candidate["witness_id"]]
        candidate_dir = output_dir / candidate["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in samples.items():
            source_encoded = source_arrays[sample_id]
            source_linear = encoded_srgb_to_linear(source_encoded)
            output_linear = operator.apply(
                source_linear,
                strength=float(candidate["strength"]),
            )
            output_encoded = linear_srgb_to_encoded(output_linear)
            output_pixels = np.rint(output_encoded * 255.0).astype(np.uint8)
            path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(output_pixels, mode="RGB").save(
                path,
                format="PNG",
                compress_level=6,
            )
            records.append(
                {
                    **candidate,
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "output": f"{candidate['candidate_id']}/{sample_id}.png",
                    "output_sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "density_operator_config_sha256": config["density_operator_config_sha256"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(samples),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    samples = validated["samples"]
    candidates = validated["candidates"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (row["candidate_id"], sample_id)
        for row in candidates
        for sample_id in samples
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected:
            raise DensityFrontierError(f"unexpected manifest record: {key}")
        if key in records:
            raise DensityFrontierError(f"duplicate manifest record: {key}")
        records[key] = dict(row)
    if records.keys() != expected:
        missing = sorted(expected - records.keys())
        raise DensityFrontierError(f"incomplete manifest: {missing[:3]}")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    for candidate_id, candidate in candidate_by_id.items():
        per_image = []
        for sample_id, sample in samples.items():
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise DensityFrontierError(
                    f"output hash mismatch: {candidate_id}/{sample_id}"
                )
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(root / str(sample["source_path"])) as source_image:
                    source_size = ImageOps.exif_transpose(source_image).size
                if output.mode != "RGB" or output.size != source_size:
                    raise DensityFrontierError(
                        f"output decode/dimension mismatch: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(root / str(sample["source_path"]), budget)
            output_pixels = sample_rgb_image(output_path, budget)
            if not np.all(np.isfinite(output_pixels)):
                raise DensityFrontierError("non-finite output pixels")
            style, residual = style_and_basic_residual(source_pixels, output_pixels)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels,
                        output_pixels,
                        epsilon,
                    ),
                    "output_sha256": record["output_sha256"],
                }
            )
        gold = [row for row in per_image if row["split"] == "gold"]
        stress = [row for row in per_image if row["split"] == "stress"]
        summary = {
            "witness_id": candidate["witness_id"],
            "strength": candidate["strength"],
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in gold])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [row["median_non_basic_residual_delta_e76"] for row in gold]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in gold)
            ),
            "worst_stress_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in stress)
            ),
            "per_image": per_image,
        }
        gates = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary["gold_median_non_basic_residual_delta_e76"]
            >= float(
                config["metrics"]["minimum_gold_median_non_basic_residual_delta_e76"]
            ),
            "gold_clipping": summary["worst_gold_new_hard_clipping_fraction"]
            <= float(
                config["metrics"]["maximum_worst_gold_new_hard_clipping_fraction"]
            ),
        }
        summary["automatic_gates"] = gates
        summary["automatic_survivor"] = all(gates.values())
        summaries[candidate_id] = summary

    shortlist = shortlist_candidates(summaries, config)
    return {
        "candidate_count": len(candidates),
        "sample_count": len(samples),
        "gold_sample_count": sum(row["split"] == "gold" for row in samples.values()),
        "stress_sample_count": sum(
            row["split"] == "stress" for row in samples.values()
        ),
        "all_source_and_output_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "automatic_survivors": sorted(
            key for key, value in summaries.items() if value["automatic_survivor"]
        ),
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required" if shortlist else "no_automatic_survivor"
        ),
        "candidates": summaries,
    }


def shortlist_candidates(
    summaries: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[str]:
    by_witness: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for candidate_id, summary in summaries.items():
        if summary["automatic_survivor"]:
            by_witness.setdefault(str(summary["witness_id"]), []).append(
                (candidate_id, summary)
            )
    representatives = []
    for rows in by_witness.values():
        rows.sort(
            key=lambda item: (
                -float(item[1]["gold_median_style_delta_e76"]),
                float(item[1]["strength"]),
                item[0],
            )
        )
        representatives.append(rows[0])
    representatives.sort(
        key=lambda item: (
            -float(item[1]["gold_median_non_basic_residual_delta_e76"]),
            -float(item[1]["gold_median_style_delta_e76"]),
            item[0],
        )
    )
    limit = int(config["shortlist"]["maximum_candidates"])
    return [candidate_id for candidate_id, _ in representatives[:limit]]


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    shortlist: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    if not shortlist:
        return {"rounds": [], "mapping": {}, "reason": "empty shortlist"}
    validated = validate_contract(root, config)
    samples = validated["samples"]
    e1_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    e1_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): manifest_path.parent
        / str(row["output"])
        for row in e1_manifest["records"]
    }
    comparator_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): root
        / str(row["output"])
        for row in validated["comparator_manifest"]["records"]
        if row["candidate_id"] in config["comparators"]
    }
    columns = [*config["comparators"], *shortlist]
    gold_ids = [
        sample_id for sample_id, row in samples.items() if row["split"] == "gold"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    private_mapping: dict[str, dict[str, str]] = {}
    round_paths = []
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(columns)
        random.Random(int(config["render_seed"]) + round_index).shuffle(shuffled)
        labels = {
            candidate: chr(ord("A") + index)
            for index, candidate in enumerate(shuffled)
        }
        private_mapping[f"round_{round_index}"] = {
            label: candidate for candidate, label in labels.items()
        }
        sheet = _contact_sheet(
            root=root,
            samples=samples,
            gold_ids=gold_ids,
            columns=shuffled,
            labels=labels,
            e1_paths=e1_paths,
            comparator_paths=comparator_paths,
        )
        path = output_dir / f"blind_round_{round_index}.png"
        sheet.save(path, format="PNG", compress_level=6)
        round_paths.append(str(path.relative_to(root)).replace("\\", "/"))
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(
        json.dumps(private_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rounds": round_paths,
        "mapping": str(mapping_path.relative_to(root)).replace("\\", "/"),
    }


def _contact_sheet(
    *,
    root: Path,
    samples: Mapping[str, Mapping[str, Any]],
    gold_ids: Sequence[str],
    columns: Sequence[str],
    labels: Mapping[str, str],
    e1_paths: Mapping[tuple[str, str], Path],
    comparator_paths: Mapping[tuple[str, str], Path],
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
        paths = [source]
        for candidate in columns:
            key = (candidate, sample_id)
            path = comparator_paths.get(key, e1_paths.get(key))
            if path is None:
                raise DensityFrontierError(f"missing visual path: {key}")
            paths.append(path)
        names = [f"INPUT {sample_id}", *(labels[value] for value in columns)]
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


__all__ = [
    "DensityFrontierError",
    "build_blind_sheets",
    "candidate_bank",
    "encoded_srgb_to_linear",
    "evaluate_bank",
    "linear_srgb_to_encoded",
    "render_bank",
    "shortlist_candidates",
    "validate_contract",
]
