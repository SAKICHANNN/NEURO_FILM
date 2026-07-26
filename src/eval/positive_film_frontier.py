"""Deterministic real-image frontier for fixed positive-film witnesses."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.eval.density_witness_frontier import (
    _contact_sheet,
    candidate_bank,
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
    shortlist_candidates,
)
from src.eval.global_frontier import (
    load_frozen_samples,
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.positive_film import positive_film_operator_from_config


class PositiveFilmFrontierError(ValueError):
    """Raised when the frozen J1 contract or evidence is invalid."""


def _load_hashed_json(root: Path, path: str, expected_sha256: str) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise PositiveFilmFrontierError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_comparator_output(
    root: Path, manifest_path: Path, output: str
) -> Path:
    relative = Path(output)
    if relative.is_absolute():
        raise PositiveFilmFrontierError("comparator output must be relative")
    candidates = {
        (root / relative).resolve(),
        (manifest_path.parent / relative).resolve(),
    }
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise PositiveFilmFrontierError(
            f"comparator output must resolve exactly once: {output}"
        )
    return existing[0]


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    operator = _load_hashed_json(
        root,
        str(config["operator_config"]),
        str(config["operator_config_sha256"]),
    )
    inherited = _load_hashed_json(
        root,
        str(config["inherited_frontier_config"]),
        str(config["inherited_frontier_config_sha256"]),
    )
    for key, value in config["metrics"].items():
        if inherited["metrics"].get(key) != value:
            raise PositiveFilmFrontierError(f"inherited metric drift: {key}")
    witness_ids = {str(value) for value in config["witness_ids"]}
    if set(operator["witnesses"]) != witness_ids:
        raise PositiveFilmFrontierError("positive-film witness bank drift")
    samples = load_frozen_samples(root, config)
    comparator_records: dict[tuple[str, str], Path] = {}
    declared_comparators = set(str(value) for value in config["comparators"])
    manifest_comparators: set[str] = set()
    for descriptor in config["comparator_manifests"]:
        manifest_path = root / str(descriptor["path"])
        if sha256_file(manifest_path) != str(descriptor["sha256"]):
            raise PositiveFilmFrontierError("comparator manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        allowed = set(str(value) for value in descriptor["candidate_ids"])
        manifest_comparators.update(allowed)
        for row in manifest["records"]:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in allowed:
                continue
            key = (candidate_id, str(row["sample_id"]))
            if key in comparator_records:
                raise PositiveFilmFrontierError(f"duplicate comparator record: {key}")
            resolved = _resolve_comparator_output(
                root, manifest_path, str(row["output"])
            )
            if sha256_file(resolved) != str(row["output_sha256"]):
                raise PositiveFilmFrontierError(f"comparator output hash mismatch: {key}")
            comparator_records[key] = resolved
    if manifest_comparators != declared_comparators:
        raise PositiveFilmFrontierError("comparator declaration drift")
    expected_comparators = {
        (candidate_id, sample_id)
        for candidate_id in declared_comparators
        for sample_id in samples
    }
    if comparator_records.keys() != expected_comparators:
        raise PositiveFilmFrontierError("incomplete comparator manifests")
    return {
        "operator": operator,
        "inherited": inherited,
        "samples": samples,
        "candidates": candidate_bank(config),
        "comparator_paths": comparator_records,
    }


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    operator_config = validated["operator"]
    samples = validated["samples"]
    operators = {
        witness: positive_film_operator_from_config(
            payload,
            exposure_floor=float(operator_config["exposure_floor"]),
            matrix_minimum_determinant=float(
                operator_config["parameter_bounds"]["matrix_minimum_determinant"]
            ),
            minimum_endpoint_span=float(
                operator_config["parameter_bounds"]["minimum_endpoint_span"]
            ),
        )
        for witness, payload in operator_config["witnesses"].items()
    }
    source_arrays: dict[str, np.ndarray] = {}
    for sample_id, sample in samples.items():
        source_path = root / str(sample["source_path"])
        if sha256_file(source_path) != str(sample["source_sha256"]):
            raise PositiveFilmFrontierError(f"source hash mismatch: {sample_id}")
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
            output_linear = operator.apply(
                encoded_srgb_to_linear(source_arrays[sample_id]),
                strength=float(candidate["strength"]),
            )
            output_pixels = np.rint(
                linear_srgb_to_encoded(output_linear) * 255.0
            ).astype(np.uint8)
            path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(output_pixels, mode="RGB").save(
                path, format="PNG", compress_level=6
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
        "operator_config_sha256": config["operator_config_sha256"],
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
        if key not in expected or key in records:
            raise PositiveFilmFrontierError(f"unexpected or duplicate record: {key}")
        records[key] = dict(row)
    if records.keys() != expected:
        raise PositiveFilmFrontierError("incomplete render manifest")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        per_image = []
        for sample_id, sample in samples.items():
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise PositiveFilmFrontierError(
                    f"output hash mismatch: {candidate_id}/{sample_id}"
                )
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(root / str(sample["source_path"])) as source_image:
                    source_size = ImageOps.exif_transpose(source_image).size
                if output.mode != "RGB" or output.size != source_size:
                    raise PositiveFilmFrontierError(
                        f"decode/dimension mismatch: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(root / str(sample["source_path"]), budget)
            output_pixels = sample_rgb_image(output_path, budget)
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
        summary["automatic_gates"] = {
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
        summary["automatic_survivor"] = all(summary["automatic_gates"].values())
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): manifest_path.parent
        / str(row["output"])
        for row in manifest["records"]
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
            e1_paths=candidate_paths,
            comparator_paths=validated["comparator_paths"],
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


__all__ = [
    "PositiveFilmFrontierError",
    "build_blind_sheets",
    "candidate_bank",
    "evaluate_bank",
    "render_bank",
    "shortlist_candidates",
    "validate_contract",
]
