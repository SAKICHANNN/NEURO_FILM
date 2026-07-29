"""AO6 fixed vivid-look plus bounded real-film-residual evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
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
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.roll2film.positive_film import positive_film_operator_from_config


class B0RealFilmResidualFrontierError(ValueError):
    """Raised when AO6 contracts, lineage, or rendered evidence drift."""


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise B0RealFilmResidualFrontierError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _validate_base_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    base = config["base"]
    manifest_path = root / str(base["manifest"])
    manifest = _load_hashed_json(
        root, str(base["manifest"]), str(base["manifest_sha256"])
    )
    if (
        manifest.get("frozen_set_sha256") != config["frozen_set_sha256"]
        or manifest.get("sample_count") != len(samples)
    ):
        raise B0RealFilmResidualFrontierError("AO6 base manifest drift")
    candidate_id = str(base["candidate_id"])
    records: dict[str, dict[str, Any]] = {}
    for row in manifest.get("records", []):
        if row.get("candidate_id") != candidate_id:
            continue
        sample_id = str(row.get("sample_id"))
        if sample_id not in samples or sample_id in records:
            raise B0RealFilmResidualFrontierError(
                f"unexpected or duplicate base record: {sample_id}"
            )
        if row.get("source_sha256") != samples[sample_id]["source_sha256"]:
            raise B0RealFilmResidualFrontierError(
                f"base source lineage mismatch: {sample_id}"
            )
        records[sample_id] = dict(row)
    if records.keys() != samples.keys():
        raise B0RealFilmResidualFrontierError("AO6 base manifest is incomplete")
    return manifest_path, records


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    experiment_id = config.get("experiment_id")
    evaluator_allowed = (
        experiment_id == "u5.r2ao6-b0-real-film-residual-frontier-v1"
        or config.get("evaluator_contract")
        == "neuro-film.b0-factorized-positive-film-residual-frontier.v1"
    )
    if (
        not evaluator_allowed
        or config.get("production_integration_allowed")
        or config.get("operator_refit_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or config["factorization"].get("hard_clip_allowed")
    ):
        raise B0RealFilmResidualFrontierError("AO6 frozen boundary drift")
    operator_payload = _load_hashed_json(
        root,
        str(config["operator_config"]),
        str(config["operator_config_sha256"]),
    )
    inherited = _load_hashed_json(
        root,
        str(config["inherited_frontier_config"]),
        str(config["inherited_frontier_config_sha256"]),
    )
    inherited_metrics = inherited["metrics"]
    for key in (
        "maximum_pixels_per_image",
        "non_basic",
        "new_hard_clipping_epsilon",
        "maximum_worst_gold_new_hard_clipping_fraction",
    ):
        if config["metrics"].get(key) != inherited_metrics.get(key):
            raise B0RealFilmResidualFrontierError(
                f"inherited metric drift: {key}"
            )
    candidates = [dict(value) for value in config["residual_bank"]]
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    if (
        len(candidates) != int(config["candidate_count"])
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise B0RealFilmResidualFrontierError("AO6 candidate bank drift")
    for candidate in candidates:
        if (
            not 0.0 <= float(candidate["tone_strength"]) <= 1.0
            or not 0.0 <= float(candidate["chroma_strength"]) <= 2.0
        ):
            raise B0RealFilmResidualFrontierError(
                "AO6 residual strength out of bounds"
            )
    witness_id = str(config["operator_witness_id"])
    if set(operator_payload["witnesses"]) != {witness_id}:
        raise B0RealFilmResidualFrontierError("AO6 operator witness drift")
    operator = positive_film_operator_from_config(
        operator_payload["witnesses"][witness_id],
        exposure_floor=float(operator_payload["exposure_floor"]),
        matrix_minimum_determinant=float(
            operator_payload["parameter_bounds"]["matrix_minimum_determinant"]
        ),
        minimum_endpoint_span=float(
            operator_payload["parameter_bounds"]["minimum_endpoint_span"]
        ),
    )
    samples = load_frozen_samples(root, config)
    base_manifest_path, base_records = _validate_base_manifest(
        root=root, config=config, samples=samples
    )
    return {
        "operator": operator,
        "samples": samples,
        "candidates": candidates,
        "base_manifest_path": base_manifest_path,
        "base_records": base_records,
    }


def _read_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        if transposed.mode != "RGB":
            raise B0RealFilmResidualFrontierError(
                f"expected exact RGB image: {path}"
            )
        return np.asarray(transposed, dtype=np.float64) / 255.0


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    samples = validated["samples"]
    base_manifest_path = validated["base_manifest_path"]
    base_arrays: dict[str, np.ndarray] = {}
    for sample_id, sample in samples.items():
        source_path = root / str(sample["source_path"])
        if sha256_file(source_path) != str(sample["source_sha256"]):
            raise B0RealFilmResidualFrontierError(
                f"source hash mismatch: {sample_id}"
            )
        base_record = validated["base_records"][sample_id]
        base_path = base_manifest_path.parent / str(base_record["output"])
        if sha256_file(base_path) != str(base_record["output_sha256"]):
            raise B0RealFilmResidualFrontierError(
                f"base output hash mismatch: {sample_id}"
            )
        base = _read_rgb8(base_path)
        with Image.open(source_path) as source_image:
            source_size = ImageOps.exif_transpose(source_image).size
        if (base.shape[1], base.shape[0]) != source_size:
            raise B0RealFilmResidualFrontierError(
                f"base/source dimension mismatch: {sample_id}"
            )
        base_arrays[sample_id] = base

    controls = config["factorization"]
    records = []
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in samples.items():
            result = apply_factorized_boundary_guard(
                validated["operator"],
                encoded_srgb_to_linear(base_arrays[sample_id]),
                tone_strength=float(candidate["tone_strength"]),
                chroma_strength=float(candidate["chroma_strength"]),
                luma_weights=np.asarray(controls["luma_weights"]),
                hard_boundary_epsilon_encoded_srgb=float(
                    controls["hard_boundary_epsilon_encoded_srgb"]
                ),
                guard_boundary_epsilon_encoded_srgb=float(
                    controls["guard_boundary_epsilon_encoded_srgb"]
                ),
            )
            output_pixels = np.rint(
                linear_srgb_to_encoded(result.output) * 255.0
            ).astype(np.uint8)
            path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(output_pixels, mode="RGB").save(
                path, format="PNG", compress_level=6
            )
            base_record = validated["base_records"][sample_id]
            records.append(
                {
                    **candidate,
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "base_output_sha256": base_record["output_sha256"],
                    "output": f"{candidate_id}/{sample_id}.png",
                    "output_sha256": sha256_file(path),
                    "tone_limited_fraction": float(
                        np.mean(result.tone_scale < 1.0 - 1e-12)
                    ),
                    "chroma_limited_fraction": float(
                        np.mean(result.chroma_scale < 1.0 - 1e-12)
                    ),
                    "minimum_tone_scale": float(np.min(result.tone_scale)),
                    "minimum_chroma_scale": float(np.min(result.chroma_scale)),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "base_manifest_sha256": config["base"]["manifest_sha256"],
        "operator_config_sha256": config["operator_config_sha256"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(samples),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(encoded)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(np.clip(first, 0.0, 1.0).reshape(-1, 1, 3))
    second_lab = rgb2lab(np.clip(second, 0.0, 1.0).reshape(-1, 1, 3))
    return float(
        np.median(
            np.linalg.norm(
                second_lab.reshape(-1, 3) - first_lab.reshape(-1, 3), axis=1
            )
        )
    )


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
    if (
        manifest.get("frozen_set_sha256") != config["frozen_set_sha256"]
        or manifest.get("base_manifest_sha256")
        != config["base"]["manifest_sha256"]
        or manifest.get("operator_config_sha256")
        != config["operator_config_sha256"]
    ):
        raise B0RealFilmResidualFrontierError("AO6 render lineage drift")
    expected = {
        (str(candidate["candidate_id"]), sample_id)
        for candidate in candidates
        for sample_id in samples
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise B0RealFilmResidualFrontierError(
                f"unexpected or duplicate render record: {key}"
            )
        records[key] = dict(row)
    if records.keys() != expected:
        raise B0RealFilmResidualFrontierError("incomplete AO6 render manifest")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        per_image = []
        for sample_id, sample in samples.items():
            record = records[(candidate_id, sample_id)]
            base_record = validated["base_records"][sample_id]
            if (
                record.get("source_sha256") != sample["source_sha256"]
                or record.get("base_output_sha256")
                != base_record["output_sha256"]
            ):
                raise B0RealFilmResidualFrontierError(
                    f"render record lineage mismatch: {candidate_id}/{sample_id}"
                )
            output_path = manifest_path.parent / str(record["output"])
            base_path = (
                validated["base_manifest_path"].parent
                / str(base_record["output"])
            )
            if sha256_file(output_path) != record["output_sha256"]:
                raise B0RealFilmResidualFrontierError(
                    f"output hash mismatch: {candidate_id}/{sample_id}"
                )
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(root / str(sample["source_path"])) as source:
                    source_size = ImageOps.exif_transpose(source).size
                if output.mode != "RGB" or output.size != source_size:
                    raise B0RealFilmResidualFrontierError(
                        f"decode/dimension mismatch: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(
                root / str(sample["source_path"]), budget
            )
            base_pixels = sample_rgb_image(base_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels, output_pixels
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "median_real_film_delta_e76_from_base": (
                        _median_delta_e76(base_pixels, output_pixels)
                    ),
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels, output_pixels, epsilon
                    ),
                    "tone_limited_fraction": record["tone_limited_fraction"],
                    "chroma_limited_fraction": record[
                        "chroma_limited_fraction"
                    ],
                    "output_sha256": record["output_sha256"],
                }
            )
        gold = [row for row in per_image if row["split"] == "gold"]
        stress = [row for row in per_image if row["split"] == "stress"]
        summary = {
            **candidate,
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in gold])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [row["median_non_basic_residual_delta_e76"] for row in gold]
                )
            ),
            "gold_median_real_film_delta_e76_from_base": float(
                np.median(
                    [
                        row["median_real_film_delta_e76_from_base"]
                        for row in gold
                    ]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in gold)
            ),
            "worst_stress_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in stress)
            ),
            "maximum_gold_chroma_limited_fraction": float(
                max(row["chroma_limited_fraction"] for row in gold)
            ),
            "per_image": per_image,
        }
        metrics = config["metrics"]
        summary["automatic_gates"] = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= float(metrics["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary[
                "gold_median_non_basic_residual_delta_e76"
            ]
            >= float(
                metrics["minimum_gold_median_non_basic_residual_delta_e76"]
            ),
            "real_film_delta_floor": summary[
                "gold_median_real_film_delta_e76_from_base"
            ]
            >= float(metrics["minimum_gold_median_real_film_delta_e76"]),
            "gold_clipping": summary[
                "worst_gold_new_hard_clipping_fraction"
            ]
            <= float(
                metrics["maximum_worst_gold_new_hard_clipping_fraction"]
            ),
        }
        summary["automatic_survivor"] = all(
            summary["automatic_gates"].values()
        )
        summaries[candidate_id] = summary

    survivors = [
        value for value in summaries.values() if value["automatic_survivor"]
    ]
    survivors.sort(
        key=lambda value: (
            -float(value["gold_median_non_basic_residual_delta_e76"]),
            -float(value["gold_median_real_film_delta_e76_from_base"]),
            -float(value["gold_median_style_delta_e76"]),
            float(value["tone_strength"]),
            float(value["chroma_strength"]),
            str(value["candidate_id"]),
        )
    )
    shortlist = [
        str(value["candidate_id"])
        for value in survivors[
            : int(config["shortlist"]["maximum_candidates"])
        ]
    ]
    return {
        "candidate_count": len(candidates),
        "sample_count": len(samples),
        "gold_sample_count": sum(
            row["split"] == "gold" for row in samples.values()
        ),
        "stress_sample_count": sum(
            row["split"] == "stress" for row in samples.values()
        ),
        "all_source_base_and_output_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "automatic_survivors": sorted(
            key
            for key, value in summaries.items()
            if value["automatic_survivor"]
        ),
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required" if shortlist else "no_automatic_survivor"
        ),
        "candidates": summaries,
    }


__all__ = [
    "B0RealFilmResidualFrontierError",
    "evaluate_bank",
    "render_bank",
    "validate_contract",
]
