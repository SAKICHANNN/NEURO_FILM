"""Real-image frontier for the factorized AO3 chart-proxy operator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

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


class FactorizedChartFrontierError(ValueError):
    """Raised when AO3 contracts or evidence fail closed."""


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise FactorizedChartFrontierError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("experiment_id")
        not in {
            "u5.r2ao3-factorized-chart-boundary-frontier-v1",
            "u5.r2ao5f-combined-velvia-factorized-frontier-v1",
        }
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or config["factorization"].get("hard_clip_allowed")
    ):
        raise FactorizedChartFrontierError("AO3 frozen boundary drift")
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
    for key, value in config["metrics"].items():
        if inherited["metrics"].get(key) != value:
            raise FactorizedChartFrontierError(f"inherited metric drift: {key}")
    candidates = [dict(value) for value in config["factor_bank"]]
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    if (
        len(candidates) != int(config["candidate_count"])
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise FactorizedChartFrontierError("AO3 candidate bank drift")
    for candidate in candidates:
        if (
            not 0.0 <= float(candidate["tone_strength"]) <= 1.0
            or not 0.0 <= float(candidate["chroma_strength"]) <= 2.0
        ):
            raise FactorizedChartFrontierError("AO3 factor strength out of bounds")
    witness_id = str(config["operator_witness_id"])
    if set(operator_payload["witnesses"]) != {witness_id}:
        raise FactorizedChartFrontierError("AO3 operator witness drift")
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
    return {
        "operator": operator,
        "samples": load_frozen_samples(root, config),
        "candidates": candidates,
    }


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    samples = validated["samples"]
    source_arrays: dict[str, np.ndarray] = {}
    for sample_id, sample in samples.items():
        source_path = root / str(sample["source_path"])
        if sha256_file(source_path) != str(sample["source_sha256"]):
            raise FactorizedChartFrontierError(
                f"source hash mismatch: {sample_id}"
            )
        with Image.open(source_path) as image:
            source_arrays[sample_id] = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )

    controls = config["factorization"]
    records = []
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in samples.items():
            source_encoded = source_arrays[sample_id]
            result = apply_factorized_boundary_guard(
                validated["operator"],
                encoded_srgb_to_linear(source_encoded),
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
            records.append(
                {
                    **candidate,
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
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
        (str(candidate["candidate_id"]), sample_id)
        for candidate in candidates
        for sample_id in samples
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise FactorizedChartFrontierError(
                f"unexpected or duplicate record: {key}"
            )
        records[key] = dict(row)
    if records.keys() != expected:
        raise FactorizedChartFrontierError("incomplete AO3 render manifest")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        per_image = []
        for sample_id, sample in samples.items():
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise FactorizedChartFrontierError(
                    f"output hash mismatch: {candidate_id}/{sample_id}"
                )
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(root / str(sample["source_path"])) as source_image:
                    source_size = ImageOps.exif_transpose(source_image).size
                if output.mode != "RGB" or output.size != source_size:
                    raise FactorizedChartFrontierError(
                        f"decode/dimension mismatch: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(
                root / str(sample["source_path"]), budget
            )
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
        summary["automatic_gates"] = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary[
                "gold_median_non_basic_residual_delta_e76"
            ]
            >= float(
                config["metrics"][
                    "minimum_gold_median_non_basic_residual_delta_e76"
                ]
            ),
            "gold_clipping": summary[
                "worst_gold_new_hard_clipping_fraction"
            ]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_new_hard_clipping_fraction"
                ]
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
        "all_source_and_output_hashes_verified": True,
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
    "FactorizedChartFrontierError",
    "evaluate_bank",
    "render_bank",
    "validate_contract",
]
