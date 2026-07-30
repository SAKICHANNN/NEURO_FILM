"""AZ0 comparison of density and linear-RGB residual factorizations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.b0_real_film_residual_frontier import (
    _median_delta_e76,
    _read_rgb8,
    validate_contract as validate_ao6_contract,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.density_residual_guard import (
    apply_density_residual_guard,
)


class DensityResidualFactorizationError(ValueError):
    """Raised when the AZ0 contract or evidence lineage drifts."""


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise DensityResidualFactorizationError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = config["candidate"]
    if (
        config.get("experiment_id")
        != "u5.r2az0-density-residual-factorization-v1"
        or config.get("hard_clipping_allowed")
        or config.get("per_image_fit_allowed")
        or config.get("parameter_search_allowed")
        or config.get("operator_refit_allowed")
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or float(candidate["neutral_strength"]) != 0.15
        or float(candidate["opponent_strength"]) != 0.35
        or float(candidate["density_floor"]) != 2.0**-16
    ):
        raise DensityResidualFactorizationError("AZ0 frozen boundary drift")
    parent = config["parent"]
    parent_config = _load_hashed_json(
        root, parent["config"], parent["config_sha256"]
    )
    parent_decision = _load_hashed_json(
        root, parent["decision"], parent["decision_sha256"]
    )
    parent_manifest = _load_hashed_json(
        root, parent["manifest"], parent["manifest_sha256"]
    )
    if (
        parent_decision["automatic_evidence"]["retained_candidate"]
        != parent["candidate_id"]
        or parent_decision["decision"]
        != "retain_b0_plus_film_t15_c35_as_B0_development_champion_open_fresh_OOD_confirmation"
    ):
        raise DensityResidualFactorizationError("AO6 parent decision drift")
    validated = validate_ao6_contract(root, parent_config)
    comparator_records: dict[str, dict[str, Any]] = {}
    for row in parent_manifest["records"]:
        if row["candidate_id"] != parent["candidate_id"]:
            continue
        sample_id = str(row["sample_id"])
        if sample_id in comparator_records:
            raise DensityResidualFactorizationError(
                f"duplicate AO6 comparator row: {sample_id}"
            )
        comparator_records[sample_id] = dict(row)
    if comparator_records.keys() != validated["samples"].keys():
        raise DensityResidualFactorizationError(
            "AO6 comparator population drift"
        )
    return {
        **validated,
        "parent_config": parent_config,
        "parent_decision": parent_decision,
        "parent_manifest_path": root / parent["manifest"],
        "comparator_records": comparator_records,
    }


def _sample_array(values: np.ndarray, maximum_pixels: int) -> np.ndarray:
    height, width = values.shape[:2]
    rows = min(
        height,
        max(
            1,
            int(np.floor(np.sqrt(maximum_pixels * height / width))),
        ),
    )
    columns = min(width, max(1, maximum_pixels // rows))
    yy, xx = np.meshgrid(
        np.linspace(0, height - 1, rows, dtype=np.int64),
        np.linspace(0, width - 1, columns, dtype=np.int64),
        indexing="ij",
    )
    return values[yy, xx].reshape(-1, 3)


def render_and_evaluate(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = config["candidate"]
    budget = int(config["maximum_pixels_per_image"])
    epsilon = float(
        candidate["hard_boundary_epsilon_encoded_srgb"]
    )
    records: list[dict[str, Any]] = []
    for sample_id, sample in validated["samples"].items():
        source_path = root / sample["source_path"]
        if sha256_file(source_path) != sample["source_sha256"]:
            raise DensityResidualFactorizationError(
                f"source hash mismatch: {sample_id}"
            )
        base_record = validated["base_records"][sample_id]
        base_path = (
            validated["base_manifest_path"].parent / base_record["output"]
        )
        comparator_record = validated["comparator_records"][sample_id]
        comparator_path = (
            validated["parent_manifest_path"].parent
            / comparator_record["output"]
        )
        if (
            sha256_file(base_path) != base_record["output_sha256"]
            or sha256_file(comparator_path)
            != comparator_record["output_sha256"]
        ):
            raise DensityResidualFactorizationError(
                f"parent image hash mismatch: {sample_id}"
            )
        base = _read_rgb8(base_path)
        result = apply_density_residual_guard(
            validated["operator"],
            encoded_srgb_to_linear(base),
            neutral_strength=float(candidate["neutral_strength"]),
            opponent_strength=float(candidate["opponent_strength"]),
            neutral_weights=np.asarray(candidate["neutral_weights"]),
            density_floor=float(candidate["density_floor"]),
            hard_boundary_epsilon_encoded_srgb=float(
                candidate["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                candidate["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        encoded = linear_srgb_to_encoded(result.output)
        if np.any(encoded < 0.0) or np.any(encoded > 1.0):
            raise DensityResidualFactorizationError(
                f"unclipped output escaped encoded cube: {sample_id}"
            )
        output_pixels = np.rint(encoded * 255.0).astype(np.uint8)
        output_path = output_dir / f"{sample_id}.png"
        Image.fromarray(output_pixels, mode="RGB").save(
            output_path, format="PNG", compress_level=6
        )

        source_pixels = sample_rgb_image(source_path, budget)
        base_pixels = sample_rgb_image(base_path, budget)
        comparator_pixels = sample_rgb_image(comparator_path, budget)
        candidate_pixels = (
            _sample_array(output_pixels, budget).astype(np.float64) / 255.0
        )
        style, non_basic = style_and_basic_residual(
            source_pixels, candidate_pixels
        )
        records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "sample_id": sample_id,
                "split": sample["split"],
                "source_sha256": sample["source_sha256"],
                "base_output_sha256": base_record["output_sha256"],
                "comparator_output_sha256": comparator_record[
                    "output_sha256"
                ],
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "median_style_delta_e76": style,
                "median_non_basic_residual_delta_e76": non_basic,
                "median_real_film_delta_e76_from_base": _median_delta_e76(
                    base_pixels, candidate_pixels
                ),
                "median_difference_delta_e76_from_ao6": _median_delta_e76(
                    comparator_pixels, candidate_pixels
                ),
                "new_hard_clipping_fraction": new_hard_clipping_fraction(
                    source_pixels, candidate_pixels, epsilon
                ),
                "neutral_limited_fraction": float(
                    np.mean(result.neutral_scale < 1.0 - 1e-12)
                ),
                "opponent_limited_fraction": float(
                    np.mean(result.opponent_scale < 1.0 - 1e-12)
                ),
            }
        )

    gold = [row for row in records if row["split"] == "gold"]
    stress = [row for row in records if row["split"] == "stress"]
    parent_metrics = validated["parent_decision"]["automatic_evidence"]
    summary = {
        "gold_median_style_delta_e76": float(
            np.median([row["median_style_delta_e76"] for row in gold])
        ),
        "gold_median_non_basic_residual_delta_e76": float(
            np.median(
                [
                    row["median_non_basic_residual_delta_e76"]
                    for row in gold
                ]
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
        "gold_median_difference_delta_e76_from_ao6": float(
            np.median(
                [
                    row["median_difference_delta_e76_from_ao6"]
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
    }
    summary["gold_median_style_gain_delta_e76_over_ao6"] = (
        summary["gold_median_style_delta_e76"]
        - float(parent_metrics["gold_median_style_delta_e76"])
    )
    summary["gold_median_non_basic_gain_delta_e76_over_ao6"] = (
        summary["gold_median_non_basic_residual_delta_e76"]
        - float(
            parent_metrics["gold_median_non_basic_residual_delta_e76"]
        )
    )
    summary[
        "gold_median_real_film_delta_gain_delta_e76_over_ao6"
    ] = (
        summary["gold_median_real_film_delta_e76_from_base"]
        - float(
            parent_metrics["gold_median_real_film_delta_e76_from_base"]
        )
    )
    gates = config["frozen_gates"]
    automatic_gates = {
        "style_gain": summary[
            "gold_median_style_gain_delta_e76_over_ao6"
        ]
        >= float(
            gates[
                "minimum_gold_median_style_gain_delta_e76_over_ao6"
            ]
        ),
        "non_basic_gain": summary[
            "gold_median_non_basic_gain_delta_e76_over_ao6"
        ]
        >= float(
            gates[
                "minimum_gold_median_non_basic_gain_delta_e76_over_ao6"
            ]
        ),
        "real_film_delta_gain": summary[
            "gold_median_real_film_delta_gain_delta_e76_over_ao6"
        ]
        >= float(
            gates[
                "minimum_gold_median_real_film_delta_gain_delta_e76_over_ao6"
            ]
        ),
        "candidate_difference": summary[
            "gold_median_difference_delta_e76_from_ao6"
        ]
        >= float(
            gates["minimum_gold_median_candidate_difference_delta_e76"]
        ),
        "gold_clipping": summary[
            "worst_gold_new_hard_clipping_fraction"
        ]
        <= float(
            gates["maximum_worst_gold_new_hard_clipping_fraction"]
        ),
        "stress_clipping": summary[
            "worst_stress_new_hard_clipping_fraction"
        ]
        <= float(
            gates["maximum_worst_stress_new_hard_clipping_fraction"]
        ),
    }
    stable_payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "parent_manifest_sha256": config["parent"]["manifest_sha256"],
        "candidate": candidate,
        "summary": summary,
        "automatic_gates": automatic_gates,
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_bytes = json.dumps(
        stable_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        **stable_payload,
        "automatic_pass": all(automatic_gates.values()),
        "next": (
            "fixed_full_resolution_blind_review"
            if all(automatic_gates.values())
            else "close_without_visual_review"
        ),
        "stable_evidence_id": hashlib.sha256(stable_bytes).hexdigest(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DensityResidualFactorizationError",
    "render_and_evaluate",
    "validate_contract",
]
