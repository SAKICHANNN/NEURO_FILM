"""BA0 comparison of perceptual, density, and linear-RGB factorizations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.b0_density_residual_factorization import (
    _sample_array,
    validate_contract as validate_az0_contract,
)
from src.eval.b0_real_film_residual_frontier import (
    _median_delta_e76,
    _read_rgb8,
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
from src.roll2film.perceptual_residual_guard import (
    apply_perceptual_residual_guard,
)


class PerceptualResidualFactorizationError(ValueError):
    """Raised when the BA0 contract or evidence lineage drifts."""


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise PerceptualResidualFactorizationError(
            f"hash mismatch: {path}"
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = config["candidate"]
    if (
        config.get("experiment_id")
        != "u5.r2ba0-perceptual-residual-factorization-v1"
        or config.get("status")
        != "contract_frozen_before_candidate_render"
        or config.get("hard_clipping_allowed")
        or config.get("per_image_fit_allowed")
        or config.get("parameter_search_allowed")
        or config.get("operator_refit_allowed")
        or config.get("strength_retuning_allowed")
        or config.get("training_allowed")
        or config.get("production_integration_allowed")
        or float(candidate["lightness_strength"]) != 0.15
        or float(candidate["chroma_strength"]) != 0.35
        or candidate.get("working_space") != "linear_srgb"
        or candidate.get("perceptual_space") != "D65_CIELAB"
    ):
        raise PerceptualResidualFactorizationError(
            "BA0 frozen boundary drift"
        )

    density = config["density_comparator"]
    density_config = _load_hashed_json(
        root, density["config"], density["config_sha256"]
    )
    density_decision = _load_hashed_json(
        root, density["decision"], density["decision_sha256"]
    )
    density_report = _load_hashed_json(
        root, density["report"], density["report_sha256"]
    )
    if (
        density_decision.get("decision")
        != "retain_density_factorization_as_development_look_approximation_champion"
        or density_report.get("candidate", {}).get("candidate_id")
        != density["candidate_id"]
        or not density_report.get("automatic_pass")
    ):
        raise PerceptualResidualFactorizationError(
            "AZ0 density comparator drift"
        )

    validated = validate_az0_contract(root, density_config)
    parent = config["parent"]
    if (
        parent["config_sha256"]
        != density_config["parent"]["config_sha256"]
        or parent["decision_sha256"]
        != density_config["parent"]["decision_sha256"]
        or parent["manifest_sha256"]
        != density_config["parent"]["manifest_sha256"]
        or parent["candidate_id"]
        != density_config["parent"]["candidate_id"]
    ):
        raise PerceptualResidualFactorizationError(
            "BA0/AZ0 parent lineage mismatch"
        )

    density_records: dict[str, dict[str, Any]] = {}
    density_report_path = root / density["report"]
    for row in density_report["records"]:
        sample_id = str(row["sample_id"])
        if sample_id in density_records:
            raise PerceptualResidualFactorizationError(
                "duplicate AZ0 density record"
            )
        path = density_report_path.parent / row["output"]
        if sha256_file(path) != row["output_sha256"]:
            raise PerceptualResidualFactorizationError(
                f"AZ0 output identity drift: {sample_id}"
            )
        density_records[sample_id] = {**row, "absolute_path": path}
    if density_records.keys() != validated["samples"].keys():
        raise PerceptualResidualFactorizationError(
            "AZ0 density population drift"
        )
    return {
        **validated,
        "density_report": density_report,
        "density_records": density_records,
    }


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
            raise PerceptualResidualFactorizationError(
                f"source hash mismatch: {sample_id}"
            )
        base_record = validated["base_records"][sample_id]
        base_path = (
            validated["base_manifest_path"].parent / base_record["output"]
        )
        ao6_record = validated["comparator_records"][sample_id]
        ao6_path = (
            validated["parent_manifest_path"].parent / ao6_record["output"]
        )
        density_record = validated["density_records"][sample_id]
        density_path = density_record["absolute_path"]
        if (
            sha256_file(base_path) != base_record["output_sha256"]
            or sha256_file(ao6_path) != ao6_record["output_sha256"]
        ):
            raise PerceptualResidualFactorizationError(
                f"parent image hash mismatch: {sample_id}"
            )

        base = _read_rgb8(base_path)
        result = apply_perceptual_residual_guard(
            validated["operator"],
            encoded_srgb_to_linear(base),
            lightness_strength=float(candidate["lightness_strength"]),
            chroma_strength=float(candidate["chroma_strength"]),
            hard_boundary_epsilon_encoded_srgb=epsilon,
            guard_boundary_epsilon_encoded_srgb=float(
                candidate["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        encoded = linear_srgb_to_encoded(result.output)
        if np.any(encoded < 0.0) or np.any(encoded > 1.0):
            raise PerceptualResidualFactorizationError(
                f"candidate escaped encoded cube: {sample_id}"
            )
        output_pixels = np.rint(encoded * 255.0).astype(np.uint8)
        output_path = output_dir / f"{sample_id}.png"
        Image.fromarray(output_pixels, mode="RGB").save(
            output_path, format="PNG", compress_level=6
        )

        source_pixels = sample_rgb_image(source_path, budget)
        base_pixels = sample_rgb_image(base_path, budget)
        ao6_pixels = sample_rgb_image(ao6_path, budget)
        density_pixels = sample_rgb_image(density_path, budget)
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
                "ao6_output_sha256": ao6_record["output_sha256"],
                "density_output_sha256": density_record["output_sha256"],
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "median_style_delta_e76": style,
                "median_non_basic_residual_delta_e76": non_basic,
                "median_real_film_delta_e76_from_base": _median_delta_e76(
                    base_pixels, candidate_pixels
                ),
                "median_difference_delta_e76_from_ao6": _median_delta_e76(
                    ao6_pixels, candidate_pixels
                ),
                "median_difference_delta_e76_from_density": (
                    _median_delta_e76(density_pixels, candidate_pixels)
                ),
                "new_hard_clipping_fraction": new_hard_clipping_fraction(
                    source_pixels, candidate_pixels, epsilon
                ),
                "lightness_limited_fraction": float(
                    np.mean(result.lightness_scale < 1.0 - 1e-12)
                ),
                "chroma_limited_fraction": float(
                    np.mean(result.chroma_scale < 1.0 - 1e-12)
                ),
            }
        )

    gold = [row for row in records if row["split"] == "gold"]
    stress = [row for row in records if row["split"] == "stress"]
    parent_metrics = validated["parent_decision"]["automatic_evidence"]
    density_summary = validated["density_report"]["summary"]

    def median(key: str) -> float:
        return float(np.median([row[key] for row in gold]))

    summary = {
        "gold_median_style_delta_e76": median(
            "median_style_delta_e76"
        ),
        "gold_median_non_basic_residual_delta_e76": median(
            "median_non_basic_residual_delta_e76"
        ),
        "gold_median_real_film_delta_e76_from_base": median(
            "median_real_film_delta_e76_from_base"
        ),
        "gold_median_difference_delta_e76_from_ao6": median(
            "median_difference_delta_e76_from_ao6"
        ),
        "gold_median_difference_delta_e76_from_density": median(
            "median_difference_delta_e76_from_density"
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
    summary["gold_median_style_ratio_to_density"] = (
        summary["gold_median_style_delta_e76"]
        / float(density_summary["gold_median_style_delta_e76"])
    )
    gates = config["frozen_gates"]
    decisions = {
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
        "difference_from_ao6": summary[
            "gold_median_difference_delta_e76_from_ao6"
        ]
        >= float(
            gates[
                "minimum_gold_median_candidate_difference_delta_e76_from_ao6"
            ]
        ),
        "difference_from_density": summary[
            "gold_median_difference_delta_e76_from_density"
        ]
        >= float(
            gates[
                "minimum_gold_median_candidate_difference_delta_e76_from_density"
            ]
        ),
        "density_style_floor": summary[
            "gold_median_style_ratio_to_density"
        ]
        >= float(gates["minimum_gold_median_style_ratio_to_density"]),
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
        "schema": "neuro-film.u5-r2ba0-perceptual-residual-report.v1",
        "experiment_id": config["experiment_id"],
        "candidate": candidate,
        "summary": summary,
        "automatic_decisions": decisions,
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_bytes = json.dumps(
        stable_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        **stable_payload,
        "automatic_pass": all(decisions.values()),
        "branch": (
            config["branch_rules"]["automatic_pass"]
            if all(decisions.values())
            else config["branch_rules"]["automatic_fail"]
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
    "PerceptualResidualFactorizationError",
    "render_and_evaluate",
    "validate_contract",
]
