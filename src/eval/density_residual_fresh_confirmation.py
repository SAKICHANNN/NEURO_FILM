"""AZ1 fresh-population confirmation of the fixed AZ0 density operator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.b0_density_residual_factorization import _sample_array
from src.eval.b0_real_film_residual_fresh_confirmation import (
    _median_delta_e76,
    validate_contract as validate_ao7_contract,
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


class DensityResidualFreshError(ValueError):
    """Raised when an AZ1 contract or evidence identity drifts."""


def _load_exact(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise DensityResidualFreshError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = config["candidate"]
    if (
        config.get("experiment_id")
        != "u5.r2az1-density-residual-fresh-confirmation-v1"
        or config.get("status")
        != "contract_frozen_before_candidate_render"
        or config.get("operator_refit_allowed")
        or config.get("strength_retuning_allowed")
        or config.get("row_replacement_allowed")
        or config.get("hard_clipping_allowed")
        or config.get("training_allowed")
        or config.get("production_integration_allowed")
        or float(candidate["neutral_strength"]) != 0.15
        or float(candidate["opponent_strength"]) != 0.35
        or float(candidate["density_floor"]) != 2.0**-16
    ):
        raise DensityResidualFreshError("AZ1 frozen boundary drift")

    development = _load_exact(
        root,
        config["development_decision"]["path"],
        config["development_decision"]["sha256"],
    )
    if (
        development.get("decision")
        != config["development_decision"]["required_decision"]
    ):
        raise DensityResidualFreshError("AZ0 development decision drift")
    population = config["fresh_population"]
    ao7_config = _load_exact(
        root, population["ao7_config"], population["ao7_config_sha256"]
    )
    ao7_decision = _load_exact(
        root,
        population["ao7_decision"],
        population["ao7_decision_sha256"],
    )
    ao7_manifest = _load_exact(
        root,
        population["ao7_manifest"],
        population["ao7_manifest_sha256"],
    )
    ao7_report = _load_exact(
        root,
        population["ao7_automatic_report"],
        population["ao7_automatic_report_sha256"],
    )
    if (
        ao7_report.get("automatic_pass")
        is not population["required_ao7_automatic_pass"]
        or ao7_decision.get("automatic_evidence", {}).get("all_gates_pass")
        is not True
    ):
        raise DensityResidualFreshError("fresh population gate drift")
    validated = validate_ao7_contract(root, ao7_config)
    if (
        len(validated["eligible_ids"])
        != int(population["required_sample_count"])
        or len(
            {validated["source_rows"][key]["make"] for key in validated[
                "eligible_ids"
            ]}
        )
        != int(population["required_camera_make_count"])
    ):
        raise DensityResidualFreshError("fresh population support drift")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_ids = {
        candidate["base_id"],
        candidate["comparator_id"],
    }
    for row in ao7_manifest["records"]:
        output_id = str(row["candidate_id"])
        sample_id = str(row["sample_id"])
        if output_id not in allowed_ids:
            continue
        key = (output_id, sample_id)
        if key in records:
            raise DensityResidualFreshError("duplicate AO7 render record")
        output_path = (
            root
            / population["ao7_manifest"]
        ).parent / str(row["output"])
        if sha256_file(output_path) != row["output_sha256"]:
            raise DensityResidualFreshError("AO7 render identity drift")
        records[key] = {**row, "absolute_path": output_path}
    expected = {
        (output_id, sample_id)
        for output_id in allowed_ids
        for sample_id in validated["eligible_ids"]
    }
    if set(records) != expected:
        raise DensityResidualFreshError("incomplete AO7 render inventory")

    az0_report = _load_exact(
        root,
        development["automatic_report"]["path"],
        development["automatic_report"]["sha256"],
    )
    development_hashes = {row["source_sha256"] for row in az0_report["records"]}
    fresh_hashes = {
        validated["source_rows"][key]["decoded_sha256"]
        for key in validated["eligible_ids"]
    }
    overlap = development_hashes & fresh_hashes
    if (
        population[
            "required_zero_decoded_sha256_overlap_with_az0_development"
        ]
        and overlap
    ):
        raise DensityResidualFreshError("development/fresh pixel overlap")
    return {
        "eligible_ids": validated["eligible_ids"],
        "source_rows": validated["source_rows"],
        "operator": validated["operator"],
        "records": records,
        "development_hash_count": len(development_hashes),
        "fresh_hash_count": len(fresh_hashes),
        "overlap_count": len(overlap),
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
    budget = 65536
    epsilon = float(candidate["hard_boundary_epsilon_encoded_srgb"])
    rows = []
    for sample_id in validated["eligible_ids"]:
        source = validated["source_rows"][sample_id]
        source_path = root / str(source["decoded_path"])
        if sha256_file(source_path) != source["decoded_sha256"]:
            raise DensityResidualFreshError("live source identity drift")
        base_record = validated["records"][
            (candidate["base_id"], sample_id)
        ]
        comparator_record = validated["records"][
            (candidate["comparator_id"], sample_id)
        ]
        with Image.open(base_record["absolute_path"]) as image:
            base_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result = apply_density_residual_guard(
            validated["operator"],
            encoded_srgb_to_linear(
                base_u8.astype(np.float64) / 255.0
            ),
            neutral_strength=float(candidate["neutral_strength"]),
            opponent_strength=float(candidate["opponent_strength"]),
            neutral_weights=np.asarray(candidate["neutral_weights"]),
            density_floor=float(candidate["density_floor"]),
            hard_boundary_epsilon_encoded_srgb=epsilon,
            guard_boundary_epsilon_encoded_srgb=float(
                candidate["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        encoded = linear_srgb_to_encoded(result.output)
        if np.any(encoded < 0.0) or np.any(encoded > 1.0):
            raise DensityResidualFreshError("candidate escaped encoded cube")
        output_u8 = np.rint(encoded * 255.0).astype(np.uint8)
        output_path = output_dir / f"{sample_id}.png"
        Image.fromarray(output_u8, mode="RGB").save(
            output_path, "PNG", compress_level=6
        )
        source_pixels = sample_rgb_image(source_path, budget)
        comparator_pixels = sample_rgb_image(
            comparator_record["absolute_path"], budget
        )
        candidate_pixels = (
            _sample_array(output_u8, budget).astype(np.float64) / 255.0
        )
        style, non_basic = style_and_basic_residual(
            source_pixels, candidate_pixels
        )
        rows.append(
            {
                "sample_id": sample_id,
                "make": source["make"],
                "source_sha256": source["decoded_sha256"],
                "base_output_sha256": base_record["output_sha256"],
                "comparator_output_sha256": comparator_record[
                    "output_sha256"
                ],
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "median_style_delta_e76": style,
                "median_non_basic_residual_delta_e76": non_basic,
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

    style = np.asarray([row["median_style_delta_e76"] for row in rows])
    non_basic = np.asarray(
        [row["median_non_basic_residual_delta_e76"] for row in rows]
    )
    difference = np.asarray(
        [row["median_difference_delta_e76_from_ao6"] for row in rows]
    )
    threshold = float(
        config["automatic_gates"][
            "minimum_per_image_difference_delta_e76_from_ao6"
        ]
    )
    make_medians = {
        make: float(
            np.median(
                [
                    row["median_difference_delta_e76_from_ao6"]
                    for row in rows
                    if row["make"] == make
                ]
            )
        )
        for make in sorted({row["make"] for row in rows})
    }
    summary = {
        "candidate_median_style_delta_e76": float(np.median(style)),
        "candidate_median_non_basic_residual_delta_e76": float(
            np.median(non_basic)
        ),
        "candidate_p95_style_delta_e76": float(np.percentile(style, 95)),
        "candidate_maximum_style_delta_e76": float(np.max(style)),
        "median_difference_delta_e76_from_ao6": float(
            np.median(difference)
        ),
        "images_above_difference": int(np.sum(difference >= threshold)),
        "camera_makes_above_difference": sum(
            value >= threshold for value in make_medians.values()
        ),
        "make_median_differences": make_medians,
        "worst_new_hard_clipping_fraction": float(
            max(row["new_hard_clipping_fraction"] for row in rows)
        ),
    }
    gates = config["automatic_gates"]
    decisions = {
        "style": summary["candidate_median_style_delta_e76"]
        >= float(gates["minimum_candidate_median_style_delta_e76"]),
        "non_basic": summary[
            "candidate_median_non_basic_residual_delta_e76"
        ]
        >= float(
            gates[
                "minimum_candidate_median_non_basic_residual_delta_e76"
            ]
        ),
        "p95_envelope": summary["candidate_p95_style_delta_e76"]
        <= float(gates["maximum_candidate_p95_style_delta_e76"]),
        "maximum_envelope": summary["candidate_maximum_style_delta_e76"]
        <= float(gates["maximum_candidate_per_image_style_delta_e76"]),
        "median_difference": summary[
            "median_difference_delta_e76_from_ao6"
        ]
        >= float(gates["minimum_median_difference_delta_e76_from_ao6"]),
        "image_support": summary["images_above_difference"]
        >= int(gates["minimum_images_above_difference"]),
        "make_support": summary["camera_makes_above_difference"]
        >= int(gates["minimum_camera_makes_above_difference"]),
        "clipping": summary["worst_new_hard_clipping_fraction"]
        <= float(gates["maximum_worst_new_hard_clipping_fraction"]),
        "zero_development_overlap": validated["overlap_count"] == 0,
    }
    core = {
        "schema": "neuro-film.u5-r2az1-density-residual-fresh-report.v1",
        "experiment_id": config["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "comparator_id": candidate["comparator_id"],
        "sample_count": len(rows),
        "camera_make_count": len(Counter(row["make"] for row in rows)),
        "development_source_hash_count": validated[
            "development_hash_count"
        ],
        "fresh_source_hash_count": validated["fresh_hash_count"],
        "development_fresh_overlap_count": validated["overlap_count"],
        "summary": summary,
        "automatic_decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": (
            "build_frozen_blind_and_full_resolution_review"
            if all(decisions.values())
            else config["branch_rules"]["automatic_fail"]
        ),
        "records": rows,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_json(payload: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["render_and_evaluate", "validate_contract", "write_json"]
