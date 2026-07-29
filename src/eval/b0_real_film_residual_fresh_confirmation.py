"""AO7 fresh-population confirmation of one fixed AO6 residual champion."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.dual_champion_composition import (
    build_operators,
    compose_rgb,
    validate_contract as validate_base_contract,
)
from src.eval.global_frontier import (
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


class FreshResidualConfirmationError(ValueError):
    """Raised when AO7 contract, lineage, or evidence drifts."""


SUPPORTED_EXPERIMENTS = {
    "u5.r2ao7-b0-real-film-residual-fresh-confirmation-v1": (0.15, 0.35),
    "u5.r2ap4-b0-ektachrome-residual-fresh-confirmation-v1": (0.10, 0.25),
}


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> Any:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise FreshResidualConfirmationError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise FreshResidualConfirmationError(f"{label} drift")


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all source and operator identities before any render."""

    experiment_id = str(config.get("experiment_id"))
    if (
        experiment_id not in SUPPORTED_EXPERIMENTS
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("production_integration_allowed")
        or config["rendering"].get("operator_refit_allowed")
        or config["rendering"].get("post_result_parameter_change_allowed")
        or config["fixed_candidate"]["factorization"].get(
            "hard_clip_allowed"
        )
    ):
        raise FreshResidualConfirmationError("AO7 frozen boundary drift")

    source_spec = config["source_preflight"]
    source_decision = _load_hashed_json(
        root,
        str(source_spec["decision"]),
        str(source_spec["decision_sha256"]),
    )
    source_manifest = _load_hashed_json(
        root,
        str(source_spec["manifest"]),
        str(source_spec["manifest_sha256"]),
    )
    source_report = _load_hashed_json(
        root,
        str(source_spec["automatic_report"]),
        str(source_spec["automatic_report_sha256"]),
    )
    source_review = _load_hashed_json(
        root,
        str(source_spec["visual_review"]),
        str(source_spec["visual_review_sha256"]),
    )
    if (
        source_decision.get("decision") != source_spec["required_decision"]
        or not source_report.get("automatic_pass")
        or not source_review.get("visual_gate", {}).get("pass")
        or source_review.get("operator_applied") is not False
    ):
        raise FreshResidualConfirmationError("AO7 source gate is not open")

    population = config["confirmation_population"]
    eligible_ids = [str(value) for value in population["eligible_ids"]]
    if (
        len(eligible_ids) != int(population["expected_rows"])
        or len(set(eligible_ids)) != len(eligible_ids)
        or population.get("per_image_exclusion_after_render_allowed")
        is not False
        or population.get("replacement_rows_allowed") is not False
        or source_review.get("eligible_ids") != eligible_ids
        or source_decision["visual_and_selection_evidence"].get(
            "eligible_ids"
        )
        != eligible_ids
    ):
        raise FreshResidualConfirmationError("AO7 population drift")
    if not isinstance(source_manifest, list):
        raise FreshResidualConfirmationError("source manifest schema drift")
    source_rows: dict[str, dict[str, Any]] = {}
    for row in source_manifest:
        sample_id = str(row.get("id"))
        if sample_id in source_rows:
            raise FreshResidualConfirmationError("duplicate source row")
        source_rows[sample_id] = dict(row)
    if not set(eligible_ids).issubset(source_rows):
        raise FreshResidualConfirmationError("eligible source row missing")
    make_counts = Counter(source_rows[key]["make"] for key in eligible_ids)
    if len(make_counts) != int(population["expected_camera_makes"]):
        raise FreshResidualConfirmationError("camera-make count drift")

    base_spec = config["fixed_base"]
    base_config = _load_hashed_json(
        root, str(base_spec["config"]), str(base_spec["config_sha256"])
    )
    base_decision = _load_hashed_json(
        root, str(base_spec["decision"]), str(base_spec["decision_sha256"])
    )
    if (
        base_decision.get("decision") != base_spec["required_decision"]
        or base_decision.get("retained_candidate")
        != base_spec["candidate_id"]
        or base_spec.get("order") != "density_then_anchor"
        or float(base_spec.get("density_strength")) != 0.5
        or int(base_spec.get("final_output_margin")) != 4
    ):
        raise FreshResidualConfirmationError("fixed B0 base drift")
    base_validated = validate_base_contract(root, base_config)

    candidate_spec = config["fixed_candidate"]
    candidate_decision = _load_hashed_json(
        root,
        str(candidate_spec["decision"]),
        str(candidate_spec["decision_sha256"]),
    )
    retained_candidate = candidate_decision.get("retained_candidate")
    if retained_candidate is None:
        retained_candidate = candidate_decision.get(
            "automatic_evidence", {}
        ).get("retained_candidate")
    expected_tone, expected_chroma = SUPPORTED_EXPERIMENTS[experiment_id]
    if (
        candidate_decision.get("decision")
        != candidate_spec["required_decision"]
        or retained_candidate != candidate_spec["candidate_id"]
        or float(candidate_spec["tone_strength"]) != expected_tone
        or float(candidate_spec["chroma_strength"]) != expected_chroma
    ):
        raise FreshResidualConfirmationError("fixed AO6 candidate drift")
    operator_payload = _load_hashed_json(
        root,
        str(candidate_spec["operator_config"]),
        str(candidate_spec["operator_config_sha256"]),
    )
    witness_id = str(candidate_spec["operator_witness_id"])
    if set(operator_payload["witnesses"]) != {witness_id}:
        raise FreshResidualConfirmationError("operator witness drift")
    operator = positive_film_operator_from_config(
        operator_payload["witnesses"][witness_id],
        exposure_floor=float(operator_payload["exposure_floor"]),
        matrix_minimum_determinant=float(
            operator_payload["parameter_bounds"][
                "matrix_minimum_determinant"
            ]
        ),
        minimum_endpoint_span=float(
            operator_payload["parameter_bounds"]["minimum_endpoint_span"]
        ),
    )

    development = _load_hashed_json(
        root,
        str(config["development_reference"]["source"]),
        str(config["development_reference"]["source_sha256"]),
    )
    development_rows = development["candidates"][
        candidate_spec["candidate_id"]
    ]["per_image"]
    reference = config["development_reference"]
    if len(development_rows) != int(reference["sample_count"]):
        raise FreshResidualConfirmationError("development row count drift")
    style = np.asarray(
        [row["median_style_delta_e76"] for row in development_rows]
    )
    residual = np.asarray(
        [
            row["median_non_basic_residual_delta_e76"]
            for row in development_rows
        ]
    )
    film_delta = np.asarray(
        [
            row["median_real_film_delta_e76_from_base"]
            for row in development_rows
        ]
    )
    _require_close(
        float(np.median(style)),
        float(reference["candidate_median_style_delta_e76"]),
        "development median style",
    )
    _require_close(
        float(np.median(residual)),
        float(reference["candidate_median_non_basic_residual_delta_e76"]),
        "development median non-basic",
    )
    _require_close(
        float(np.median(film_delta)),
        float(reference["candidate_median_real_film_delta_e76_from_base"]),
        "development median real-film delta",
    )
    _require_close(
        float(np.percentile(style, 95)),
        float(reference["candidate_p95_style_delta_e76"]),
        "development p95 style",
    )
    _require_close(
        float(np.max(style)),
        float(reference["candidate_maximum_style_delta_e76"]),
        "development maximum style",
    )
    metrics = config["metrics"]
    retention = float(metrics["development_retention_fraction"])
    envelope = float(metrics["development_upper_envelope_multiplier"])
    _require_close(
        float(metrics["minimum_confirmation_median_style_delta_e76"]),
        retention * float(reference["candidate_median_style_delta_e76"]),
        "style retention threshold",
    )
    _require_close(
        float(
            metrics[
                "minimum_confirmation_median_non_basic_residual_delta_e76"
            ]
        ),
        retention
        * float(reference["candidate_median_non_basic_residual_delta_e76"]),
        "non-basic retention threshold",
    )
    _require_close(
        float(metrics["maximum_confirmation_p95_style_delta_e76"]),
        envelope * float(reference["candidate_p95_style_delta_e76"]),
        "p95 style envelope",
    )
    _require_close(
        float(metrics["maximum_confirmation_per_image_style_delta_e76"]),
        envelope * float(reference["candidate_maximum_style_delta_e76"]),
        "maximum style envelope",
    )
    return {
        "eligible_ids": eligible_ids,
        "source_rows": {key: source_rows[key] for key in eligible_ids},
        "base_config": base_config,
        "base_validated": base_validated,
        "operator": operator,
    }


def _save_rgb8_png(path: Path, encoded: np.ndarray) -> str:
    value = np.asarray(encoded, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise FreshResidualConfirmationError(
            "operator output is not finite RGB in [0,1]"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(
        np.rint(value * 255.0).astype(np.uint8), mode="RGB"
    ).save(temporary, format="PNG", compress_level=6)
    temporary.replace(path)
    return sha256_file(path)


def render_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    software_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Render the fixed B0 base and its fixed AO6 residual successor."""

    validated = validate_contract(root, config)
    apply_anchor, apply_density = build_operators(
        validated["base_config"], validated["base_validated"]
    )
    base_spec = config["fixed_base"]
    candidate_spec = config["fixed_candidate"]
    controls = candidate_spec["factorization"]
    candidate_ids = [
        str(base_spec["candidate_id"]),
        str(candidate_spec["candidate_id"]),
    ]
    records: list[dict[str, Any]] = []
    for sample_id in validated["eligible_ids"]:
        row = validated["source_rows"][sample_id]
        raw_path = root / str(row["raw_path"])
        source_path = root / str(row["decoded_path"])
        if (
            sha256_file(raw_path) != row["raw_sha256"]
            or sha256_file(source_path) != row["decoded_sha256"]
        ):
            raise FreshResidualConfirmationError(
                f"source hash mismatch: {sample_id}"
            )
        with Image.open(source_path) as image:
            source = ImageOps.exif_transpose(image)
            if source.mode != "RGB":
                raise FreshResidualConfirmationError(
                    f"source is not RGB: {sample_id}"
                )
            source_float32 = np.asarray(source, dtype=np.float32) / 255.0
        base = compose_rgb(
            source_float32,
            order=str(base_spec["order"]),
            density_strength=float(base_spec["density_strength"]),
            apply_anchor=apply_anchor,
            apply_density=apply_density,
            output_margin=int(base_spec["final_output_margin"]),
        )
        guarded = apply_factorized_boundary_guard(
            validated["operator"],
            encoded_srgb_to_linear(base),
            tone_strength=float(candidate_spec["tone_strength"]),
            chroma_strength=float(candidate_spec["chroma_strength"]),
            luma_weights=np.asarray(controls["luma_weights"]),
            hard_boundary_epsilon_encoded_srgb=float(
                controls["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                controls["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        candidate = linear_srgb_to_encoded(guarded.output)
        outputs = {
            candidate_ids[0]: base,
            candidate_ids[1]: candidate,
        }
        for candidate_id, output in outputs.items():
            relative = Path(candidate_id) / f"{sample_id}.png"
            output_sha = _save_rgb8_png(output_dir / relative, output)
            record = {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "make": row["make"],
                "raw_sha256": row["raw_sha256"],
                "decoded_source_sha256": row["decoded_sha256"],
                "output": relative.as_posix(),
                "output_sha256": output_sha,
            }
            if candidate_id == candidate_ids[1]:
                record.update(
                    {
                        "tone_limited_fraction": float(
                            np.mean(guarded.tone_scale < 1.0 - 1e-12)
                        ),
                        "chroma_limited_fraction": float(
                            np.mean(guarded.chroma_scale < 1.0 - 1e-12)
                        ),
                        "minimum_tone_scale": float(
                            np.min(guarded.tone_scale)
                        ),
                        "minimum_chroma_scale": float(
                            np.min(guarded.chroma_scale)
                        ),
                    }
                )
            records.append(record)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "source_manifest_sha256": config["source_preflight"][
            "manifest_sha256"
        ],
        "source_decision_sha256": config["source_preflight"][
            "decision_sha256"
        ],
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "sample_count": len(validated["eligible_ids"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return {
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(np.clip(first, 0.0, 1.0).reshape(-1, 1, 3))
    second_lab = rgb2lab(np.clip(second, 0.0, 1.0).reshape(-1, 1, 3))
    return float(
        np.median(
            np.linalg.norm(
                second_lab.reshape(-1, 3) - first_lab.reshape(-1, 3),
                axis=1,
            )
        )
    )


def _manifest_records(
    *,
    root: Path,
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    manifest_path: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_ids = [
        str(config["fixed_base"]["candidate_id"]),
        str(config["fixed_candidate"]["candidate_id"]),
    ]
    if (
        manifest.get("experiment_id") != config["experiment_id"]
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("source_manifest_sha256")
        != config["source_preflight"]["manifest_sha256"]
        or manifest.get("source_decision_sha256")
        != config["source_preflight"]["decision_sha256"]
        or manifest.get("candidate_ids") != candidate_ids
        or manifest.get("sample_count") != len(validated["eligible_ids"])
    ):
        raise FreshResidualConfirmationError("manifest identity drift")
    expected = {
        (candidate_id, sample_id)
        for candidate_id in candidate_ids
        for sample_id in validated["eligible_ids"]
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    checked_sources: set[str] = set()
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise FreshResidualConfirmationError(
                f"invalid manifest record: {key}"
            )
        source = validated["source_rows"][key[1]]
        if (
            row.get("make") != source["make"]
            or row.get("raw_sha256") != source["raw_sha256"]
            or row.get("decoded_source_sha256")
            != source["decoded_sha256"]
        ):
            raise FreshResidualConfirmationError(
                "manifest source binding drift"
            )
        if key[1] not in checked_sources:
            if (
                sha256_file(root / str(source["raw_path"]))
                != source["raw_sha256"]
                or sha256_file(root / str(source["decoded_path"]))
                != source["decoded_sha256"]
            ):
                raise FreshResidualConfirmationError("live source hash drift")
            checked_sources.add(key[1])
        expected_output = Path(key[0]) / f"{key[1]}.png"
        relative = Path(str(row.get("output")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != expected_output.as_posix()
            or sha256_file(manifest_path.parent / relative)
            != row.get("output_sha256")
        ):
            raise FreshResidualConfirmationError(
                "manifest output binding drift"
            )
        records[key] = dict(row)
    if set(records) != expected:
        raise FreshResidualConfirmationError("incomplete AO7 manifest")
    return manifest, records


def evaluate_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Evaluate the fixed pair under the preregistered AO7 gates."""

    validated = validate_contract(root, config)
    manifest, records = _manifest_records(
        root=root,
        config=config,
        validated=validated,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    base_id = str(config["fixed_base"]["candidate_id"])
    candidate_id = str(config["fixed_candidate"]["candidate_id"])
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    sampled: dict[tuple[str, str], np.ndarray] = {}
    for output_id in (base_id, candidate_id):
        per_image = []
        for sample_id in validated["eligible_ids"]:
            source = validated["source_rows"][sample_id]
            source_path = root / str(source["decoded_path"])
            output_path = (
                manifest_path.parent
                / str(records[(output_id, sample_id)]["output"])
            )
            with Image.open(source_path) as source_image:
                source_size = ImageOps.exif_transpose(source_image).size
            with Image.open(output_path) as output_image:
                output = ImageOps.exif_transpose(output_image)
                if output.mode != "RGB" or output.size != source_size:
                    raise FreshResidualConfirmationError(
                        "output decode/dimension mismatch"
                    )
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            sampled[(output_id, sample_id)] = output_pixels
            style, residual = style_and_basic_residual(
                source_pixels, output_pixels
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "make": str(source["make"]),
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels, output_pixels, epsilon
                    ),
                    "output_sha256": records[
                        (output_id, sample_id)
                    ]["output_sha256"],
                }
            )
        style_values = np.asarray(
            [row["median_style_delta_e76"] for row in per_image]
        )
        residual_values = np.asarray(
            [
                row["median_non_basic_residual_delta_e76"]
                for row in per_image
            ]
        )
        summaries[output_id] = {
            "median_style_delta_e76": float(np.median(style_values)),
            "median_non_basic_residual_delta_e76": float(
                np.median(residual_values)
            ),
            "p95_style_delta_e76": float(np.percentile(style_values, 95)),
            "maximum_style_delta_e76": float(np.max(style_values)),
            "worst_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in per_image)
            ),
            "per_image": per_image,
        }

    base_rows = {
        row["sample_id"]: row for row in summaries[base_id]["per_image"]
    }
    candidate_rows = {
        row["sample_id"]: row
        for row in summaries[candidate_id]["per_image"]
    }
    paired_rows = []
    for sample_id in validated["eligible_ids"]:
        film_delta = _median_delta_e76(
            sampled[(base_id, sample_id)],
            sampled[(candidate_id, sample_id)],
        )
        paired_rows.append(
            {
                "sample_id": sample_id,
                "make": candidate_rows[sample_id]["make"],
                "median_real_film_delta_e76_from_base": film_delta,
                "candidate_to_base_style_ratio": (
                    candidate_rows[sample_id]["median_style_delta_e76"]
                    / max(
                        base_rows[sample_id]["median_style_delta_e76"],
                        1e-12,
                    )
                ),
                "candidate_to_base_non_basic_ratio": (
                    candidate_rows[sample_id][
                        "median_non_basic_residual_delta_e76"
                    ]
                    / max(
                        base_rows[sample_id][
                            "median_non_basic_residual_delta_e76"
                        ],
                        1e-12,
                    )
                ),
            }
        )
    film_values = np.asarray(
        [row["median_real_film_delta_e76_from_base"] for row in paired_rows]
    )
    make_medians = {
        make: float(
            np.median(
                [
                    row["median_real_film_delta_e76_from_base"]
                    for row in paired_rows
                    if row["make"] == make
                ]
            )
        )
        for make in sorted({row["make"] for row in paired_rows})
    }
    candidate = summaries[candidate_id]
    base = summaries[base_id]
    metrics = config["metrics"]
    style_ratio = candidate["median_style_delta_e76"] / max(
        base["median_style_delta_e76"], 1e-12
    )
    non_basic_ratio = candidate[
        "median_non_basic_residual_delta_e76"
    ] / max(base["median_non_basic_residual_delta_e76"], 1e-12)
    automatic_gates = {
        "style_retention": candidate["median_style_delta_e76"]
        >= float(metrics["minimum_confirmation_median_style_delta_e76"]),
        "non_basic_retention": candidate[
            "median_non_basic_residual_delta_e76"
        ]
        >= float(
            metrics[
                "minimum_confirmation_median_non_basic_residual_delta_e76"
            ]
        ),
        "real_film_delta": float(np.median(film_values))
        >= float(
            metrics[
                "minimum_confirmation_median_real_film_delta_e76_from_base"
            ]
        ),
        "p95_style_envelope": candidate["p95_style_delta_e76"]
        <= float(metrics["maximum_confirmation_p95_style_delta_e76"]),
        "maximum_style_envelope": candidate["maximum_style_delta_e76"]
        <= float(metrics["maximum_confirmation_per_image_style_delta_e76"]),
        "style_ratio_to_base": style_ratio
        >= float(metrics["minimum_candidate_to_base_median_style_ratio"]),
        "non_basic_ratio_to_base": non_basic_ratio
        >= float(
            metrics["minimum_candidate_to_base_median_non_basic_ratio"]
        ),
        "material_image_support": int(
            np.sum(
                film_values
                >= float(metrics["minimum_per_image_real_film_delta_e76"])
            )
        )
        >= int(metrics["minimum_images_above_real_film_delta"]),
        "material_make_support": sum(
            value
            >= float(metrics["minimum_per_image_real_film_delta_e76"])
            for value in make_medians.values()
        )
        >= int(metrics["minimum_camera_makes_above_real_film_delta"]),
        "clipping": all(
            value["worst_new_hard_clipping_fraction"]
            <= float(metrics["maximum_worst_new_hard_clipping_fraction"])
            for value in summaries.values()
        ),
    }
    automatic_pass = all(automatic_gates.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": manifest["software_commit"],
        "config_sha256": config_sha256,
        "render_manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(validated["eligible_ids"]),
        "camera_make_count": len(make_medians),
        "candidate_ids": [base_id, candidate_id],
        "automatic_gates": automatic_gates,
        "automatic_pass": automatic_pass,
        "automatic_decision": (
            "blind_and_full_resolution_review_required"
            if automatic_pass
            else "close_fresh_confirmation_without_rescue"
        ),
        "paired_summary": {
            "median_real_film_delta_e76_from_base": float(
                np.median(film_values)
            ),
            "candidate_to_base_median_style_ratio": style_ratio,
            "candidate_to_base_median_non_basic_ratio": non_basic_ratio,
            "images_above_real_film_delta": int(
                np.sum(
                    film_values
                    >= float(
                        metrics["minimum_per_image_real_film_delta_e76"]
                    )
                )
            ),
            "camera_makes_above_real_film_delta": sum(
                value
                >= float(metrics["minimum_per_image_real_film_delta_e76"])
                for value in make_medians.values()
            ),
            "make_medians": make_medians,
            "per_image": paired_rows,
        },
        "candidates": summaries,
        "claim_ceiling": config["claim_ceiling"],
    }


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    config_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build three source-visible, two-output blind sheets."""

    validated = validate_contract(root, config)
    report = evaluate_confirmation(
        root=root,
        config=config,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    if not report["automatic_pass"]:
        raise FreshResidualConfirmationError(
            "automatic gate forbids blind-sheet generation"
        )
    _, records = _manifest_records(
        root=root,
        config=config,
        validated=validated,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    candidate_ids = report["candidate_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, dict[str, str]] = {}
    paths: list[Path] = []
    hashes: list[str] = []
    tile_width, tile_height, header = 360, 240, 28
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(candidate_ids)
        random.Random(2026072900 + round_index).shuffle(shuffled)
        labels = {
            candidate_id: chr(ord("A") + index)
            for index, candidate_id in enumerate(shuffled)
        }
        mappings[f"round_{round_index}"] = {
            label: candidate_id for candidate_id, label in labels.items()
        }
        sheet = Image.new(
            "RGB",
            (
                tile_width * 3,
                (tile_height + header) * len(validated["eligible_ids"]),
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, sample_id in enumerate(validated["eligible_ids"]):
            source = validated["source_rows"][sample_id]
            row_paths = [
                ("SOURCE", root / str(source["decoded_path"])),
                *[
                    (
                        labels[candidate_id],
                        manifest_path.parent
                        / records[(candidate_id, sample_id)]["output"],
                    )
                    for candidate_id in shuffled
                ],
            ]
            for column_index, (label, path) in enumerate(row_paths):
                with Image.open(path) as opened:
                    tile = ImageOps.contain(
                        ImageOps.exif_transpose(opened).convert("RGB"),
                        (tile_width, tile_height),
                    )
                x = column_index * tile_width + (
                    tile_width - tile.width
                ) // 2
                y = row_index * (tile_height + header) + header
                sheet.paste(tile, (x, y))
                draw.text(
                    (column_index * tile_width + 4, y - header + 5),
                    f"{label} | {sample_id}",
                    fill="black",
                )
        path = output_dir / f"blind_round_{round_index}.png"
        temporary = path.with_suffix(".png.tmp")
        sheet.save(temporary, format="PNG", compress_level=6)
        temporary.replace(path)
        paths.append(path)
        hashes.append(sha256_file(path))
    mapping_path = output_dir / "private_mapping.json"
    mapping_encoded = (
        json.dumps(mappings, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    mapping_path.write_text(mapping_encoded, encoding="utf-8")
    return {
        "rounds": paths,
        "round_sha256": hashes,
        "mapping": mapping_path,
        "mapping_sha256": sha256_file(mapping_path),
    }


__all__ = [
    "FreshResidualConfirmationError",
    "build_blind_sheets",
    "evaluate_confirmation",
    "render_confirmation",
    "validate_contract",
]
