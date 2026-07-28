"""Frozen U5.R2AI1 independent confirmation of one global composition."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.pipeline_color_baseline import apply_output_margin
from src.eval.dual_champion_composition import (
    build_operators,
    compose_rgb,
    validate_contract as validate_parent_contract,
)
from src.eval.global_frontier import (
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


class IndependentConfirmationError(ValueError):
    """Raised when the frozen confirmation contract or evidence drifts."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(root: Path, path: str, expected: str) -> Path:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise IndependentConfirmationError(f"hash mismatch: {path}")
    return resolved


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise IndependentConfirmationError(f"{label} drift")


def _candidate_ids(config: Mapping[str, Any]) -> list[str]:
    operators = config["fixed_operators"]
    ids = [
        str(row["candidate_id"])
        for row in operators["comparators"]
    ]
    ids.append(str(operators["candidate"]["candidate_id"]))
    if len(ids) != 3 or len(set(ids)) != 3:
        raise IndependentConfirmationError("fixed candidate identities drift")
    return ids


def validate_contract(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every parent/source identity before rendering any output."""

    if config.get("status") != "contract_frozen_implementation_ready":
        raise IndependentConfirmationError("confirmation contract is not frozen")

    parent_spec = config["parent_algorithm"]
    source_spec = config["source_preflight"]
    parent_config = _load_json(
        _require_hash(
            root,
            str(parent_spec["config"]),
            str(parent_spec["config_sha256"]),
        )
    )
    parent_decision = _load_json(
        _require_hash(
            root,
            str(parent_spec["decision"]),
            str(parent_spec["decision_sha256"]),
        )
    )
    source_contract = _load_json(
        _require_hash(
            root,
            str(source_spec["contract"]),
            str(source_spec["contract_sha256"]),
        )
    )
    source_decision = _load_json(
        _require_hash(
            root,
            str(source_spec["decision"]),
            str(source_spec["decision_sha256"]),
        )
    )
    source_manifest = _load_json(
        _require_hash(
            root,
            str(source_spec["manifest"]),
            str(source_spec["manifest_sha256"]),
        )
    )
    source_report = _load_json(
        _require_hash(
            root,
            str(source_spec["automatic_report"]),
            str(source_spec["automatic_report_sha256"]),
        )
    )
    source_review = _load_json(
        _require_hash(
            root,
            str(source_spec["visual_review"]),
            str(source_spec["visual_review_sha256"]),
        )
    )
    development_report = _load_json(
        _require_hash(
            root,
            str(config["development_reference"]["source"]),
            str(config["development_reference"]["source_sha256"]),
        )
    )

    if parent_decision.get("decision") != parent_spec["required_decision"]:
        raise IndependentConfirmationError("parent decision is not retained")
    if (
        parent_decision.get("retained_candidate")
        != parent_spec["retained_candidate"]
    ):
        raise IndependentConfirmationError("retained parent candidate drift")
    if source_decision.get("decision") != source_spec["required_decision"]:
        raise IndependentConfirmationError("source preflight did not pass")
    if not source_report.get("automatic_pass"):
        raise IndependentConfirmationError("source automatic gate did not pass")
    if not source_review.get("visual_gate", {}).get("pass"):
        raise IndependentConfirmationError("source visual gate did not pass")
    if source_review.get("operator_applied") is not False:
        raise IndependentConfirmationError("source preflight applied an operator")
    if source_contract.get("operator_fitting_allowed") is not False:
        raise IndependentConfirmationError("source fitting boundary drift")

    population = config["confirmation_population"]
    eligible_ids = [str(value) for value in population["eligible_ids"]]
    if (
        len(eligible_ids) != int(population["expected_rows"])
        or len(set(eligible_ids)) != len(eligible_ids)
    ):
        raise IndependentConfirmationError("eligible population drift")
    excluded = {str(value) for value in source_spec["excluded_ids"]}
    if excluded.intersection(eligible_ids):
        raise IndependentConfirmationError("excluded row re-entered confirmation")
    if source_spec.get("replacement_rows_allowed") is not False:
        raise IndependentConfirmationError("replacement boundary drift")
    if population.get("per_image_exclusion_after_render_allowed") is not False:
        raise IndependentConfirmationError("post-render exclusion boundary drift")
    if source_decision["visual_and_selection_evidence"]["eligible_ids"] != eligible_ids:
        raise IndependentConfirmationError("source decision population drift")
    if source_review["eligible_ids"] != eligible_ids:
        raise IndependentConfirmationError("source review population drift")
    if set(source_review["excluded_ids"]) != excluded:
        raise IndependentConfirmationError("source exclusion drift")

    if not isinstance(source_manifest, list):
        raise IndependentConfirmationError("source manifest schema drift")
    source_rows: dict[str, dict[str, Any]] = {}
    for row in source_manifest:
        if not isinstance(row, dict):
            raise IndependentConfirmationError("source manifest row schema drift")
        sample_id = str(row.get("id"))
        if sample_id in source_rows:
            raise IndependentConfirmationError("duplicate source manifest row")
        source_rows[sample_id] = row
    if not set(eligible_ids).issubset(source_rows):
        raise IndependentConfirmationError("eligible source row missing")
    make_counts = Counter(
        str(source_rows[sample_id]["make"]) for sample_id in eligible_ids
    )
    if dict(sorted(make_counts.items())) != dict(
        sorted(population["expected_make_counts"].items())
    ):
        raise IndependentConfirmationError("camera-make population drift")
    if len(make_counts) != int(population["expected_camera_makes"]):
        raise IndependentConfirmationError("camera-make count drift")

    fixed = config["fixed_operators"]
    candidate = fixed["candidate"]
    if (
        candidate["candidate_id"] != parent_spec["retained_candidate"]
        or candidate["order"] != "density_then_anchor"
        or float(candidate["density_strength"]) != 0.5
        or int(candidate["final_output_margin"]) != 4
    ):
        raise IndependentConfirmationError("retained operator parameter drift")
    expected_comparators = [
        parent_config["parent_anchor"]["candidate_id"],
        parent_config["parent_density"]["candidate_id"],
    ]
    if _candidate_ids(config)[:2] != expected_comparators:
        raise IndependentConfirmationError("parent comparator drift")
    if fixed.get("per_image_selection_allowed") is not False:
        raise IndependentConfirmationError("per-image selection boundary drift")
    if fixed.get("post_result_parameter_change_allowed") is not False:
        raise IndependentConfirmationError("post-result parameter boundary drift")

    parent_validated = validate_parent_contract(root, parent_config)
    development_rows = development_report["candidates"][
        parent_spec["retained_candidate"]
    ]["per_image"]
    if len(development_rows) != int(
        config["development_reference"]["sample_count"]
    ):
        raise IndependentConfirmationError("development sample count drift")
    development_style = np.asarray(
        [row["median_style_delta_e76"] for row in development_rows],
        dtype=np.float64,
    )
    development_residual = np.asarray(
        [
            row["median_non_basic_residual_delta_e76"]
            for row in development_rows
        ],
        dtype=np.float64,
    )
    reference = config["development_reference"]
    _require_close(
        float(np.median(development_style)),
        float(reference["candidate_all_median_style_delta_e76"]),
        "development median style",
    )
    _require_close(
        float(np.median(development_residual)),
        float(reference["candidate_all_median_non_basic_residual_delta_e76"]),
        "development median non-basic",
    )
    _require_close(
        float(np.percentile(development_style, 95)),
        float(reference["candidate_all_p95_style_delta_e76"]),
        "development p95 style",
    )
    _require_close(
        float(np.max(development_style)),
        float(reference["candidate_all_maximum_style_delta_e76"]),
        "development maximum style",
    )
    metrics = config["metrics"]
    retention = float(metrics["development_retention_fraction"])
    upper = float(metrics["development_upper_envelope_multiplier"])
    _require_close(
        float(metrics["minimum_confirmation_median_style_delta_e76"]),
        retention * float(reference["candidate_all_median_style_delta_e76"]),
        "style-retention threshold",
    )
    _require_close(
        float(
            metrics[
                "minimum_confirmation_median_non_basic_residual_delta_e76"
            ]
        ),
        retention
        * float(reference["candidate_all_median_non_basic_residual_delta_e76"]),
        "non-basic-retention threshold",
    )
    _require_close(
        float(metrics["maximum_confirmation_p95_style_delta_e76"]),
        upper * float(reference["candidate_all_p95_style_delta_e76"]),
        "p95-style envelope",
    )
    _require_close(
        float(metrics["maximum_confirmation_per_image_style_delta_e76"]),
        upper * float(reference["candidate_all_maximum_style_delta_e76"]),
        "maximum-style envelope",
    )

    return {
        "parent_config": parent_config,
        "parent_validated": parent_validated,
        "source_rows": {sample_id: source_rows[sample_id] for sample_id in eligible_ids},
        "eligible_ids": eligible_ids,
        "candidate_ids": _candidate_ids(config),
    }


def _save_rgb8_png(path: Path, encoded: np.ndarray) -> str:
    value = np.asarray(encoded)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise IndependentConfirmationError("operator output is not finite RGB in [0,1]")
    pixels = np.rint(value * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(pixels, mode="RGB").save(
        temporary,
        format="PNG",
        compress_level=6,
    )
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
    """Render both fixed parents and the one fixed candidate."""

    validated = validate_contract(root, config)
    parent_config = validated["parent_config"]
    apply_anchor, apply_density = build_operators(
        parent_config,
        validated["parent_validated"],
    )
    fixed_candidate = config["fixed_operators"]["candidate"]
    anchor_id, density_id, candidate_id = validated["candidate_ids"]
    anchor_margin = int(
        parent_config["parent_anchor"]["output_margin"]
    )
    density_strength = float(fixed_candidate["density_strength"])
    final_margin = int(fixed_candidate["final_output_margin"])

    records: list[dict[str, Any]] = []
    for sample_id in validated["eligible_ids"]:
        row = validated["source_rows"][sample_id]
        raw_path = root / str(row["raw_path"])
        decoded_path = root / str(row["decoded_path"])
        if sha256_file(raw_path) != str(row["raw_sha256"]):
            raise IndependentConfirmationError(f"RAW hash mismatch: {sample_id}")
        if sha256_file(decoded_path) != str(row["decoded_sha256"]):
            raise IndependentConfirmationError(
                f"decoded source hash mismatch: {sample_id}"
            )
        with Image.open(decoded_path) as image:
            decoded = ImageOps.exif_transpose(image)
            if decoded.mode != "RGB":
                raise IndependentConfirmationError(
                    f"decoded source is not RGB: {sample_id}"
                )
            source_u8 = np.asarray(decoded, dtype=np.uint8)
            source_float32 = source_u8.astype(np.float32) / 255.0
            source_float64 = source_u8.astype(np.float64) / 255.0

        outputs = {
            anchor_id: apply_output_margin(
                np.asarray(apply_anchor(source_float32), dtype=np.float64),
                anchor_margin,
            ),
            density_id: apply_density(source_float64, density_strength),
            candidate_id: compose_rgb(
                source_float32,
                order=str(fixed_candidate["order"]),
                density_strength=density_strength,
                apply_anchor=apply_anchor,
                apply_density=apply_density,
                output_margin=final_margin,
            ),
        }
        for output_id in validated["candidate_ids"]:
            relative = Path(output_id) / f"{sample_id}.png"
            output_path = output_dir / relative
            output_sha = _save_rgb8_png(output_path, outputs[output_id])
            records.append(
                {
                    "candidate_id": output_id,
                    "sample_id": sample_id,
                    "make": str(row["make"]),
                    "raw_sha256": str(row["raw_sha256"]),
                    "decoded_source_sha256": str(row["decoded_sha256"]),
                    "output": relative.as_posix(),
                    "output_sha256": output_sha,
                }
            )

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "source_manifest_sha256": config["source_preflight"]["manifest_sha256"],
        "source_decision_sha256": config["source_preflight"]["decision_sha256"],
        "candidate_ids": validated["candidate_ids"],
        "candidate_count": len(validated["candidate_ids"]),
        "sample_count": len(validated["eligible_ids"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(output_dir / "manifest.json")
    return {
        "manifest_path": output_dir / "manifest.json",
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _manifest_records(
    *,
    root: Path,
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    manifest_path: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    software_commit = str(manifest.get("software_commit", ""))
    if len(software_commit) != 40 or any(
        character not in "0123456789abcdef" for character in software_commit
    ):
        raise IndependentConfirmationError("manifest software commit drift")
    if (
        manifest.get("experiment_id") != config["experiment_id"]
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("source_manifest_sha256")
        != config["source_preflight"]["manifest_sha256"]
        or manifest.get("source_decision_sha256")
        != config["source_preflight"]["decision_sha256"]
        or manifest.get("candidate_ids") != validated["candidate_ids"]
        or int(manifest.get("candidate_count", -1))
        != len(validated["candidate_ids"])
        or int(manifest.get("sample_count", -1))
        != len(validated["eligible_ids"])
    ):
        raise IndependentConfirmationError("manifest identity drift")
    expected = {
        (candidate_id, sample_id)
        for candidate_id in validated["candidate_ids"]
        for sample_id in validated["eligible_ids"]
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    checked_sources: set[str] = set()
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise IndependentConfirmationError(f"invalid manifest record: {key}")
        source = validated["source_rows"][key[1]]
        if (
            row.get("make") != source["make"]
            or row.get("raw_sha256") != source["raw_sha256"]
            or row.get("decoded_source_sha256") != source["decoded_sha256"]
        ):
            raise IndependentConfirmationError("manifest source binding drift")
        if key[1] not in checked_sources:
            raw_path = root / str(source["raw_path"])
            decoded_path = root / str(source["decoded_path"])
            if (
                sha256_file(raw_path) != source["raw_sha256"]
                or sha256_file(decoded_path) != source["decoded_sha256"]
            ):
                raise IndependentConfirmationError("live source hash drift")
            checked_sources.add(key[1])
        relative_output = Path(str(row["output"]))
        expected_output = Path(key[0]) / f"{key[1]}.png"
        if (
            relative_output.is_absolute()
            or ".." in relative_output.parts
            or relative_output.as_posix() != expected_output.as_posix()
        ):
            raise IndependentConfirmationError("manifest output path drift")
        output_path = manifest_path.parent / relative_output
        if sha256_file(output_path) != row.get("output_sha256"):
            raise IndependentConfirmationError("manifest output hash mismatch")
        records[key] = row
    if set(records) != expected:
        raise IndependentConfirmationError("incomplete confirmation manifest")
    return manifest, records


def paired_advantage(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Sequence[Mapping[str, Any]]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize paired image and camera-make gains over the better parent."""

    candidate_by_id = {
        str(row["sample_id"]): row for row in candidate_rows
    }
    parents_by_id = [
        {str(row["sample_id"]): row for row in rows} for rows in parent_rows
    ]
    if (
        len(candidate_by_id) != len(candidate_rows)
        or any(set(rows) != set(candidate_by_id) for rows in parents_by_id)
    ):
        raise IndependentConfirmationError("paired metric population drift")

    epsilon = float(metrics["paired_win_epsilon"])
    paired_rows = []
    for sample_id, candidate in candidate_by_id.items():
        parents = [rows[sample_id] for rows in parents_by_id]
        best_style = max(float(row["median_style_delta_e76"]) for row in parents)
        best_residual = max(
            float(row["median_non_basic_residual_delta_e76"])
            for row in parents
        )
        style_gain = float(candidate["median_style_delta_e76"]) - best_style
        residual_gain = (
            float(candidate["median_non_basic_residual_delta_e76"])
            - best_residual
        )
        paired_rows.append(
            {
                "sample_id": sample_id,
                "make": str(candidate["make"]),
                "style_gain_over_best_parent_delta_e76": style_gain,
                "non_basic_gain_over_best_parent_delta_e76": residual_gain,
                "style_win": style_gain > epsilon,
                "non_basic_win": residual_gain > epsilon,
            }
        )

    make_rows = []
    for make in sorted({str(row["make"]) for row in candidate_rows}):
        sample_ids = [
            str(row["sample_id"])
            for row in candidate_rows
            if str(row["make"]) == make
        ]
        candidate_style = float(
            np.median(
                [
                    candidate_by_id[sample_id]["median_style_delta_e76"]
                    for sample_id in sample_ids
                ]
            )
        )
        candidate_residual = float(
            np.median(
                [
                    candidate_by_id[sample_id][
                        "median_non_basic_residual_delta_e76"
                    ]
                    for sample_id in sample_ids
                ]
            )
        )
        parent_style = max(
            float(
                np.median(
                    [
                        rows[sample_id]["median_style_delta_e76"]
                        for sample_id in sample_ids
                    ]
                )
            )
            for rows in parents_by_id
        )
        parent_residual = max(
            float(
                np.median(
                    [
                        rows[sample_id][
                            "median_non_basic_residual_delta_e76"
                        ]
                        for sample_id in sample_ids
                    ]
                )
            )
            for rows in parents_by_id
        )
        make_rows.append(
            {
                "make": make,
                "sample_count": len(sample_ids),
                "style_gain_over_best_parent_delta_e76": (
                    candidate_style - parent_style
                ),
                "non_basic_gain_over_best_parent_delta_e76": (
                    candidate_residual - parent_residual
                ),
                "style_win": candidate_style - parent_style > epsilon,
                "non_basic_win": candidate_residual - parent_residual > epsilon,
            }
        )

    return {
        "median_style_gain_over_best_parent_delta_e76": float(
            np.median(
                [
                    row["style_gain_over_best_parent_delta_e76"]
                    for row in paired_rows
                ]
            )
        ),
        "median_non_basic_gain_over_best_parent_delta_e76": float(
            np.median(
                [
                    row["non_basic_gain_over_best_parent_delta_e76"]
                    for row in paired_rows
                ]
            )
        ),
        "per_image_style_wins": sum(row["style_win"] for row in paired_rows),
        "per_image_non_basic_wins": sum(
            row["non_basic_win"] for row in paired_rows
        ),
        "camera_make_style_wins": sum(row["style_win"] for row in make_rows),
        "camera_make_non_basic_wins": sum(
            row["non_basic_win"] for row in make_rows
        ),
        "paired_rows": paired_rows,
        "make_rows": make_rows,
    }


def evaluate_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Evaluate the three fixed outputs under the frozen confirmation gates."""

    validated = validate_contract(root, config)
    manifest, records = _manifest_records(
        root=root,
        config=config,
        validated=validated,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    for candidate_id in validated["candidate_ids"]:
        per_image = []
        for sample_id in validated["eligible_ids"]:
            source = validated["source_rows"][sample_id]
            source_path = root / str(source["decoded_path"])
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            with Image.open(source_path) as source_image:
                source_decoded = ImageOps.exif_transpose(source_image)
                source_size = source_decoded.size
            with Image.open(output_path) as output_image:
                output_decoded = ImageOps.exif_transpose(output_image)
                if output_decoded.mode != "RGB" or output_decoded.size != source_size:
                    raise IndependentConfirmationError(
                        "output decode/dimension mismatch"
                    )
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels,
                output_pixels,
            )
            clipping = new_hard_clipping_fraction(
                source_pixels,
                output_pixels,
                epsilon,
            )
            if not np.all(np.isfinite([style, residual, clipping])):
                raise IndependentConfirmationError("non-finite evaluation metric")
            per_image.append(
                {
                    "sample_id": sample_id,
                    "make": str(source["make"]),
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": clipping,
                    "output_sha256": record["output_sha256"],
                }
            )
        style_values = np.asarray(
            [row["median_style_delta_e76"] for row in per_image],
            dtype=np.float64,
        )
        residual_values = np.asarray(
            [
                row["median_non_basic_residual_delta_e76"]
                for row in per_image
            ],
            dtype=np.float64,
        )
        summaries[candidate_id] = {
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

    anchor_id, density_id, candidate_id = validated["candidate_ids"]
    paired = paired_advantage(
        candidate_rows=summaries[candidate_id]["per_image"],
        parent_rows=[
            summaries[anchor_id]["per_image"],
            summaries[density_id]["per_image"],
        ],
        metrics=config["metrics"],
    )
    thresholds = config["metrics"]
    candidate = summaries[candidate_id]
    automatic_gates = {
        "style_retention": candidate["median_style_delta_e76"]
        >= float(thresholds["minimum_confirmation_median_style_delta_e76"]),
        "non_basic_retention": candidate[
            "median_non_basic_residual_delta_e76"
        ]
        >= float(
            thresholds[
                "minimum_confirmation_median_non_basic_residual_delta_e76"
            ]
        ),
        "p95_style_envelope": candidate["p95_style_delta_e76"]
        <= float(thresholds["maximum_confirmation_p95_style_delta_e76"]),
        "maximum_style_envelope": candidate["maximum_style_delta_e76"]
        <= float(
            thresholds["maximum_confirmation_per_image_style_delta_e76"]
        ),
        "median_style_gain": paired[
            "median_style_gain_over_best_parent_delta_e76"
        ]
        >= float(
            thresholds[
                "minimum_candidate_median_gain_over_best_parent_delta_e76"
            ]
        ),
        "median_non_basic_gain": paired[
            "median_non_basic_gain_over_best_parent_delta_e76"
        ]
        >= float(
            thresholds[
                "minimum_candidate_median_gain_over_best_parent_delta_e76"
            ]
        ),
        "per_image_style_wins": paired["per_image_style_wins"]
        >= int(thresholds["minimum_per_image_wins_over_both_parents"]),
        "per_image_non_basic_wins": paired["per_image_non_basic_wins"]
        >= int(thresholds["minimum_per_image_wins_over_both_parents"]),
        "camera_make_style_wins": paired["camera_make_style_wins"]
        >= int(thresholds["minimum_camera_make_wins_over_both_parents"]),
        "camera_make_non_basic_wins": paired["camera_make_non_basic_wins"]
        >= int(thresholds["minimum_camera_make_wins_over_both_parents"]),
        "all_output_clipping": all(
            summary["worst_new_hard_clipping_fraction"]
            <= float(thresholds["maximum_worst_new_hard_clipping_fraction"])
            for summary in summaries.values()
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
        "candidate_count": len(validated["candidate_ids"]),
        "candidate_ids": validated["candidate_ids"],
        "automatic_gates": automatic_gates,
        "automatic_pass": automatic_pass,
        "automatic_decision": (
            "blind_and_full_resolution_review_required"
            if automatic_pass
            else "close_independent_confirmation_without_rescue"
        ),
        "paired_advantage": paired,
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
    """Build three source-visible, output-blind confirmation sheets."""

    validated = validate_contract(root, config)
    automatic_report = evaluate_confirmation(
        root=root,
        config=config,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    if not automatic_report["automatic_pass"]:
        raise IndependentConfirmationError(
            "automatic gate forbids blind-sheet generation"
        )
    _, records = _manifest_records(
        root=root,
        config=config,
        validated=validated,
        manifest_path=manifest_path,
        config_sha256=config_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, dict[str, str]] = {}
    round_paths = []
    round_hashes = []
    tile_width, tile_height, header = 300, 205, 26
    columns_count = 1 + len(validated["candidate_ids"])
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(validated["candidate_ids"])
        random.Random(2026072800 + round_index).shuffle(shuffled)
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
                tile_width * columns_count,
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
                        / str(records[(candidate_id, sample_id)]["output"]),
                    )
                    for candidate_id in shuffled
                ],
            ]
            for column_index, (label, path) in enumerate(row_paths):
                with Image.open(path) as image:
                    tile = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (tile_width, tile_height),
                    )
                x = column_index * tile_width + (tile_width - tile.width) // 2
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
        round_paths.append(path)
        round_hashes.append(sha256_file(path))
    mapping_path = output_dir / "private_mapping.json"
    mapping_encoded = (
        json.dumps(mappings, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    mapping_temporary = mapping_path.with_suffix(".json.tmp")
    mapping_temporary.write_text(mapping_encoded, encoding="utf-8")
    mapping_temporary.replace(mapping_path)
    return {
        "rounds": round_paths,
        "round_sha256": round_hashes,
        "mapping": mapping_path,
        "mapping_sha256": sha256_file(mapping_path),
    }


__all__ = [
    "IndependentConfirmationError",
    "build_blind_sheets",
    "evaluate_confirmation",
    "paired_advantage",
    "render_confirmation",
    "validate_contract",
]
