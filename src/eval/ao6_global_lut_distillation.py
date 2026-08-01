"""Grouped development audit for a fixed AO6-distilled global LUT."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from skimage.color import rgb2lab

from src.eval.filmmatch_identity_residual_ood import _encode_scene_linear
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image
from src.roll2film.global_lut_distillation import fit_shaped_global_lut


SCHEMA = "neuro_film.u5_r2bm0_ao6_global_lut_distillation_report.v1"


class AO6GlobalLUTDistillationError(RuntimeError):
    """Raised when the frozen BM0 evidence or protocol drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise AO6GlobalLUTDistillationError(f"frozen evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _fold(source_id: str, count: int) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _aligned_sample(
    source: np.ndarray,
    target: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise AO6GlobalLUTDistillationError("source/target shape mismatch")
    source_flat = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    count = min(maximum, len(source_flat))
    if count < 256:
        raise AO6GlobalLUTDistillationError("insufficient aligned pixels")
    indices = np.linspace(0, len(source_flat) - 1, count, dtype=np.int64)
    return source_flat[indices], target_flat[indices]


def _decode_target(path: Path, expected: str) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != expected:
        raise AO6GlobalLUTDistillationError("AO6 target identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape[-1] != 3:
        raise AO6GlobalLUTDistillationError("expected AO6 RGB16 PNG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float64) / 65535.0


def _load_source(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    raw_path = root / str(row["raw_path"])
    if not raw_path.is_file() or sha256_file(raw_path) != row["raw_sha256"]:
        raise AO6GlobalLUTDistillationError("RAW identity drift")
    working = load_raw_working_image(raw_path)
    if (
        working.transfer_state != "scene_linear"
        or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
        or not working.orientation_applied
    ):
        raise AO6GlobalLUTDistillationError("WorkingImage contract drift")
    return np.asarray(_encode_scene_linear(working.pixels), dtype=np.float64)


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config["protocol"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bm0_ao6_global_lut_distillation.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or not config.get("training_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("product_integration_allowed")
        or protocol["folds"] != 3
        or protocol["lut_size"] != 17
        or protocol["interpolation"] != "tetrahedral"
    ):
        raise AO6GlobalLUTDistillationError("BM0 frozen contract drift")
    parent = config["parent"]
    _load_exact_json(root, parent["config"], parent["config_sha256"])
    decision = _load_exact_json(root, parent["decision"], parent["decision_sha256"])
    manifest = _load_exact_json(
        root, parent["source_manifest"], parent["source_manifest_sha256"]
    )
    report = _load_exact_json(root, parent["ao6_report"], parent["ao6_report_sha256"])
    if decision.get("decision") != parent["required_decision"]:
        raise AO6GlobalLUTDistillationError("BL8 decision drift")
    all_source_rows = {str(row["id"]): dict(row) for row in manifest}
    target_rows = {
        str(row["source_id"]): dict(row)
        for row in report["rows"]
        if row["arm_id"] == parent["target_arm"]
    }
    source_rows = {
        source_id: all_source_rows[source_id]
        for source_id in target_rows
        if source_id in all_source_rows
    }
    if (
        len(source_rows) != parent["expected_sources"]
        or len(target_rows) != parent["expected_sources"]
        or set(source_rows) != set(target_rows)
        or not report["automatic_gate_pass"]
    ):
        raise AO6GlobalLUTDistillationError("BM0 population drift")
    fold_map = {
        source_id: _fold(source_id, int(protocol["folds"]))
        for source_id in sorted(source_rows)
    }
    if set(fold_map.values()) != set(range(int(protocol["folds"]))):
        raise AO6GlobalLUTDistillationError("empty held group")
    return {
        "source_rows": source_rows,
        "target_rows": target_rows,
        "target_root": (root / parent["ao6_report"]).parent,
        "fold_map": fold_map,
    }


def _lab_delta_e(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_lab = rgb2lab(np.asarray(source, dtype=np.float64)[None, ...])[0]
    target_lab = rgb2lab(np.asarray(target, dtype=np.float64)[None, ...])[0]
    return np.linalg.norm(source_lab - target_lab, axis=-1)


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AO6GlobalLUTDistillationError("BM0 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BM0 output is create-only")
    output_dir.mkdir(parents=True)
    validated = validate_contract(root, config)
    protocol = config["protocol"]
    source_rows = validated["source_rows"]
    target_rows = validated["target_rows"]
    fold_map = validated["fold_map"]

    fit_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source_id in sorted(source_rows):
        source = _load_source(root, source_rows[source_id])
        target_row = target_rows[source_id]
        target = _decode_target(
            validated["target_root"] / target_row["output"],
            target_row["output_sha256"],
        )
        fit_samples[source_id] = _aligned_sample(
            source, target, int(protocol["samples_per_image_fit"])
        )
        del source, target

    models: dict[int, Any] = {}
    model_rows: list[dict[str, Any]] = []
    for held_fold in range(int(protocol["folds"])):
        development = [
            source_id
            for source_id in sorted(fit_samples)
            if fold_map[source_id] != held_fold
        ]
        source_fit = np.concatenate([fit_samples[key][0] for key in development])
        target_fit = np.concatenate([fit_samples[key][1] for key in development])
        model = fit_shaped_global_lut(
            source_fit,
            target_fit,
            shaper_knots=int(protocol["shaper_knots"]),
            minimum_shaper_increment=float(protocol["minimum_shaper_increment"]),
            lut_size=int(protocol["lut_size"]),
            identity_regularization=float(protocol["identity_regularization"]),
            difference_regularization=float(protocol["first_difference_regularization"]),
            lsqr_iteration_limit=int(protocol["lsqr_iteration_limit"]),
            output_margin=float(protocol["output_margin"]),
            minimum_determinant=float(protocol["minimum_tetrahedral_jacobian_determinant"]),
            strength_iterations=int(protocol["residual_strength_binary_search_iterations"]),
        )
        model_payload = model.to_dict()
        model_path = output_dir / f"fold_{held_fold}_model.json"
        model_path.write_text(
            json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        determinants = model.lut.tetrahedron_jacobian_determinants()
        model_rows.append(
            {
                "held_fold": held_fold,
                "development_source_ids": development,
                "held_source_ids": [
                    key for key in sorted(fold_map) if fold_map[key] == held_fold
                ],
                "model_sha256": sha256_file(model_path),
                "residual_strength": model.residual_strength,
                "minimum_tetrahedral_jacobian_determinant": float(np.min(determinants)),
                "nonpositive_tetrahedral_jacobian_count": int(np.sum(determinants <= 0.0)),
                "minimum_output_node": float(np.min(model.lut.values)),
                "maximum_output_node": float(np.max(model.lut.values)),
            }
        )
        models[held_fold] = model

    rows: list[dict[str, Any]] = []
    for source_id in sorted(source_rows):
        held_fold = fold_map[source_id]
        model = models[held_fold]
        source = _load_source(root, source_rows[source_id])
        target_row = target_rows[source_id]
        target = _decode_target(
            validated["target_root"] / target_row["output"],
            target_row["output_sha256"],
        )
        candidate = model.apply(source)
        output_path = output_dir / "renders" / f"{source_id}.png"
        save_srgb16_png(np.asarray(candidate, dtype=np.float32), output_path)
        source_sample, target_sample = _aligned_sample(
            source, target, int(protocol["samples_per_image_evaluation"])
        )
        candidate_sample = model.apply(source_sample)
        shaper_sample = model.shaper.apply(source_sample)
        identity_rmse = float(np.sqrt(np.mean((source_sample - target_sample) ** 2)))
        shaper_rmse = float(np.sqrt(np.mean((shaper_sample - target_sample) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate_sample - target_sample) ** 2)))
        target_style = float(np.sqrt(np.mean((target_sample - source_sample) ** 2)))
        candidate_style = float(np.sqrt(np.mean((candidate_sample - source_sample) ** 2)))
        epsilon = float(protocol["output_margin"])
        target_boundary = np.any(
            (target_sample <= epsilon) | (target_sample >= 1.0 - epsilon), axis=1
        )
        candidate_boundary = np.any(
            (candidate_sample <= epsilon) | (candidate_sample >= 1.0 - epsilon), axis=1
        )
        rows.append(
            {
                "source_id": source_id,
                "make": source_rows[source_id]["make"],
                "held_fold": held_fold,
                "identity_target_rmse": identity_rmse,
                "shaper_target_rmse": shaper_rmse,
                "candidate_target_rmse": candidate_rmse,
                "candidate_vs_identity_target_rmse_ratio": candidate_rmse / identity_rmse,
                "candidate_vs_shaper_target_rmse_ratio": candidate_rmse / shaper_rmse,
                "ao6_style_rgb_rmse_from_source": target_style,
                "candidate_style_rgb_rmse_from_source": candidate_style,
                "style_retention_ratio": candidate_style / target_style,
                "median_target_delta_e76": float(
                    np.median(_lab_delta_e(candidate_sample, target_sample))
                ),
                "new_boundary_fraction_vs_ao6": float(
                    np.mean(candidate_boundary & ~target_boundary)
                ),
                "output": output_path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(output_path),
            }
        )
        del source, target, candidate

    ratios_identity = np.asarray(
        [row["candidate_vs_identity_target_rmse_ratio"] for row in rows]
    )
    ratios_shaper = np.asarray(
        [row["candidate_vs_shaper_target_rmse_ratio"] for row in rows]
    )
    style_ratios = np.asarray([row["style_retention_ratio"] for row in rows])
    aggregate = {
        "source_count": len(rows),
        "fold_counts": {
            str(fold): int(sum(row["held_fold"] == fold for row in rows))
            for fold in range(int(protocol["folds"]))
        },
        "median_target_rmse_reduction_vs_identity": float(1.0 - np.median(ratios_identity)),
        "median_target_rmse_reduction_vs_shaper_only": float(1.0 - np.median(ratios_shaper)),
        "worst_source_target_rmse_ratio": float(np.max(ratios_identity)),
        "median_style_retention_ratio": float(np.median(style_ratios)),
        "median_target_delta_e76": float(np.median([row["median_target_delta_e76"] for row in rows])),
        "maximum_new_boundary_fraction_vs_ao6": float(max(row["new_boundary_fraction_vs_ao6"] for row in rows)),
        "minimum_fold_residual_strength": float(min(row["residual_strength"] for row in model_rows)),
        "minimum_tetrahedral_jacobian_determinant": float(min(row["minimum_tetrahedral_jacobian_determinant"] for row in model_rows)),
        "maximum_nonpositive_tetrahedral_jacobian_count": int(max(row["nonpositive_tetrahedral_jacobian_count"] for row in model_rows)),
    }
    gates = config["gates"]
    gate_results = {
        "median_reduction_vs_identity": aggregate["median_target_rmse_reduction_vs_identity"]
        >= gates["minimum_median_target_rmse_reduction_vs_identity"],
        "median_reduction_vs_shaper": aggregate["median_target_rmse_reduction_vs_shaper_only"]
        >= gates["minimum_median_target_rmse_reduction_vs_shaper_only"],
        "worst_source_ratio": aggregate["worst_source_target_rmse_ratio"]
        <= gates["maximum_worst_source_target_rmse_ratio"],
        "style_retention": gates["minimum_median_style_retention_ratio"]
        <= aggregate["median_style_retention_ratio"]
        <= gates["maximum_median_style_retention_ratio"],
        "target_delta_e": aggregate["median_target_delta_e76"]
        <= gates["maximum_median_target_delta_e76"],
        "boundary": aggregate["maximum_new_boundary_fraction_vs_ao6"]
        <= gates["maximum_new_boundary_fraction_vs_ao6"],
        "residual_strength": aggregate["minimum_fold_residual_strength"]
        >= gates["minimum_fold_residual_strength"],
        "topology": aggregate["maximum_nonpositive_tetrahedral_jacobian_count"]
        <= gates["maximum_nonpositive_tetrahedral_jacobian_count"]
        and aggregate["minimum_tetrahedral_jacobian_determinant"]
        >= gates["minimum_tetrahedral_jacobian_determinant"],
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "fold_assignment": fold_map,
        "models": model_rows,
        "rows": rows,
        "aggregate": aggregate,
        "gates": gate_results,
        "automatic_pass": bool(all(gate_results.values())),
        "visual_review_allowed": bool(all(gate_results.values())),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "AO6GlobalLUTDistillationError",
    "run_audit",
    "validate_contract",
]
