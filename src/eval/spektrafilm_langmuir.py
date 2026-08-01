"""Evaluate the isolated Spektrafilm Langmuir donor ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2lab

from src.eval.spektrafilm_spatial_dir import (
    _chroma_hf_p99,
    _gradient_p99,
    _luma,
    _safe_ratio,
    new_isolated_red_speckle_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual


MANIFEST_SCHEMA = "u5-r2bs0-external-render-manifest-v1"


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected BS0 external manifest schema")
    return value


def _uniform_indices(
    height: int,
    width: int,
    maximum_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = min(height, max(1, int(np.floor(np.sqrt(maximum_pixels * height / width)))))
    columns = min(width, max(1, maximum_pixels // rows))
    yy, xx = np.meshgrid(
        np.linspace(0, height - 1, rows, dtype=np.int64),
        np.linspace(0, width - 1, columns, dtype=np.int64),
        indexing="ij",
    )
    return yy.reshape(-1), xx.reshape(-1)


def _sample_triplet(
    source: np.ndarray,
    linear: np.ndarray,
    langmuir: np.ndarray,
    maximum_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source.shape != linear.shape or linear.shape != langmuir.shape:
        raise ValueError("source/linear/Langmuir shape mismatch")
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("BS0 arrays must be HxWx3")
    yy, xx = _uniform_indices(source.shape[0], source.shape[1], maximum_pixels)
    return source[yy, xx], linear[yy, xx], langmuir[yy, xx]


def _delta_e_pixels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_lab = rgb2lab(np.clip(np.asarray(left, dtype=np.float64), 0.0, 1.0))
    right_lab = rgb2lab(np.clip(np.asarray(right, dtype=np.float64), 0.0, 1.0))
    return np.linalg.norm(left_lab - right_lab, axis=-1)


def _luma_quartile_effect(
    linear: np.ndarray, langmuir: np.ndarray
) -> list[dict[str, float]]:
    luma = _luma(linear)
    delta = _delta_e_pixels(linear, langmuir)
    edges = np.quantile(luma, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(luma, edges[1:-1], right=False)
    rows = []
    for index in range(4):
        mask = bins == index
        rows.append(
            {
                "quartile": index,
                "luma_min": float(edges[index]),
                "luma_max": float(edges[index + 1]),
                "mean_delta_e76": float(np.mean(delta[mask])) if np.any(mask) else 0.0,
            }
        )
    return rows


def paired_metrics(
    source: np.ndarray,
    linear: np.ndarray,
    langmuir: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    langmuir = np.asarray(langmuir, dtype=np.float64)
    budget = int(config["diagnostics"]["metric_sample_pixels_per_image"])
    sampled_source, sampled_linear, sampled_langmuir = _sample_triplet(
        source, linear, langmuir, budget
    )
    delta = _delta_e_pixels(sampled_linear, sampled_langmuir)
    style, non_basic = style_and_basic_residual(sampled_source, sampled_langmuir)
    epsilon = float(config["diagnostics"]["hard_boundary_epsilon"])
    new_boundary = ((langmuir <= epsilon) & (linear > epsilon)) | (
        (langmuir >= 1.0 - epsilon) & (linear < 1.0 - epsilon)
    )
    return {
        "shape": list(langmuir.shape),
        "sampled_pixels": int(len(sampled_source)),
        "linear_finite_and_bounded": bool(
            np.all(np.isfinite(linear))
            and np.min(linear) >= 0.0
            and np.max(linear) <= 1.0
        ),
        "langmuir_finite_and_bounded": bool(
            np.all(np.isfinite(langmuir))
            and np.min(langmuir) >= 0.0
            and np.max(langmuir) <= 1.0
        ),
        "langmuir_effect_delta_e76": float(np.mean(delta)),
        "langmuir_effect_p95_delta_e76": float(np.percentile(delta, 95.0)),
        "style_delta_e76_vs_input": float(style),
        "non_basic_residual_delta_e76_vs_input": float(non_basic),
        "new_hard_boundary_fraction_vs_linear": float(np.mean(new_boundary)),
        "luma_gradient_p99_ratio_vs_linear": _safe_ratio(
            _gradient_p99(langmuir), _gradient_p99(linear)
        ),
        "chroma_high_frequency_p99_ratio_vs_linear": _safe_ratio(
            _chroma_hf_p99(langmuir), _chroma_hf_p99(linear)
        ),
        "new_isolated_red_speckle_fraction_vs_linear": new_isolated_red_speckle_fraction(
            linear, langmuir
        ),
        "linear_output_luma_quartile_effect": _luma_quartile_effect(
            sampled_linear, sampled_langmuir
        ),
    }


def _validate_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected BS0 external manifest schema")
    if manifest["external_revision"] != config["external_source"]["revision"]:
        raise ValueError("external revision mismatch")
    if manifest["external_file_sha256"] != config["external_source"]["file_sha256"]:
        raise ValueError("external source hash inventory mismatch")
    if manifest["python_version"] != config["runtime"]["python_version"]:
        raise ValueError("Python version mismatch")
    if manifest["input_manifest_sha256"] != config["inputs"]["frozen_set_sha256"]:
        raise ValueError("input manifest mismatch")
    if len(manifest["records"]) != config["inputs"]["expected_samples"]:
        raise ValueError("manifest source count mismatch")


def evaluate_manifests(
    manifest_a: dict[str, Any],
    manifest_b: dict[str, Any],
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    _validate_manifest(manifest_a, config)
    _validate_manifest(manifest_b, config)
    if manifest_a["run_id"] == manifest_b["run_id"]:
        raise ValueError("two distinct external run IDs are required")
    if manifest_a["package_versions"] != manifest_b["package_versions"]:
        raise ValueError("package version mismatch between runs")
    rows_a = {row["sample_id"]: row for row in manifest_a["records"]}
    rows_b = {row["sample_id"]: row for row in manifest_b["records"]}
    if (
        set(rows_a) != set(rows_b)
        or len(rows_a) != config["inputs"]["expected_samples"]
    ):
        raise ValueError("manifest sample population mismatch")

    repeat_keys = (
        "source_sha256",
        "source_npy_sha256",
        "linear_npy_sha256",
        "langmuir_npy_sha256",
        "linear_png_sha256",
        "langmuir_png_sha256",
    )
    exact_repeat = True
    records = []
    for sample_id in sorted(rows_a):
        first = rows_a[sample_id]
        second = rows_b[sample_id]
        exact_repeat &= all(first[key] == second[key] for key in repeat_keys)
        if sha256_file(root / first["source_path"]) != first["source_sha256"]:
            raise ValueError(f"source hash mismatch: {sample_id}")
        for prefix in ("source", "linear", "langmuir"):
            path = root / first[f"{prefix}_npy_path"]
            if sha256_file(path) != first[f"{prefix}_npy_sha256"]:
                raise ValueError(f"{prefix} array hash mismatch: {sample_id}")
        for prefix in ("linear", "langmuir"):
            path = root / first[f"{prefix}_png_path"]
            if sha256_file(path) != first[f"{prefix}_png_sha256"]:
                raise ValueError(f"{prefix} PNG hash mismatch: {sample_id}")
        source = np.load(root / first["source_npy_path"], allow_pickle=False)
        linear = np.load(root / first["linear_npy_path"], allow_pickle=False)
        langmuir = np.load(root / first["langmuir_npy_path"], allow_pickle=False)
        records.append(
            {
                "sample_id": sample_id,
                **{key: first[key] for key in repeat_keys},
                **paired_metrics(source, linear, langmuir, config),
            }
        )

    def median(key: str) -> float:
        return float(np.median([row[key] for row in records]))

    def worst(key: str) -> float:
        return float(max(row[key] for row in records))

    summary = {
        "sample_count": len(records),
        "exact_repeat": bool(exact_repeat),
        "all_outputs_finite_and_bounded": all(
            row["linear_finite_and_bounded"] and row["langmuir_finite_and_bounded"]
            for row in records
        ),
        "median_langmuir_effect_delta_e76": median("langmuir_effect_delta_e76"),
        "median_style_delta_e76_vs_input": median("style_delta_e76_vs_input"),
        "median_non_basic_residual_delta_e76_vs_input": median(
            "non_basic_residual_delta_e76_vs_input"
        ),
        "worst_new_hard_boundary_fraction_vs_linear": worst(
            "new_hard_boundary_fraction_vs_linear"
        ),
        "worst_new_isolated_red_speckle_fraction_vs_linear": worst(
            "new_isolated_red_speckle_fraction_vs_linear"
        ),
        "worst_luma_gradient_p99_ratio_vs_linear": worst(
            "luma_gradient_p99_ratio_vs_linear"
        ),
        "worst_chroma_high_frequency_p99_ratio_vs_linear": worst(
            "chroma_high_frequency_p99_ratio_vs_linear"
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "exact_repeat": summary["exact_repeat"]
        is gates["two_complete_runs_must_be_exact"],
        "source_count": summary["sample_count"] == gates["source_count_exact"],
        "finite_and_bounded": summary["all_outputs_finite_and_bounded"]
        is gates["all_outputs_finite_and_bounded"],
        "effect_min": summary["median_langmuir_effect_delta_e76"]
        >= gates["minimum_median_langmuir_effect_delta_e76"],
        "effect_max": summary["median_langmuir_effect_delta_e76"]
        <= gates["maximum_median_langmuir_effect_delta_e76"],
        "style": summary["median_style_delta_e76_vs_input"]
        >= gates["minimum_median_style_delta_e76_vs_input"],
        "non_basic": summary["median_non_basic_residual_delta_e76_vs_input"]
        >= gates["minimum_median_non_basic_residual_delta_e76_vs_input"],
        "boundary": summary["worst_new_hard_boundary_fraction_vs_linear"]
        <= gates["maximum_worst_new_hard_boundary_fraction_vs_linear"],
        "red_speckle": summary["worst_new_isolated_red_speckle_fraction_vs_linear"]
        <= gates["maximum_worst_new_isolated_red_speckle_fraction_vs_linear"],
        "gradient": summary["worst_luma_gradient_p99_ratio_vs_linear"]
        <= gates["maximum_worst_luma_gradient_p99_ratio_vs_linear"],
        "chroma_hf": summary["worst_chroma_high_frequency_p99_ratio_vs_linear"]
        <= gates["maximum_worst_chroma_high_frequency_p99_ratio_vs_linear"],
    }
    if not checks["exact_repeat"]:
        decision = "close_runtime_or_replay_failure"
    elif not checks["effect_min"]:
        decision = "close_effect_too_weak"
    elif not all(checks.values()):
        decision = "close_artifact_or_structure_gate"
    else:
        decision = "automatic_pass_requires_blind_visual_review"
    return {
        "summary": summary,
        "gate_checks": checks,
        "decision": decision,
        "visual_review_allowed": decision
        == "automatic_pass_requires_blind_visual_review",
        "package_versions": manifest_a["package_versions"],
        "records": records,
    }


def adjudicate_blind_observations(
    observations: dict[str, Any],
    mapping: dict[str, list[str]],
) -> dict[str, Any]:
    records = observations["records"]
    if observations.get("mapping_read_before_observations") is not False:
        raise ValueError("blind observations were not locked before reveal")
    if {row["sample_id"] for row in records} != set(mapping):
        raise ValueError("blind observation population mismatch")
    pairwise = {
        "langmuir_over_linear": 0,
        "linear_over_langmuir": 0,
        "langmuir_over_ao6": 0,
        "ao6_over_langmuir": 0,
    }
    decoded = []
    severe = {candidate: 0 for candidate in ("linear", "langmuir", "ao6")}
    for row in records:
        sample_id = row["sample_id"]
        candidate_by_letter = dict(zip("ABC", mapping[sample_id], strict=True))
        ranking = [candidate_by_letter[letter] for letter in row["ranking"]]
        if set(ranking) != {"linear", "langmuir", "ao6"}:
            raise ValueError(f"invalid blind ranking: {sample_id}")
        for letter, failed in row["severe_by_candidate"].items():
            severe[candidate_by_letter[letter]] += int(bool(failed))
        if ranking.index("langmuir") < ranking.index("linear"):
            pairwise["langmuir_over_linear"] += 1
        else:
            pairwise["linear_over_langmuir"] += 1
        if ranking.index("langmuir") < ranking.index("ao6"):
            pairwise["langmuir_over_ao6"] += 1
        else:
            pairwise["ao6_over_langmuir"] += 1
        decoded.append(
            {"sample_id": sample_id, "ranking": ranking, "note": row["note"]}
        )
    decision = (
        "close_visual_severe_artifact"
        if severe["langmuir"]
        else "retain_external_mechanism_development_evidence"
    )
    return {
        "pairwise_preference_counts": pairwise,
        "severe_counts": severe,
        "decision": decision,
        "preference_gate_status": "descriptive-no-preregistered-win-threshold",
        "next_allowed_leaf": (
            "independent-bounded-Langmuir-mechanism-pilot"
            if decision == "retain_external_mechanism_development_evidence"
            else "none"
        ),
        "decoded_records": decoded,
    }


__all__ = [
    "MANIFEST_SCHEMA",
    "adjudicate_blind_observations",
    "evaluate_manifests",
    "load_manifest",
    "paired_metrics",
]
