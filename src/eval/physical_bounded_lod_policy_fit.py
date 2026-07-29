"""U6.P4N development-only fit of a bounded explicit LOD policy."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    adaptive_exact_area_mask,
    compile_two_cumulant_profiles,
    render_density_conditioned_structure,
    render_density_conditioned_structure_adaptive_lod,
)


SCHEMA = "neuro_film.u6_p4n_bounded_lod_policy_fit_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4n_bounded_lod_policy_fit_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4N parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4N contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    policy = contract["policy"]
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or policy["per_image_fit_or_normalization_allowed"]
        or policy["continuous_or_dense_mixture_allowed"]
        or policy["output_clipping_allowed"]
        or policy["final_rgb_generation_allowed"]
    ):
        raise ValueError("unsupported U6.P4N contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4m_decision"],
        parents["p4m_decision_sha256"],
    )
    if (
        decision["decision"]
        != "retain_group_split_reference_pairs_open_bounded_compiler_fit"
        or not decision["next_leaf"].startswith("U6.P4N")
    ):
        raise ValueError("U6.P4M did not open bounded compiler fit")
    manifest_path = root / parents["dataset_manifest"]
    manifest = _load_exact(
        root,
        parents["dataset_manifest"],
        parents["dataset_manifest_sha256"],
    )
    if manifest.get("schema") != (
        "neuro_film.u6_p4m_reference_profile_dataset_manifest.v1"
    ):
        raise ValueError("unsupported U6.P4M dataset manifest")
    return contract, parent, manifest, manifest_path.parent


def _load_array(
    dataset_root: Path, file_row: dict[str, Any]
) -> np.ndarray:
    relative = Path(file_row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe U6.P4N dataset path")
    path = dataset_root / relative
    if (
        path.stat().st_size != int(file_row["bytes"])
        or _sha256(path) != file_row["sha256"]
    ):
        raise ValueError(f"U6.P4N dataset file drift: {relative}")
    values = np.load(path, allow_pickle=False)
    if (
        values.dtype != np.dtype("<f4")
        or values.ndim != 3
        or values.shape[2] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"invalid U6.P4N dataset array: {relative}")
    return values


def _block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    height = values.shape[0] // block_size * block_size
    width = values.shape[1] // block_size * block_size
    cropped = values[:height, :width]
    return cropped.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
        values.shape[2],
    ).mean(axis=(1, 3))


def summarize_candidate(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    fallback_fraction: float,
    block_size: int,
) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    global_mean = float(
        np.max(
            np.abs(
                np.mean(candidate64, axis=(0, 1))
                - np.mean(reference64, axis=(0, 1))
            )
        )
    )
    block_mean = float(
        np.max(
            np.abs(
                _block_means(candidate64, block_size)
                - _block_means(reference64, block_size)
            )
        )
    )
    ratios = []
    for channel in range(reference.shape[2]):
        reference_variance = float(
            np.var(reference64[..., channel])
        )
        if reference_variance > 1e-12:
            ratios.append(
                float(np.var(candidate64[..., channel]))
                / reference_variance
            )
    if not ratios:
        ratios = [1.0]
    return {
        "global_mean_absolute_error": global_mean,
        "block_mean_absolute_error": block_mean,
        "minimum_variance_ratio": min(ratios),
        "maximum_variance_ratio": max(ratios),
        "variance_log2_absolute_error": max(
            abs(float(np.log2(value))) for value in ratios
        ),
        "exact_fallback_fraction": fallback_fraction,
        "minimum_density": float(np.min(candidate64)),
        "maximum_transmittance": float(
            np.max(np.exp(-candidate64))
        ),
    }


def _score(
    metrics: dict[str, float], contract: dict[str, Any]
) -> float:
    components = contract["metrics"]["score_components"]
    weights = contract["metrics"]["score_weights"]
    return (
        float(weights["global_mean"])
        * metrics["global_mean_absolute_error"]
        / float(components["global_mean_absolute_error_scale"])
        + float(weights["block_mean"])
        * metrics["block_mean_absolute_error"]
        / float(components["block_mean_absolute_error_scale"])
        + float(weights["variance"])
        * metrics["variance_log2_absolute_error"]
        / float(components["variance_log2_absolute_scale"])
        + float(weights["fallback"])
        * metrics["exact_fallback_fraction"]
        / float(components["exact_fallback_fraction_scale"])
    )


def _base_profiles(parent: dict[str, Any], seed: int) -> tuple:
    return tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles_from_contract(parent, seed_offset=seed)
    )


def _evaluate_record(
    contract: dict[str, Any],
    parent: dict[str, Any],
    dataset_root: Path,
    record: dict[str, Any],
    *,
    threshold: float | None,
) -> tuple[dict[str, Any], str]:
    target = _load_array(
        dataset_root, record["files"]["input_density"]
    )
    reference = _load_array(
        dataset_root, record["files"]["target_density"]
    )
    reference_transmittance = _load_array(
        dataset_root, record["files"]["target_transmittance"]
    )
    profiles = _base_profiles(parent, int(record["seed_offset"]))
    factor = int(contract["policy"]["pixel_size_factor"])
    if threshold is None:
        compiled = compile_two_cumulant_profiles(
            profiles, pixel_size_factor=factor
        )
        result = render_density_conditioned_structure(
            target.astype(np.float64), compiled
        )
        candidate = result.density
        fallback_fraction = 0.0
    else:
        result = render_density_conditioned_structure_adaptive_lod(
            target.astype(np.float64),
            profiles,
            pixel_size_factor=factor,
            density_range_threshold=threshold,
        )
        candidate = result.density
        fallback_fraction = float(
            np.mean(
                adaptive_exact_area_mask(
                    target.astype(np.float64),
                    profiles,
                    pixel_size_factor=factor,
                    density_range_threshold=threshold,
                )
            )
        )
    metrics = summarize_candidate(
        reference,
        candidate,
        fallback_fraction=fallback_fraction,
        block_size=int(contract["metrics"]["block_size"]),
    )
    self_error = float(
        np.max(
            np.abs(
                reference.astype(np.float64)
                + np.log(reference_transmittance.astype(np.float64))
            )
        )
    )
    metrics.update(
        {
            "split": record["split"],
            "group_id": record["group_id"],
            "field_family": record["field_family"],
            "self_consistency_error": self_error,
            "score": _score(metrics, contract),
        }
    )
    digest = hashlib.sha256(candidate.tobytes(order="C")).hexdigest()
    return metrics, digest


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "maximum_global_mean_absolute_error": max(
            row["global_mean_absolute_error"] for row in rows
        ),
        "maximum_block_mean_absolute_error": max(
            row["block_mean_absolute_error"] for row in rows
        ),
        "minimum_variance_ratio": min(
            row["minimum_variance_ratio"] for row in rows
        ),
        "maximum_variance_ratio": max(
            row["maximum_variance_ratio"] for row in rows
        ),
        "mean_exact_fallback_fraction": float(
            np.mean([row["exact_fallback_fraction"] for row in rows])
        ),
        "maximum_self_consistency_error": max(
            row["self_consistency_error"] for row in rows
        ),
    }


def evaluate_bounded_lod_policy(
    contract: dict[str, Any],
    parent: dict[str, Any],
    manifest: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    records = manifest["records"]
    development = [
        row
        for row in records
        if row["split"] == contract["policy"]["selection_split"]
    ]
    sealed_names = set(contract["policy"]["sealed_splits"])
    sealed = [row for row in records if row["split"] in sealed_names]
    group_sets = {
        split: {
            row["group_id"] for row in records if row["split"] == split
        }
        for split in {
            contract["policy"]["selection_split"],
            *contract["policy"]["sealed_splits"],
        }
    }
    group_overlap = 0
    split_names = sorted(group_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            group_overlap += len(group_sets[left] & group_sets[right])

    baseline_rows = [
        _evaluate_record(
            contract,
            parent,
            dataset_root,
            row,
            threshold=None,
        )[0]
        for row in development
    ]
    development_grid = []
    for threshold in contract["policy"]["threshold_candidates"]:
        rows = [
            _evaluate_record(
                contract,
                parent,
                dataset_root,
                row,
                threshold=float(threshold),
            )[0]
            for row in development
        ]
        development_grid.append(
            {
                "threshold": float(threshold),
                "aggregate": _aggregate(rows),
            }
        )
    selected = min(
        development_grid,
        key=lambda row: (
            row["aggregate"]["mean_score"],
            row["threshold"],
        ),
    )
    selected_threshold = float(selected["threshold"])

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_hashes: dict[str, list[str]] = {}
    for split in [
        contract["policy"]["selection_split"],
        *contract["policy"]["sealed_splits"],
    ]:
        rows = []
        hashes = []
        for record in records:
            if record["split"] != split:
                continue
            metrics, digest = _evaluate_record(
                contract,
                parent,
                dataset_root,
                record,
                threshold=selected_threshold,
            )
            rows.append(metrics)
            hashes.append(digest)
        split_rows[split] = rows
        split_hashes[split] = hashes

    repeat_hashes = {}
    for split in split_rows:
        second = [
            _evaluate_record(
                contract,
                parent,
                dataset_root,
                record,
                threshold=selected_threshold,
            )[1]
            for record in records
            if record["split"] == split
        ]
        repeat_hashes[split] = second

    aggregates = {
        split: _aggregate(rows) for split, rows in split_rows.items()
    }
    development_aggregate = aggregates[
        contract["policy"]["selection_split"]
    ]
    baseline = _aggregate(baseline_rows)
    improvement = (
        baseline["mean_score"] - development_aggregate["mean_score"]
    ) / baseline["mean_score"]
    gates = contract["automatic_gates"]
    sealed_aggregates = [
        aggregates[name] for name in contract["policy"]["sealed_splits"]
    ]
    checks = {
        "development_improvement": improvement
        >= float(
            gates[
                "minimum_development_score_improvement_fraction_vs_global_p4h"
            ]
        ),
        "confirmation_score_stability": aggregates["confirmation"][
            "mean_score"
        ]
        / development_aggregate["mean_score"]
        <= float(gates["maximum_confirmation_score_ratio_to_development"]),
        "stress_score_stability": aggregates["stress"]["mean_score"]
        / development_aggregate["mean_score"]
        <= float(gates["maximum_stress_score_ratio_to_development"]),
        "sealed_global_mean": max(
            row["maximum_global_mean_absolute_error"]
            for row in sealed_aggregates
        )
        <= float(gates["maximum_sealed_global_mean_absolute_error"]),
        "sealed_block_mean": max(
            row["maximum_block_mean_absolute_error"]
            for row in sealed_aggregates
        )
        <= float(gates["maximum_sealed_block_mean_absolute_error"]),
        "sealed_variance": min(
            row["minimum_variance_ratio"] for row in sealed_aggregates
        )
        >= float(gates["minimum_sealed_variance_ratio"])
        and max(
            row["maximum_variance_ratio"] for row in sealed_aggregates
        )
        <= float(gates["maximum_sealed_variance_ratio"]),
        "sealed_fallback": max(
            row["mean_exact_fallback_fraction"]
            for row in sealed_aggregates
        )
        <= float(gates["maximum_sealed_exact_fallback_fraction_mean"]),
        "exact_control": max(
            row["maximum_self_consistency_error"]
            for row in aggregates.values()
        )
        <= 1e-6,
        "group_split_disjoint": group_overlap
        == int(gates["group_split_overlap_count"]),
        "repeat_exact": all(
            split_hashes[name] == repeat_hashes[name]
            for name in split_hashes
        )
        == bool(gates["repeat_exact"]),
        "physical_domain": (
            min(
                row["minimum_density"]
                for rows in split_rows.values()
                for row in rows
            )
            >= 0.0
            and max(
                row["maximum_transmittance"]
                for rows in split_rows.values()
                for row in rows
            )
            <= 1.0
        )
        == bool(gates["density_and_transmittance_domain_valid"]),
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "selected_threshold": selected_threshold,
        "development_grid": development_grid,
        "global_p4h_development": baseline,
        "selected_policy": aggregates,
        "development_score_improvement_fraction_vs_global_p4h": improvement,
        "group_split_overlap_count": group_overlap,
        "output_hashes": split_hashes,
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_bounded_lod_policy",
    "load_contract",
    "summarize_candidate",
]
