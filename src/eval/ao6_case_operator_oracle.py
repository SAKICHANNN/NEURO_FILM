"""Hard case-operator Oracle audit for the deterministic AO6 control."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.ao6_global_lut_distillation import (
    _aligned_sample,
    _decode_target,
    _fold,
    _load_source,
    _load_exact_json,
    validate_contract as validate_bm0_contract,
)
from src.eval.global_frontier import sha256_file
from src.film_physics.display_look import (
    build_density_source_context_row_staged,
    build_display_look_stages_from_density_context,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


SCHEMA = "neuro_film.u5_r2bm1_ao6_case_operator_oracle_report.v1"


class AO6CaseOperatorOracleError(RuntimeError):
    """Raised when the frozen BM1 contract or evidence drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _grid(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((red, green, blue), axis=-1).reshape(size, size * size, 3)


def _apply_context(
    payload: Mapping[str, Any],
    context: Any,
    encoded: np.ndarray,
) -> np.ndarray:
    apply_base, apply_residual = build_display_look_stages_from_density_context(
        dict(payload), context, application_shape=encoded.shape
    )
    return np.asarray(apply_residual(apply_base(encoded)), dtype=np.float64)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _safe_strength_oracle(
    source: np.ndarray,
    medoid: np.ndarray,
    target: np.ndarray,
    *,
    maximum_strength: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Fit one scalar along a fixed residual with an analytic RGB bound."""

    source_value = np.asarray(source, dtype=np.float64)
    direction = np.asarray(medoid, dtype=np.float64) - source_value
    denominator = float(np.sum(direction * direction))
    if denominator <= 1e-18:
        return source_value.copy(), 0.0
    unconstrained = float(
        np.sum(direction * (np.asarray(target, dtype=np.float64) - source_value))
        / denominator
    )
    positive = direction > 0.0
    negative = direction < 0.0
    limits = [float(maximum_strength)]
    if np.any(positive):
        limits.append(float(np.min((1.0 - source_value[positive]) / direction[positive])))
    if np.any(negative):
        limits.append(float(np.min((0.0 - source_value[negative]) / direction[negative])))
    safe_maximum = max(0.0, min(limits))
    strength = float(np.clip(unconstrained, 0.0, safe_maximum))
    output = source_value + strength * direction
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise AO6CaseOperatorOracleError("strength oracle left encoded RGB")
    return output, strength


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config["protocol"]
    if (
        config.get("schema") != "neuro_film.u5_r2bm1_ao6_case_operator_oracle.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("router_training_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("product_integration_allowed")
        or protocol.get("folds") != 3
        or not protocol.get("selection_is_hard")
        or protocol.get("dense_case_blending_allowed")
        or protocol.get("content_features_used")
    ):
        raise AO6CaseOperatorOracleError("BM1 frozen contract drift")
    parents = config["parents"]
    bm0 = _load_exact_json(
        root,
        parents["population_contract"],
        parents["population_contract_sha256"],
    )
    decision = _load_exact_json(
        root,
        parents["population_decision"],
        parents["population_decision_sha256"],
    )
    profile_config = _load_exact_json(
        root,
        parents["profile_contract"],
        parents["profile_contract_sha256"],
    )
    if decision.get("decision") != parents["required_population_decision"]:
        raise AO6CaseOperatorOracleError("BM0 decision drift")
    population = validate_bm0_contract(root, bm0)
    if len(population["source_rows"]) != parents["expected_sources"]:
        raise AO6CaseOperatorOracleError("BM1 population drift")
    artifact = compile_standalone_profile_artifact(
        root=root, config=profile_config
    )
    payload = artifact["component_payloads"].get(
        "ao6-source-context-display-look"
    )
    if not isinstance(payload, dict):
        raise AO6CaseOperatorOracleError("AO6 display payload absent")
    return {**population, "display_payload": payload}


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AO6CaseOperatorOracleError("BM1 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BM1 output is create-only")
    output_dir.mkdir(parents=True)
    validated = validate_contract(root, config)
    protocol = config["protocol"]
    source_rows = validated["source_rows"]
    target_rows = validated["target_rows"]
    fold_map = validated["fold_map"]
    payload = validated["display_payload"]

    contexts: dict[str, Any] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    self_replay: dict[str, float] = {}
    for source_id in sorted(source_rows):
        source = _load_source(root, source_rows[source_id])
        contexts[source_id] = build_density_source_context_row_staged(
            payload,
            source,
            tile_rows=int(protocol["context_tile_rows"]),
        )
        target_row = target_rows[source_id]
        target = _decode_target(
            validated["target_root"] / target_row["output"],
            target_row["output_sha256"],
        )
        source_sample, target_sample = _aligned_sample(
            source, target, int(protocol["evaluation_samples_per_source"])
        )
        side = int(round(np.sqrt(len(source_sample))))
        if side * side != len(source_sample):
            raise AO6CaseOperatorOracleError("evaluation sample count must be square")
        sample_shape = (side, side, 3)
        source_sample = source_sample.reshape(sample_shape)
        target_sample = target_sample.reshape(sample_shape)
        self_output = _apply_context(payload, contexts[source_id], source_sample)
        self_replay[source_id] = _rmse(self_output, target_sample)
        samples[source_id] = (source_sample, target_sample)
        del source, target, self_output

    grid = _grid(int(protocol["operator_signature_grid_size"]))
    signatures = {
        source_id: _apply_context(payload, contexts[source_id], grid)
        for source_id in sorted(contexts)
    }
    signature_ids = sorted(signatures)
    pairwise: dict[tuple[str, str], float] = {}
    direction_cosines: list[float] = []
    for index, left_id in enumerate(signature_ids):
        left_residual = (signatures[left_id] - grid).reshape(-1)
        for right_id in signature_ids[index + 1 :]:
            pairwise[left_id, right_id] = _rmse(
                signatures[left_id], signatures[right_id]
            )
            right_residual = (signatures[right_id] - grid).reshape(-1)
            denominator = float(
                np.linalg.norm(left_residual) * np.linalg.norm(right_residual)
            )
            direction_cosines.append(
                float(np.dot(left_residual, right_residual) / denominator)
                if denominator > 1e-18
                else 1.0
            )

    rows: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    epsilon = float(protocol["boundary_epsilon_encoded_srgb"])
    for held_fold in range(int(protocol["folds"])):
        development = [
            source_id
            for source_id in signature_ids
            if fold_map[source_id] != held_fold
        ]
        held = [
            source_id
            for source_id in signature_ids
            if fold_map[source_id] == held_fold
        ]
        mean_distances = {
            candidate: float(
                np.mean(
                    [
                        _rmse(signatures[candidate], signatures[other])
                        for other in development
                        if other != candidate
                    ]
                )
            )
            for candidate in development
        }
        medoid_id = min(development, key=lambda item: (mean_distances[item], item))
        fold_models.append(
            {
                "held_fold": held_fold,
                "development_case_ids": development,
                "held_source_ids": held,
                "global_medoid_case_id": medoid_id,
                "global_medoid_mean_signature_rmse": mean_distances[medoid_id],
            }
        )
        for source_id in held:
            source_sample, target_sample = samples[source_id]
            candidate_outputs = {
                case_id: _apply_context(payload, contexts[case_id], source_sample)
                for case_id in development
            }
            candidate_errors = {
                case_id: _rmse(output, target_sample)
                for case_id, output in candidate_outputs.items()
            }
            selected_id = min(
                development, key=lambda item: (candidate_errors[item], item)
            )
            global_output = candidate_outputs[medoid_id]
            case_output = candidate_outputs[selected_id]
            strength_output, strength = _safe_strength_oracle(
                source_sample, global_output, target_sample
            )
            global_error = candidate_errors[medoid_id]
            case_error = candidate_errors[selected_id]
            strength_error = _rmse(strength_output, target_sample)
            target_style = _rmse(target_sample, source_sample)
            case_style = _rmse(case_output, source_sample)
            target_boundary = np.any(
                (target_sample <= epsilon) | (target_sample >= 1.0 - epsilon),
                axis=-1,
            )
            case_boundary = np.any(
                (case_output <= epsilon) | (case_output >= 1.0 - epsilon),
                axis=-1,
            )
            rows.append(
                {
                    "source_id": source_id,
                    "make": source_rows[source_id]["make"],
                    "held_fold": held_fold,
                    "global_medoid_case_id": medoid_id,
                    "selected_case_id": selected_id,
                    "selected_nonmedoid": selected_id != medoid_id,
                    "self_context_target_rgb_rmse": self_replay[source_id],
                    "global_medoid_target_rgb_rmse": global_error,
                    "strength_oracle_target_rgb_rmse": strength_error,
                    "case_oracle_target_rgb_rmse": case_error,
                    "case_vs_global_rmse_ratio": case_error / global_error,
                    "case_vs_strength_rmse_ratio": case_error / strength_error,
                    "strength_oracle_scalar": strength,
                    "case_style_retention_ratio": case_style / target_style,
                    "new_boundary_fraction_vs_ao6": float(
                        np.mean(case_boundary & ~target_boundary)
                    ),
                }
            )

    global_ratios = np.asarray([row["case_vs_global_rmse_ratio"] for row in rows])
    strength_ratios = np.asarray([row["case_vs_strength_rmse_ratio"] for row in rows])
    aggregate = {
        "source_count": len(rows),
        "fold_counts": {
            str(fold): int(sum(row["held_fold"] == fold for row in rows))
            for fold in range(int(protocol["folds"]))
        },
        "maximum_self_context_target_rgb_rmse": float(max(self_replay.values())),
        "median_case_rmse_reduction_vs_global_medoid": float(1.0 - np.median(global_ratios)),
        "median_case_rmse_reduction_vs_strength_oracle": float(1.0 - np.median(strength_ratios)),
        "worst_case_vs_global_rmse_ratio": float(np.max(global_ratios)),
        "sources_case_beats_strength_by_two_percent": int(np.sum(strength_ratios <= 0.98)),
        "distinct_selected_case_ids": len({row["selected_case_id"] for row in rows}),
        "nonmedoid_selection_count": int(sum(row["selected_nonmedoid"] for row in rows)),
        "median_style_retention_ratio": float(np.median([row["case_style_retention_ratio"] for row in rows])),
        "maximum_new_boundary_fraction_vs_ao6": float(max(row["new_boundary_fraction_vs_ao6"] for row in rows)),
        "median_pairwise_operator_signature_rmse": float(np.median(list(pairwise.values()))),
        "median_pairwise_residual_direction_cosine": float(np.median(direction_cosines)),
        "minimum_pairwise_residual_direction_cosine": float(np.min(direction_cosines)),
    }
    gates = config["gates"]
    gate_results = {
        "self_replay": aggregate["maximum_self_context_target_rgb_rmse"] <= gates["maximum_self_context_target_rgb_rmse"],
        "oracle_vs_global": aggregate["median_case_rmse_reduction_vs_global_medoid"] >= gates["minimum_median_case_rmse_reduction_vs_global_medoid"],
        "oracle_vs_strength": aggregate["median_case_rmse_reduction_vs_strength_oracle"] >= gates["minimum_median_case_rmse_reduction_vs_strength_oracle"],
        "worst_case": aggregate["worst_case_vs_global_rmse_ratio"] <= gates["maximum_worst_case_vs_global_rmse_ratio"],
        "source_support": aggregate["sources_case_beats_strength_by_two_percent"] >= gates["minimum_sources_case_beats_strength_by_two_percent"],
        "selection_diversity": aggregate["distinct_selected_case_ids"] >= gates["minimum_distinct_selected_case_ids"],
        "style_retention": gates["minimum_median_style_retention_ratio"] <= aggregate["median_style_retention_ratio"] <= gates["maximum_median_style_retention_ratio"],
        "boundary": aggregate["maximum_new_boundary_fraction_vs_ao6"] <= gates["maximum_new_boundary_fraction_vs_ao6"],
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "fold_assignment": fold_map,
        "operator_space": {
            "representation": "fixed-grid AO6 output signature",
            "content_features_used": False,
            "pair_count": len(pairwise),
        },
        "fold_models": fold_models,
        "rows": rows,
        "aggregate": aggregate,
        "gates": gate_results,
        "automatic_pass": bool(all(gate_results.values())),
        "content_retrieval_opened": bool(all(gate_results.values())),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "AO6CaseOperatorOracleError",
    "_safe_strength_oracle",
    "run_audit",
    "validate_contract",
]
