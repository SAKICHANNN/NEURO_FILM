"""Colour-blind hard content retrieval for the frozen AO6 case bank."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.eval.ao6_case_operator_oracle import (
    _apply_context,
    _canonical_sha256,
    _rmse,
    _safe_strength_oracle,
    validate_contract as validate_bm1_contract,
)
from src.eval.ao6_global_lut_distillation import (
    _aligned_sample,
    _decode_target,
    _load_exact_json,
    _load_source,
)
from src.eval.global_frontier import sha256_file
from src.film_physics.display_look import (
    build_density_source_context_row_staged,
)


SCHEMA = "neuro_film.u5_r2bm2_ao6_content_hard_retrieval_report.v1"


class AO6ContentHardRetrievalError(RuntimeError):
    """Raised when BM2 evidence, information flow or execution drifts."""


def _source_view(
    encoded: np.ndarray, *, resize_short_edge: int, image_size: int
) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise AO6ContentHardRetrievalError("source view must be HxWx3")
    luma = (
        0.2126 * value[..., 0]
        + 0.7152 * value[..., 1]
        + 0.0722 * value[..., 2]
    )
    height, width = luma.shape
    scale = float(resize_short_edge) / float(min(height, width))
    resized = cv2.resize(
        luma,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    top = (resized.shape[0] - image_size) // 2
    left = (resized.shape[1] - image_size) // 2
    crop = resized[top : top + image_size, left : left + image_size]
    if crop.shape != (image_size, image_size) or not np.all(np.isfinite(crop)):
        raise AO6ContentHardRetrievalError("source view crop failed")
    return np.repeat(crop[..., None], 3, axis=-1)


def _extract_features(
    *, root: Path, views: list[np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    spec = config["feature_model"]
    device = str(spec["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise AO6ContentHardRetrievalError("frozen CUDA device unavailable")
    torch.use_deterministic_algorithms(True)
    model = AutoModel.from_pretrained(
        root / spec["root"], local_files_only=True
    ).to(device).eval()
    mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32
    )[:, None, None]
    std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=torch.float32
    )[:, None, None]
    batches: list[np.ndarray] = []
    batch_size = int(spec["batch_size"])
    for start in range(0, len(views), batch_size):
        colours = torch.from_numpy(
            np.stack(views[start : start + batch_size])
        ).permute(0, 3, 1, 2)
        inputs = ((colours - mean) / std).to(device)
        with torch.inference_mode():
            tokens = model(pixel_values=inputs).last_hidden_state[:, 1:].float()
            pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        batches.append(pooled.cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    features = np.concatenate(batches).astype(np.float32, copy=False)
    if features.shape[0] != len(views) or not np.all(np.isfinite(features)):
        raise AO6ContentHardRetrievalError("feature extraction failed")
    return features


def _normalized(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or np.any(norms <= 0.0) or not np.all(np.isfinite(values)):
        raise AO6ContentHardRetrievalError("features must be finite and nonzero")
    return values / norms[:, None]


def _fold_selector(
    *,
    normalized_features: np.ndarray,
    source_ids: list[str],
    development_ids: list[str],
    held_ids: list[str],
) -> tuple[float, dict[str, dict[str, Any]]]:
    indices = {source_id: index for index, source_id in enumerate(source_ids)}
    development_nearest = []
    for source_id in development_ids:
        candidates = [item for item in development_ids if item != source_id]
        similarities = [
            float(
                normalized_features[indices[source_id]]
                @ normalized_features[indices[candidate]]
            )
            for candidate in candidates
        ]
        development_nearest.append(max(similarities))
    threshold = float(
        np.quantile(
            np.asarray(development_nearest, dtype=np.float64),
            0.1,
            method="linear",
        )
    )
    selections: dict[str, dict[str, Any]] = {}
    for source_id in held_ids:
        ranked = sorted(
            development_ids,
            key=lambda candidate: (
                -float(
                    normalized_features[indices[source_id]]
                    @ normalized_features[indices[candidate]]
                ),
                candidate,
            ),
        )
        nearest = ranked[0]
        similarity = float(
            normalized_features[indices[source_id]]
            @ normalized_features[indices[nearest]]
        )
        selections[source_id] = {
            "nearest_source_id": nearest,
            "nearest_cosine_similarity": similarity,
            "fallback": similarity < threshold,
        }
    return threshold, selections


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    selector = config["selector"]
    feature = config["feature_model"]
    if (
        config.get("schema") != "neuro_film.u5_r2bm2_ao6_content_hard_retrieval.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("final_rgb_learning_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("latent_mode_claim_allowed")
        or config.get("product_integration_allowed")
        or feature.get("training_allowed")
        or feature.get("fine_tuning_allowed")
        or selector.get("target_pixels_used")
        or selector.get("source_colour_statistics_used")
        or selector.get("operator_signatures_used_at_inference")
        or selector.get("fitting_allowed")
        or selector.get("dense_blending_allowed")
    ):
        raise AO6ContentHardRetrievalError("BM2 frozen contract drift")
    parent = config["parent"]
    decision = _load_exact_json(
        root, parent["decision"], parent["decision_sha256"]
    )
    report = _load_exact_json(root, parent["report"], parent["report_sha256"])
    if (
        decision.get("decision") != parent["required_decision"]
        or not report.get("automatic_pass")
        or report.get("stable_evidence_id")
        != decision["repeat_evidence"]["stable_evidence_id"]
        or report["aggregate"]["source_count"] != parent["expected_sources"]
    ):
        raise AO6ContentHardRetrievalError("BM1 evidence does not open BM2")
    bm1_config = _load_exact_json(
        root,
        decision["contract"]["path"],
        decision["contract"]["sha256"],
    )
    validated = validate_bm1_contract(root, bm1_config)
    model_root = root / feature["root"]
    for filename, key in (
        ("model.safetensors", "model_sha256"),
        ("config.json", "config_sha256"),
        ("preprocessor_config.json", "preprocessor_sha256"),
    ):
        if sha256_file(model_root / filename) != feature[key]:
            raise AO6ContentHardRetrievalError(f"feature asset drift: {filename}")
    return {
        **validated,
        "bm1_report": report,
        "case_protocol": bm1_config["protocol"],
    }


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AO6ContentHardRetrievalError("BM2 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BM2 output is create-only")
    output_dir.mkdir(parents=True)
    validated = validate_contract(root, config)
    source_rows = validated["source_rows"]
    target_rows = validated["target_rows"]
    fold_map = validated["fold_map"]
    payload = validated["display_payload"]
    bm1_report = validated["bm1_report"]
    protocol = bm1_report
    case_protocol = validated["case_protocol"]

    source_ids = sorted(source_rows)
    contexts: dict[str, Any] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    views: list[np.ndarray] = []
    feature_spec = config["feature_model"]
    for source_id in source_ids:
        source = _load_source(root, source_rows[source_id])
        contexts[source_id] = build_density_source_context_row_staged(
            payload,
            source,
            tile_rows=int(case_protocol["context_tile_rows"]),
        )
        views.append(
            _source_view(
                source,
                resize_short_edge=int(feature_spec["resize_short_edge"]),
                image_size=int(feature_spec["image_size"]),
            )
        )
        target_row = target_rows[source_id]
        target = _decode_target(
            validated["target_root"] / target_row["output"],
            target_row["output_sha256"],
        )
        source_sample, target_sample = _aligned_sample(
            source,
            target,
            int(case_protocol["evaluation_samples_per_source"]),
        )
        side = int(round(np.sqrt(len(source_sample))))
        samples[source_id] = (
            source_sample.reshape(side, side, 3),
            target_sample.reshape(side, side, 3),
        )
        del source, target

    features = _extract_features(root=root, views=views, config=config)
    normalized = _normalized(features)
    feature_sha256 = hashlib.sha256(
        features.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    bm1_folds = {
        int(row["held_fold"]): row for row in protocol["fold_models"]
    }
    bm1_rows = {row["source_id"]: row for row in protocol["rows"]}
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_errors: dict[str, dict[str, float]] = {}
    epsilon = float(case_protocol["boundary_epsilon_encoded_srgb"])
    for held_fold in range(int(config["selector"]["folds"])):
        model = bm1_folds[held_fold]
        development = list(model["development_case_ids"])
        held = list(model["held_source_ids"])
        medoid_id = str(model["global_medoid_case_id"])
        threshold, selections = _fold_selector(
            normalized_features=normalized,
            source_ids=source_ids,
            development_ids=development,
            held_ids=held,
        )
        fold_rows.append(
            {
                "held_fold": held_fold,
                "development_source_ids": development,
                "held_source_ids": held,
                "global_medoid_case_id": medoid_id,
                "ood_cosine_threshold": threshold,
            }
        )
        for source_id in held:
            source_sample, target_sample = samples[source_id]
            outputs = {
                case_id: _apply_context(payload, contexts[case_id], source_sample)
                for case_id in development
            }
            errors = {
                case_id: _rmse(output, target_sample)
                for case_id, output in outputs.items()
            }
            all_errors[source_id] = errors
            oracle_id = min(development, key=lambda item: (errors[item], item))
            selection = selections[source_id]
            retrieved_id = (
                medoid_id if selection["fallback"] else selection["nearest_source_id"]
            )
            global_output = outputs[medoid_id]
            retrieval_output = outputs[retrieved_id]
            strength_output, strength = _safe_strength_oracle(
                source_sample, global_output, target_sample
            )
            global_error = errors[medoid_id]
            retrieval_error = errors[retrieved_id]
            oracle_error = errors[oracle_id]
            strength_error = _rmse(strength_output, target_sample)
            parent_row = bm1_rows[source_id]
            if max(
                abs(global_error - parent_row["global_medoid_target_rgb_rmse"]),
                abs(oracle_error - parent_row["case_oracle_target_rgb_rmse"]),
                abs(strength_error - parent_row["strength_oracle_target_rgb_rmse"]),
            ) > 1e-12:
                raise AO6ContentHardRetrievalError("BM1 metric replay drift")
            target_style = _rmse(target_sample, source_sample)
            retrieval_style = _rmse(retrieval_output, source_sample)
            target_boundary = np.any(
                (target_sample <= epsilon) | (target_sample >= 1.0 - epsilon),
                axis=-1,
            )
            retrieval_boundary = np.any(
                (retrieval_output <= epsilon)
                | (retrieval_output >= 1.0 - epsilon),
                axis=-1,
            )
            rows.append(
                {
                    "source_id": source_id,
                    "make": source_rows[source_id]["make"],
                    "held_fold": held_fold,
                    "global_medoid_case_id": medoid_id,
                    "nearest_content_case_id": selection["nearest_source_id"],
                    "nearest_cosine_similarity": selection["nearest_cosine_similarity"],
                    "ood_fallback": bool(selection["fallback"]),
                    "retrieved_case_id": retrieved_id,
                    "oracle_case_id": oracle_id,
                    "global_target_rgb_rmse": global_error,
                    "strength_target_rgb_rmse": strength_error,
                    "retrieval_target_rgb_rmse": retrieval_error,
                    "oracle_target_rgb_rmse": oracle_error,
                    "retrieval_vs_global_rmse_ratio": retrieval_error / global_error,
                    "retrieval_vs_oracle_rmse_ratio": retrieval_error / oracle_error,
                    "strength_oracle_scalar": strength,
                    "retrieval_style_retention_ratio": retrieval_style / target_style,
                    "new_boundary_fraction_vs_ao6": float(
                        np.mean(retrieval_boundary & ~target_boundary)
                    ),
                }
            )

    global_ratios = np.asarray(
        [row["retrieval_vs_global_rmse_ratio"] for row in rows], dtype=np.float64
    )
    retrieval_reduction = float(1.0 - np.median(global_ratios))
    oracle_reduction = float(
        bm1_report["aggregate"]["median_case_rmse_reduction_vs_global_medoid"]
    )
    rng = random.Random(int(config["controls"]["shuffled_case_mapping_seed"]))
    shuffled_reductions = []
    for _ in range(int(config["controls"]["shuffled_case_mapping_permutations"])):
        shuffled_ratios = []
        for fold in fold_rows:
            development = list(fold["development_source_ids"])
            shuffled = development.copy()
            rng.shuffle(shuffled)
            mapping = dict(zip(development, shuffled, strict=True))
            medoid_id = str(fold["global_medoid_case_id"])
            for source_id in fold["held_source_ids"]:
                row = next(item for item in rows if item["source_id"] == source_id)
                mapped = (
                    medoid_id
                    if row["ood_fallback"]
                    else mapping[row["nearest_content_case_id"]]
                )
                shuffled_ratios.append(
                    all_errors[source_id][mapped] / row["global_target_rgb_rmse"]
                )
        shuffled_reductions.append(
            float(1.0 - np.median(np.asarray(shuffled_ratios)))
        )
    shuffled_p95 = float(
        np.quantile(
            np.asarray(shuffled_reductions), 0.95, method="higher"
        )
    )
    aggregate = {
        "source_count": len(rows),
        "feature_sha256": feature_sha256,
        "median_retrieval_rmse_reduction_vs_global_medoid": retrieval_reduction,
        "fraction_of_bm1_oracle_gain_recovered": retrieval_reduction / oracle_reduction,
        "worst_retrieval_vs_global_rmse_ratio": float(np.max(global_ratios)),
        "sources_retrieval_beats_global": int(np.sum(global_ratios < 1.0)),
        "nonfallback_sources": int(sum(not row["ood_fallback"] for row in rows)),
        "shuffled_reduction_p95": shuffled_p95,
        "retrieval_reduction_advantage_over_shuffled_p95": retrieval_reduction - shuffled_p95,
        "median_style_retention_ratio": float(np.median([row["retrieval_style_retention_ratio"] for row in rows])),
        "maximum_new_boundary_fraction_vs_ao6": float(max(row["new_boundary_fraction_vs_ao6"] for row in rows)),
        "distinct_retrieved_case_ids": len({row["retrieved_case_id"] for row in rows}),
        "oracle_exact_match_count": int(sum(row["retrieved_case_id"] == row["oracle_case_id"] for row in rows)),
        "zero_self_matches": bool(all(row["nearest_content_case_id"] != row["source_id"] for row in rows)),
    }
    gates = config["gates"]
    gate_results = {
        "retrieval_value": retrieval_reduction >= gates["minimum_median_retrieval_rmse_reduction_vs_global_medoid"],
        "oracle_recovery": aggregate["fraction_of_bm1_oracle_gain_recovered"] >= gates["minimum_fraction_of_bm1_oracle_gain_recovered"],
        "tail": aggregate["worst_retrieval_vs_global_rmse_ratio"] <= gates["maximum_worst_retrieval_vs_global_rmse_ratio"],
        "source_support": aggregate["sources_retrieval_beats_global"] >= gates["minimum_sources_retrieval_beats_global"],
        "nonfallback_support": aggregate["nonfallback_sources"] >= gates["minimum_nonfallback_sources"],
        "shuffled_control": aggregate["retrieval_reduction_advantage_over_shuffled_p95"] >= gates["minimum_retrieval_reduction_advantage_over_shuffled_p95"],
        "style_retention": gates["minimum_median_style_retention_ratio"] <= aggregate["median_style_retention_ratio"] <= gates["maximum_median_style_retention_ratio"],
        "boundary": aggregate["maximum_new_boundary_fraction_vs_ao6"] <= gates["maximum_new_boundary_fraction_vs_ao6"],
        "zero_self_matches": aggregate["zero_self_matches"] is bool(gates["require_zero_self_matches"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "feature_sha256": feature_sha256,
        "folds": fold_rows,
        "rows": rows,
        "controls": {
            "shuffled_case_mapping_reductions": shuffled_reductions,
            "bm1_oracle_reduction": oracle_reduction,
        },
        "aggregate": aggregate,
        "gates": gate_results,
        "automatic_pass": bool(all(gate_results.values())),
        "fresh_confirmation_opened": bool(all(gate_results.values())),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "AO6ContentHardRetrievalError",
    "_fold_selector",
    "run_audit",
    "validate_contract",
]
