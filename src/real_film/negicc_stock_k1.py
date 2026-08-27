"""Controlled stock-specific K=1 baseline for frozen NegICC IT8 observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.real_film.negicc_it8_source import (
    NegiccIt8SourceError,
    _canonical_bytes,
    _fetch_blob,
    _parse_train,
    _sha256,
)


def _basis(reference_xyz: np.ndarray, exposure_ev: float) -> np.ndarray:
    stimulus = np.log2(np.maximum(reference_xyz / 100.0, 1e-6)) + exposure_ev
    x, y, z = stimulus.T
    return np.column_stack(
        (
            np.ones(len(stimulus), dtype=np.float64),
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
        )
    )


def _target(scanner_rgb: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(scanner_rgb / 65535.0, 1e-8))


def _fit(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> np.ndarray:
    gram = features.T @ features
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, features.T @ targets)


def _predict(
    coefficients: np.ndarray, reference_xyz: np.ndarray, exposure_ev: float
) -> np.ndarray:
    log_rgb = _basis(reference_xyz, exposure_ev) @ coefficients
    return np.exp2(log_rgb) * 65535.0


def _load_train(
    source_config: dict[str, Any], stock_id: str, exposure_ev: int
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    exposure = next(
        row for row in source_config["stocks"][stock_id] if row["ev"] == exposure_ev
    )
    data, fact = _fetch_blob(
        source_config["repository"]["raw_base_url"],
        exposure["train"],
        user_agent="NeuroFilm-SF3-A3I/1.0",
    )
    rows = _parse_train(data)
    patches = sorted(rows)
    scanner_rgb = np.asarray([rows[patch][0] for patch in patches], dtype=np.float64)
    reference_xyz = np.asarray([rows[patch][1] for patch in patches], dtype=np.float64)
    return patches, scanner_rgb, reference_xyz, fact


def _relative_improvement(candidate: np.ndarray, control: np.ndarray) -> np.ndarray:
    return (control - candidate) / np.maximum(control, 1e-12)


def run_baseline(
    config: dict[str, Any], source_config: dict[str, Any], *, reverse: bool = False
) -> dict[str, Any]:
    stocks = sorted(config["stocks"])
    build_exposures = sorted(config["roles"]["build_exposures_ev"])
    if reverse:
        fetch_stocks = list(reversed(stocks))
        fetch_exposures = list(reversed(build_exposures))
    else:
        fetch_stocks = stocks
        fetch_exposures = build_exposures

    build: dict[tuple[str, int], tuple[list[str], np.ndarray, np.ndarray]] = {}
    source_facts: list[dict[str, Any]] = []
    for stock_id in fetch_stocks:
        for exposure_ev in fetch_exposures:
            patches, scanner_rgb, reference_xyz, fact = _load_train(
                source_config, stock_id, exposure_ev
            )
            build[(stock_id, exposure_ev)] = (patches, scanner_rgb, reference_xyz)
            source_facts.append(fact)

    ridge_lambda = float(config["operator"]["ridge_lambda"])
    stock_models: dict[str, np.ndarray] = {}
    pooled_features = []
    pooled_targets = []
    for stock_id in stocks:
        feature_parts = []
        target_parts = []
        for exposure_ev in build_exposures:
            _, scanner_rgb, reference_xyz = build[(stock_id, exposure_ev)]
            feature_parts.append(_basis(reference_xyz, exposure_ev))
            target_parts.append(_target(scanner_rgb))
        features = np.vstack(feature_parts)
        targets = np.vstack(target_parts)
        stock_models[stock_id] = _fit(features, targets, ridge_lambda)
        pooled_features.append(features)
        pooled_targets.append(targets)
    pooled_model = _fit(
        np.vstack(pooled_features), np.vstack(pooled_targets), ridge_lambda
    )

    bundle_payload = {
        "family": config["operator"]["family"],
        "ridge_lambda": ridge_lambda,
        "build_exposures_ev": build_exposures,
        "stock_models": {
            stock_id: stock_models[stock_id].tolist() for stock_id in stocks
        },
        "pooled_model": pooled_model.tolist(),
    }
    bundle_identity = _sha256(_canonical_bytes(bundle_payload))
    confirmation_reads_before_bundle_freeze = 0

    confirmation_ev = int(config["roles"]["confirmation_exposure_ev"])
    confirmation: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for stock_id in fetch_stocks:
        patches, scanner_rgb, reference_xyz, fact = _load_train(
            source_config, stock_id, confirmation_ev
        )
        confirmation[stock_id] = (patches, scanner_rgb, reference_xyz)
        source_facts.append(fact)

    results = []
    classified_correct = 0
    boundary_count = 0
    for stock_id in stocks:
        patches, scanner_rgb, reference_xyz = confirmation[stock_id]
        other_stock = next(candidate for candidate in stocks if candidate != stock_id)
        correct = _predict(stock_models[stock_id], reference_xyz, confirmation_ev)
        wrong = _predict(stock_models[other_stock], reference_xyz, confirmation_ev)
        pooled = _predict(pooled_model, reference_xyz, confirmation_ev)
        observed_log = _target(scanner_rgb)
        correct_error = np.sqrt(np.mean((_target(correct) - observed_log) ** 2, axis=1))
        wrong_error = np.sqrt(np.mean((_target(wrong) - observed_log) ** 2, axis=1))
        pooled_error = np.sqrt(np.mean((_target(pooled) - observed_log) ** 2, axis=1))
        correct_total = float(np.sqrt(np.mean((_target(correct) - observed_log) ** 2)))
        wrong_total = float(np.sqrt(np.mean((_target(wrong) - observed_log) ** 2)))
        if correct_total < wrong_total:
            classified_correct += 1
        new_boundary = int(
            np.count_nonzero(
                ((scanner_rgb > 0.0) & (scanner_rgb < 65535.0))
                & ((correct <= 0.0) | (correct >= 65535.0))
            )
        )
        boundary_count += new_boundary
        results.append(
            {
                "stock_id": stock_id,
                "confirmation_exposure_ev": confirmation_ev,
                "confirmation_rows": len(patches),
                "correct_total_log2_rmse": correct_total,
                "wrong_total_log2_rmse": wrong_total,
                "pooled_total_log2_rmse": float(
                    np.sqrt(np.mean((_target(pooled) - observed_log) ** 2))
                ),
                "correct_vs_wrong_patch_win_rate": float(
                    np.mean(correct_error < wrong_error)
                ),
                "correct_vs_wrong_median_improvement": float(
                    np.median(_relative_improvement(correct_error, wrong_error))
                ),
                "correct_vs_pooled_patch_win_rate": float(
                    np.mean(correct_error < pooled_error)
                ),
                "correct_vs_pooled_median_improvement": float(
                    np.median(_relative_improvement(correct_error, pooled_error))
                ),
                "new_boundary_values": new_boundary,
            }
        )

    gates = config["gates"]
    finite = all(
        np.isfinite(value)
        for row in results
        for key, value in row.items()
        if key not in {"stock_id"}
    )
    gate_results = {
        "confirmation_rows": all(
            row["confirmation_rows"] == gates["required_confirmation_rows_per_stock"]
            for row in results
        ),
        "stock_classification": classified_correct
        == gates["required_stock_classification_correct"],
        "correct_vs_wrong_patch_win_rate": all(
            row["correct_vs_wrong_patch_win_rate"]
            >= gates["correct_vs_wrong_patch_win_rate_min"]
            for row in results
        ),
        "correct_vs_wrong_median_improvement": all(
            row["correct_vs_wrong_median_improvement"]
            >= gates["correct_vs_wrong_median_improvement_min"]
            for row in results
        ),
        "correct_vs_pooled_patch_win_rate": all(
            row["correct_vs_pooled_patch_win_rate"]
            >= gates["correct_vs_pooled_patch_win_rate_min"]
            for row in results
        ),
        "correct_vs_pooled_median_improvement": all(
            row["correct_vs_pooled_median_improvement"]
            >= gates["correct_vs_pooled_median_improvement_min"]
            for row in results
        ),
        "finite": finite,
        "confirmation_reads_before_bundle_freeze": confirmation_reads_before_bundle_freeze
        <= gates["confirmation_reads_before_bundle_freeze_max"],
        "new_boundary": boundary_count <= gates["new_boundary_max"],
    }
    source_facts.sort(key=lambda fact: fact["path"])
    scientific_payload = {
        "source_lock_scientific_identity": config["source_lock"]["scientific_identity"],
        "source_facts": source_facts,
        "bundle_identity": bundle_identity,
        "bundle": bundle_payload,
        "confirmation_reads_before_bundle_freeze": confirmation_reads_before_bundle_freeze,
        "results": results,
        "classified_correct": classified_correct,
        "new_boundary_values": boundary_count,
        "gates": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": "neuro-film.sf3-a3i-negicc-stock-specific-k1-baseline-result.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS_CONTROLLED_K1_SIGNAL"
        if passed
        else "FAIL_CLOSED_CONTROLLED_K1_BASELINE",
        "decision": "OPEN_INDEPENDENT_CONFIRMATION"
        if passed
        else "CLOSE_EXACT_CONTROLLED_K1_FAMILY",
        "scientific_payload": scientific_payload,
        "scientific_identity": _sha256(_canonical_bytes(scientific_payload)),
        "network_reads": len(source_facts),
        "image_or_tiff_requests": 0,
        "pixel_decodes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["NegiccIt8SourceError", "run_baseline"]
