"""Evaluator Oracle for hard per-scene explicit weak-pair cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file
from src.eval.flickr_weak_pair_operator_development import _load_rgb, _registered_samples
from src.roll2film.monotone_curve_matrix import fit_monotone_curve_positive_matrix


SCHEMA = "neuro-film.u5-r2bo4-flickr-weak-pair-case-oracle.v1"


class FlickrWeakPairCaseOracleError(ValueError):
    """Raised on case-Oracle contract, input, split or fit drift."""


def _fit(source: np.ndarray, target: np.ndarray, values: Mapping[str, Any], seed_offset: int = 0):
    return fit_monotone_curve_positive_matrix(
        source,
        target,
        curve_identity_mixture=float(values["curve_identity_mixture"]),
        matrix_identity_mixture=float(values["matrix_identity_mixture"]),
        free_logit_bounds=tuple(float(v) for v in values["free_logit_bounds"]),
        restart_count=int(values["restart_count"]),
        maximum_function_evaluations=int(values["maximum_fit_evaluations"]),
        loss=str(values["loss"]),
        loss_scale=float(values["loss_scale"]),
        seed=int(values["seed"]) + seed_offset,
    )


def _rmse(output: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(output - target))))


def _load_parents(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrWeakPairCaseOracleError("invalid BO4 contract")
    loaded = {}
    for key, value in config["parents"].items():
        path = root / str(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise FlickrWeakPairCaseOracleError(f"parent hash drift: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    development = loaded["development_report"]
    if (
        development.get("stable_evidence_id")
        != config["parents"]["development_report"]["required_stable_evidence_id"]
        or development.get("selected_candidate") is not None
        or development.get("confirmation_pixels_loaded") is not False
    ):
        raise FlickrWeakPairCaseOracleError("BO3 decision boundary drift")
    return loaded


def _prepare_samples(root: Path, config: Mapping[str, Any], parents: Mapping[str, Any]):
    development_ids = set(parents["development_report"]["split"]["development_pair_ids"])
    confirmation_ids = set(parents["development_report"]["split"]["confirmation_pair_ids_held_unread"])
    registration_rows = {str(row["pair_id"]): row for row in parents["registration_report"]["pairs"]}
    manifest_rows = {str(row["local_path"]): row for row in parents["download_manifest"]["rows"]}
    data_root = root / str(config["data_root"])
    samples = {}
    metadata = {}
    for pair_id in sorted(development_ids):
        if pair_id in confirmation_ids:
            raise FlickrWeakPairCaseOracleError("development/confirmation overlap")
        row = registration_rows[pair_id]
        if not row["diagnostics"]["registration_gate_passed"]:
            raise FlickrWeakPairCaseOracleError("development contains rejected registration")
        digital_meta = manifest_rows[str(row["digital_local_path"])]
        film_meta = manifest_rows[str(row["film_local_path"])]
        source, target, facts = _registered_samples(
            _load_rgb(data_root / str(row["digital_local_path"]), str(digital_meta["sha256"])),
            _load_rgb(data_root / str(row["film_local_path"]), str(film_meta["sha256"])),
            np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["paired_samples"],
        )
        samples[pair_id] = (source, target)
        metadata[pair_id] = {"family_id": row["family_id"], "scene_id": int(row["scene_id"]), "sample_facts": facts}
    return samples, metadata, len(confirmation_ids)


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parents = _load_parents(root, config)
    samples, metadata, confirmation_count = _prepare_samples(root, config, parents)
    case_cfg = config["case_operator"]
    case_fits = {}
    fit_records = []
    for index, pair_id in enumerate(sorted(samples)):
        fit = _fit(*samples[pair_id], case_cfg, seed_offset=index)
        case_fits[pair_id] = fit.operator
        fit_records.append({"pair_id": pair_id, "converged": fit.converged, "operator": fit.operator.to_dict(), "development_rgb_rmse": fit.development_rgb_rmse})
    rows = []
    global_fits = []
    for fold in range(3):
        families = sorted({value["family_id"] for value in metadata.values()})
        for family in families:
            train = sorted(pair_id for pair_id, value in metadata.items() if value["family_id"] == family and value["scene_id"] % 3 != fold)
            test = sorted(pair_id for pair_id, value in metadata.items() if value["family_id"] == family and value["scene_id"] % 3 == fold)
            train_source = np.concatenate([samples[pair_id][0] for pair_id in train])
            train_target = np.concatenate([samples[pair_id][1] for pair_id in train])
            global_fit = _fit(train_source, train_target, case_cfg, seed_offset=100 + fold)
            global_fits.append({"fold": fold, "family_id": family, "converged": global_fit.converged, "operator": global_fit.operator.to_dict()})
            train_case_scores = {}
            for candidate in train:
                other = [pair_id for pair_id in train if pair_id != candidate]
                train_case_scores[candidate] = float(np.mean([_rmse(case_fits[candidate].apply(samples[pair_id][0]), samples[pair_id][1]) for pair_id in other]))
            medoid_id = min(train, key=lambda pair_id: (train_case_scores[pair_id], pair_id))
            fixed_id = train[0]
            for pair_id in test:
                source, target = samples[pair_id]
                case_errors = {candidate: _rmse(case_fits[candidate].apply(source), target) for candidate in train}
                ordered = sorted(case_errors, key=lambda candidate: (case_errors[candidate], candidate))
                case_outputs = [case_fits[candidate].apply(source) for candidate in train]
                diversity = [_rmse(case_outputs[i], case_outputs[j]) for i in range(len(case_outputs)) for j in range(i + 1, len(case_outputs))]
                oracle_output = case_fits[ordered[0]].apply(source)
                boundary_epsilon = 1.0 / 65536.0
                interior = np.all((source > boundary_epsilon) & (source < 1.0 - boundary_epsilon), axis=1)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "family_id": family,
                        "scene_id": metadata[pair_id]["scene_id"],
                        "fold": fold,
                        "train_case_count": len(train),
                        "global_rmse": _rmse(global_fit.operator.apply(source), target),
                        "medoid_case_id": medoid_id,
                        "medoid_rmse": case_errors[medoid_id],
                        "fixed_first_case_id": fixed_id,
                        "fixed_first_rmse": case_errors[fixed_id],
                        "oracle_case_id": ordered[0],
                        "oracle_rmse": case_errors[ordered[0]],
                        "second_case_id": ordered[1],
                        "second_rmse": case_errors[ordered[1]],
                        "best_vs_second_relative_margin": (case_errors[ordered[1]] - case_errors[ordered[0]]) / case_errors[ordered[1]],
                        "median_case_output_pairwise_rmse": float(np.median(diversity)),
                        "oracle_new_boundary_fraction": float(
                            np.mean(
                                np.any(
                                    (oracle_output[interior] <= boundary_epsilon)
                                    | (oracle_output[interior] >= 1.0 - boundary_epsilon),
                                    axis=1,
                                )
                            )
                        ),
                        "case_errors": case_errors,
                    }
                )
    global_error = np.asarray([row["global_rmse"] for row in rows])
    medoid_error = np.asarray([row["medoid_rmse"] for row in rows])
    oracle_error = np.asarray([row["oracle_rmse"] for row in rows])
    improvement_global = (global_error - oracle_error) / global_error
    improvement_medoid = (medoid_error - oracle_error) / medoid_error
    families = sorted({row["family_id"] for row in rows})
    family_improvements = {family: float(np.median([improvement_global[index] for index, row in enumerate(rows) if row["family_id"] == family])) for family in families}
    gates = config["evaluation"]
    checks = {
        "mean_improvement_over_global": float(np.mean(improvement_global)) >= float(gates["minimum_oracle_mean_improvement_over_family_global"]),
        "median_improvement_over_global": float(np.median(improvement_global)) >= float(gates["minimum_oracle_median_improvement_over_family_global"]),
        "median_improvement_over_medoid": float(np.median(improvement_medoid)) >= float(gates["minimum_oracle_median_improvement_over_train_medoid"]),
        "win_fraction_over_global": float(np.mean(oracle_error < global_error)) >= float(gates["minimum_oracle_win_fraction_over_family_global"]),
        "each_family_median_improvement": all(value >= float(gates["minimum_each_family_median_improvement_over_family_global"]) for value in family_improvements.values()),
        "p95_tail": float(np.quantile(oracle_error, 0.95)) / float(np.quantile(global_error, 0.95)) <= float(gates["maximum_oracle_p95_error_ratio_to_family_global"]),
        "worst_tail": float(np.max(oracle_error)) / float(np.max(global_error)) <= float(gates["maximum_oracle_worst_error_ratio_to_family_global"]),
        "best_second_margin": float(np.median([row["best_vs_second_relative_margin"] for row in rows])) >= float(gates["minimum_median_best_vs_second_relative_margin"]),
        "case_output_diversity": float(np.median([row["median_case_output_pairwise_rmse"] for row in rows])) >= float(gates["minimum_median_case_output_pairwise_rmse"]),
        "all_fits_converged": all(row["converged"] for row in fit_records + global_fits),
        "new_boundary": max(row["oracle_new_boundary_fraction"] for row in rows) <= float(gates["maximum_new_boundary_fraction"]),
    }
    stable = {
        "schema": "neuro-film.u5-r2bo4-flickr-weak-pair-case-oracle-report.v1",
        "node": config["node"],
        "split": {"development_pairs": len(samples), "confirmation_pairs_held_unread": confirmation_count},
        "metrics": {
            "mean_global_rmse": float(np.mean(global_error)),
            "mean_medoid_rmse": float(np.mean(medoid_error)),
            "mean_fixed_first_rmse": float(np.mean([row["fixed_first_rmse"] for row in rows])),
            "mean_oracle_rmse": float(np.mean(oracle_error)),
            "mean_oracle_improvement_over_global": float(np.mean(improvement_global)),
            "median_oracle_improvement_over_global": float(np.median(improvement_global)),
            "median_oracle_improvement_over_medoid": float(np.median(improvement_medoid)),
            "oracle_win_fraction_over_global": float(np.mean(oracle_error < global_error)),
            "per_family_median_improvement_over_global": family_improvements,
            "p95_oracle_error_ratio_to_global": float(np.quantile(oracle_error, 0.95)) / float(np.quantile(global_error, 0.95)),
            "worst_oracle_error_ratio_to_global": float(np.max(oracle_error)) / float(np.max(global_error)),
            "median_best_vs_second_relative_margin": float(np.median([row["best_vs_second_relative_margin"] for row in rows])),
            "median_case_output_pairwise_rmse": float(np.median([row["median_case_output_pairwise_rmse"] for row in rows])),
            "maximum_new_boundary_fraction": max(row["oracle_new_boundary_fraction"] for row in rows),
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "branch": config["branches"]["oracle_pass" if all(checks.values()) else "oracle_fail"],
        "fold_results": rows,
        "case_fit_records": fit_records,
        "global_fit_records": global_fits,
        "confirmation_pixels_loaded": False,
        "router_trained": False,
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


__all__ = ["FlickrWeakPairCaseOracleError", "SCHEMA", "evaluate"]
