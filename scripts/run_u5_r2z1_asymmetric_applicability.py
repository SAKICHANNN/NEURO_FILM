#!/usr/bin/env python
"""Run frozen U5.R2Z1 asymmetric query/operator applicability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2w2f0_filmset_recipe_global_explainability import (  # noqa: E402
    _contract,
    _group_internal_rows,
    _ids_sha256,
    _load_partition_samples,
)
from src.roll2film.ct5_data import load_ct5_internal_dev_rows  # noqa: E402
from src.roll2film.filmset_asymmetric_applicability import (  # noqa: E402
    all_gates_pass,
    applicability_gate_results,
    applicability_metrics,
    fit_sealed_policy,
    leave_one_case_medoid,
    leave_one_case_oracle,
    leave_one_case_photometric,
    leave_one_case_random,
    leave_one_case_selection,
    operator_signature_matrix,
    oracle_metrics,
    selected_policy_loss,
)
from src.roll2film.filmset_case_retrieval import (  # noqa: E402
    _apply_operator_bank,
    content_descriptor,
    evaluate_within_recipe_case_bank,
    standardized_nearest_indices,
)
from src.roll2film.filmset_reference_preflight import (  # noqa: E402
    partition_filmset_content_ids,
)
from src.roll2film.manifests import FilmSetManifestError  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _software_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(1, 2)))


def _validate_activation(
    config: dict[str, Any],
    z0_decision: dict[str, Any],
    *,
    decision_sha256: str,
) -> None:
    if config.get("status") != "implemented_frozen_pending_formal_run":
        raise ValueError("unexpected Z1 config status")
    parent = config["parent_evidence"]
    domain = str(parent["required_z0_domain"])
    observed_domain = z0_decision.get("domain_decisions", {}).get(domain, {})
    if (
        decision_sha256 != parent["required_z0_decision_sha256"]
        or z0_decision.get("repeat_report_sha256_equal") is not True
        or z0_decision.get("report_a_sha256")
        != parent["required_z0_report_sha256"]
        or z0_decision.get("report_b_sha256")
        != parent["required_z0_report_sha256"]
        or observed_domain.get("decision_branch")
        != parent["required_z0_branch"]
        or observed_domain.get("structure_all")
        is not parent["required_z0_structure_all"]
    ):
        raise RuntimeError("Z0 evidence does not activate Z1")
    if (
        parent["velvia_allowed"] is not False
        or parent["w2f1_output_only_allowed"] is not False
        or config["dataset"]["domains"] != ["classneg"]
    ):
        raise ValueError("Z1 domain boundary drift")


def _fit_bank(
    development_input_fit: np.ndarray,
    development_target_fit: np.ndarray,
    development_input_evaluation: np.ndarray,
    development_target_evaluation: np.ndarray,
    cell_ids: np.ndarray,
    *,
    config: dict[str, Any],
    device: str,
) -> Any:
    operator_settings = config["operators"]["o0"]
    return evaluate_within_recipe_case_bank(
        development_input_fit,
        development_target_fit,
        development_input_evaluation,
        development_target_evaluation,
        development_input_evaluation,
        development_target_evaluation,
        cell_ids,
        operator_settings=operator_settings,
        gates=config["gates"],
        shared_seed=int(operator_settings["shared_fit_seed_base"]),
        case_seed_base=int(operator_settings["case_fit_seed_base"]),
        shuffle_seed=int(config["controls"]["signature_permutation_seed"]),
        bootstrap_seed=int(config["controls"]["bootstrap_seed"]),
        device=device,
    )


def _fixed_medoid_index(
    loss_matrix: np.ndarray, operator_source_indices: np.ndarray
) -> int:
    query_indices = np.arange(len(loss_matrix), dtype=np.int64)
    means = []
    for column, source_index in enumerate(operator_source_indices):
        keep = query_indices != source_index
        means.append(float(np.mean(loss_matrix[keep, column])))
    return int(np.argmin(means))


def _development_rank(
    *,
    rank: int,
    query_raw: np.ndarray,
    operator_raw: np.ndarray,
    loss_matrix: np.ndarray,
    shared_loss: np.ndarray,
    oracle_loss: np.ndarray,
    medoid_loss: np.ndarray,
    spatial_loss: np.ndarray,
    random_loss: np.ndarray,
    operator_source_indices: np.ndarray,
    structure_all: bool,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    reranker = config["reranker"]
    ood = config["ood"]
    controls = config["controls"]
    gates = config["gates"]
    selection = leave_one_case_selection(
        query_raw,
        operator_raw,
        loss_matrix,
        shared_loss,
        operator_source_indices,
        rank=rank,
        ridge_penalty=float(reranker["ridge_penalty"]),
        ood_quantile=float(ood["quantile"]),
    )
    additive = leave_one_case_selection(
        query_raw,
        operator_raw,
        loss_matrix,
        shared_loss,
        operator_source_indices,
        rank=rank,
        ridge_penalty=float(reranker["ridge_penalty"]),
        ood_quantile=float(ood["quantile"]),
        interaction=False,
    )
    rng = np.random.default_rng(
        int(controls["signature_permutation_seed"])
    )
    shuffled_losses = []
    shuffle_selection_sha = hashlib.sha256()
    for shuffle_index in range(
        int(controls["operator_signature_permutations"])
    ):
        if shuffle_index % 8 == 0:
            print(
                f"rank {rank} development shuffle "
                f"{shuffle_index + 1}/"
                f"{controls['operator_signature_permutations']}",
                flush=True,
            )
        shuffle_seed = int(
            rng.integers(0, np.iinfo(np.int32).max)
        ) + shuffle_index
        shuffled = leave_one_case_selection(
            query_raw,
            operator_raw,
            loss_matrix,
            shared_loss,
            operator_source_indices,
            rank=rank,
            ridge_penalty=float(reranker["ridge_penalty"]),
            ood_quantile=float(ood["quantile"]),
            signature_permutation_seed=shuffle_seed,
        )
        shuffled_losses.append(shuffled.selected_loss)
        shuffle_selection_sha.update(
            np.asarray(shuffled.selected_indices, dtype="<i8").tobytes()
        )
    shuffled_loss = np.mean(np.stack(shuffled_losses), axis=0)
    metrics = applicability_metrics(
        shared_loss=shared_loss,
        oracle_loss=oracle_loss,
        policy_loss=selection.selected_loss,
        medoid_loss=medoid_loss,
        spatial_loss=spatial_loss,
        random_loss=random_loss,
        shuffled_loss=shuffled_loss,
        fallback_mask=selection.fallback_mask,
        bootstrap_seed=int(controls["bootstrap_seed"]) + rank,
        bootstrap_samples=int(gates["bootstrap_samples"]),
        bootstrap_confidence=float(gates["bootstrap_confidence"]),
    )
    oracle = oracle_metrics(
        shared_loss,
        oracle_loss,
        bootstrap_seed=int(controls["bootstrap_seed"]) + 100 + rank,
        bootstrap_samples=int(gates["bootstrap_samples"]),
        bootstrap_confidence=float(gates["bootstrap_confidence"]),
    )
    gate_results = applicability_gate_results(
        metrics,
        oracle,
        eligible_case_count=len(operator_raw),
        structure_all=structure_all,
        gates=gates,
    )
    return (
        {
            "rank": rank,
            "metrics": metrics,
            "oracle": oracle,
            "gate_results": gate_results,
            "all_gates_pass": all_gates_pass(gate_results),
            "selection": {
                "operator_indices": selection.selected_indices.tolist(),
                "score_margin": _summary(selection.score_margins),
                "support_distance": _summary(
                    selection.support_distances
                ),
                "ood_threshold": selection.ood_threshold,
                "fallback_count": int(np.sum(selection.fallback_mask)),
            },
            "additive_operator_only": {
                "linear_rgb_rmse": _summary(additive.selected_loss),
                "operator_indices": additive.selected_indices.tolist(),
            },
            "shuffled_signature": {
                "count": len(shuffled_losses),
                "mean_policy_rmse": _summary(shuffled_loss),
                "selection_sha256": shuffle_selection_sha.hexdigest(),
            },
        },
        selection,
    )


def _fresh_photometric_loss(
    bank_features: np.ndarray,
    query_features: np.ndarray,
    loss_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices, _margins = standardized_nearest_indices(
        bank_features, query_features
    )
    return (
        indices,
        loss_matrix[np.arange(len(query_features)), indices],
    )


def _fresh_evaluation(
    *,
    development_query_raw: np.ndarray,
    development_global: np.ndarray,
    development_spatial: np.ndarray,
    operator_raw: np.ndarray,
    development_loss_matrix: np.ndarray,
    operator_source_indices: np.ndarray,
    selected_rank: int,
    ood_threshold: float,
    fresh_input_evaluation: np.ndarray,
    fresh_target_evaluation: np.ndarray,
    fresh_cell_ids: np.ndarray,
    operators: tuple[Any, ...],
    shared_operator: Any,
    structure_all: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    controls = config["controls"]
    reranker = config["reranker"]
    loss_matrix = _apply_operator_bank(
        operators, fresh_input_evaluation, fresh_target_evaluation
    )
    shared_prediction = np.stack(
        [shared_operator.apply(image) for image in fresh_input_evaluation]
    )
    shared_loss = _rmse(shared_prediction, fresh_target_evaluation)
    oracle_loss = np.min(loss_matrix, axis=1)
    sealed = fit_sealed_policy(
        development_query_raw,
        operator_raw,
        development_loss_matrix,
        operator_source_indices,
        rank=selected_rank,
        ridge_penalty=float(reranker["ridge_penalty"]),
        ood_threshold=ood_threshold,
    )
    selected_indices, score_margins, support_distances = sealed.select(
        content_descriptor(
            fresh_input_evaluation,
            fresh_cell_ids,
            include_spatial_cells=True,
        )
    )
    policy_loss = selected_policy_loss(
        selected_indices, loss_matrix, shared_loss
    )
    medoid_index = _fixed_medoid_index(
        development_loss_matrix, operator_source_indices
    )
    medoid_loss = loss_matrix[:, medoid_index]
    fresh_global = content_descriptor(
        fresh_input_evaluation,
        fresh_cell_ids,
        include_spatial_cells=False,
    )
    fresh_spatial = content_descriptor(
        fresh_input_evaluation,
        fresh_cell_ids,
        include_spatial_cells=True,
    )
    global_indices, global_loss = _fresh_photometric_loss(
        development_global, fresh_global, loss_matrix
    )
    spatial_indices, spatial_loss = _fresh_photometric_loss(
        development_spatial, fresh_spatial, loss_matrix
    )
    random_loss = np.mean(loss_matrix, axis=1)
    rng = np.random.default_rng(
        int(controls["signature_permutation_seed"])
    )
    shuffled_losses = []
    shuffle_selection_sha = hashlib.sha256()
    for _ in range(int(controls["operator_signature_permutations"])):
        permutation = rng.permutation(len(operator_raw))
        shuffled_policy = fit_sealed_policy(
            development_query_raw,
            operator_raw[permutation],
            development_loss_matrix,
            operator_source_indices,
            rank=selected_rank,
            ridge_penalty=float(reranker["ridge_penalty"]),
            ood_threshold=ood_threshold,
        )
        shuffled_indices, _margins, _support = shuffled_policy.select(
            fresh_spatial
        )
        shuffled_losses.append(
            selected_policy_loss(
                shuffled_indices, loss_matrix, shared_loss
            )
        )
        shuffle_selection_sha.update(
            np.asarray(shuffled_indices, dtype="<i8").tobytes()
        )
    shuffled_loss = np.mean(np.stack(shuffled_losses), axis=0)
    metrics = applicability_metrics(
        shared_loss=shared_loss,
        oracle_loss=oracle_loss,
        policy_loss=policy_loss,
        medoid_loss=medoid_loss,
        spatial_loss=spatial_loss,
        random_loss=random_loss,
        shuffled_loss=shuffled_loss,
        fallback_mask=selected_indices == -1,
        bootstrap_seed=int(controls["bootstrap_seed"]) + 1000,
        bootstrap_samples=int(gates["bootstrap_samples"]),
        bootstrap_confidence=float(gates["bootstrap_confidence"]),
    )
    oracle = oracle_metrics(
        shared_loss,
        oracle_loss,
        bootstrap_seed=int(controls["bootstrap_seed"]) + 1100,
        bootstrap_samples=int(gates["bootstrap_samples"]),
        bootstrap_confidence=float(gates["bootstrap_confidence"]),
    )
    gate_results = applicability_gate_results(
        metrics,
        oracle,
        eligible_case_count=len(operators),
        structure_all=structure_all,
        gates=gates,
    )
    oracle_gate_names = (
        "oracle_improvement",
        "oracle_win_fraction",
        "oracle_bootstrap",
    )
    oracle_pass = all(gate_results[name] for name in oracle_gate_names)
    if not oracle_pass:
        branch = "fresh_case_bank_oracle_fail"
    elif all_gates_pass(gate_results):
        branch = "asymmetric_applicability_pass"
    else:
        branch = "asymmetric_applicability_fail"
    selected_outputs = []
    for index, selected in enumerate(selected_indices):
        operator = (
            shared_operator if selected == -1 else operators[int(selected)]
        )
        selected_outputs.append(operator.apply(fresh_input_evaluation[index]))
    selected_output = np.stack(selected_outputs)
    output_range_pass = (
        float(np.min(selected_output)) >= float(gates["minimum_output"])
        and float(np.max(selected_output)) <= float(gates["maximum_output"])
    )
    if not output_range_pass:
        branch = "structure_or_repeat_fail"
    return {
        "fresh_target_payloads_accessed": len(fresh_target_evaluation),
        "linear_rgb_rmse": {
            **metrics["linear_rgb_rmse"],
            "global_photometric": _summary(global_loss),
        },
        "metrics": metrics,
        "oracle": oracle,
        "gate_results": {
            **gate_results,
            "selected_output_range": output_range_pass,
        },
        "selection": {
            "operator_indices": selected_indices.tolist(),
            "global_photometric_indices": global_indices.tolist(),
            "spatial_photometric_indices": spatial_indices.tolist(),
            "score_margin": _summary(score_margins),
            "support_distance": _summary(support_distances),
            "ood_threshold": ood_threshold,
            "fixed_medoid_operator_index": medoid_index,
        },
        "selected_output": {
            "minimum": float(np.min(selected_output)),
            "maximum": float(np.max(selected_output)),
        },
        "shuffled_signature": {
            "count": len(shuffled_losses),
            "mean_policy_rmse": _summary(shuffled_loss),
            "selection_sha256": shuffle_selection_sha.hexdigest(),
        },
        "decision_branch_before_repeat": branch,
    }


def run_experiment(
    config: dict[str, Any],
    z0_decision: dict[str, Any],
    *,
    config_sha256: str,
    decision_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    _validate_activation(
        config, z0_decision, decision_sha256=decision_sha256
    )
    contract = _contract(config, config_sha256=config_sha256)
    all_rows = load_ct5_internal_dev_rows(contract)
    allowed_domains = {"input", *contract.domains}
    rows = [
        row for row in all_rows if str(row["domain"]) in allowed_domains
    ]
    grouped = _group_internal_rows(rows, contract.domains)
    partition = config["partition"]
    order = partition_filmset_content_ids(
        grouped, seed=int(partition["seed"]), pool="internal"
    )
    development_count = int(
        partition["preflight_internal_development_identities"]
    )
    confirmatory_count = int(
        partition["preflight_internal_confirmatory_identities"]
    )
    stress_count = int(partition["preflight_internal_stress_identities"])
    if development_count + confirmatory_count + stress_count != len(order):
        raise FilmSetManifestError("Z1 partition does not exhaust identities")
    development_partition = order[:development_count]
    confirmatory_partition = order[
        development_count : development_count + confirmatory_count
    ]
    stress_partition = order[development_count + confirmatory_count :]
    development_start = int(partition["development_offset"])
    development_ids = development_partition[
        development_start : development_start
        + int(partition["development_identities_used"])
    ]
    z0_start = int(partition["z0_confirmatory_offset"])
    z0_ids = confirmatory_partition[
        z0_start : z0_start
        + int(partition["z0_confirmatory_identities_used"])
    ]
    fresh_start = int(partition["fresh_confirmatory_offset"])
    fresh_ids = confirmatory_partition[
        fresh_start : fresh_start
        + int(partition["fresh_confirmatory_identities_used"])
    ]
    if (
        set(development_ids) & set(fresh_ids)
        or set(z0_ids) & set(fresh_ids)
        or len(fresh_ids) != int(
            partition["fresh_confirmatory_identities_used"]
        )
    ):
        raise FilmSetManifestError("Z1 fresh partition leakage")
    (
        development_input_fit,
        development_target_fit,
        development_input_evaluation,
        development_target_evaluation,
        development_cells,
    ) = _load_partition_samples(
        contract,
        grouped,
        development_ids,
        config=config,
        progress_label="development",
    )
    requested_device = str(config["operators"]["o0"]["device"])
    device = (
        "cuda"
        if requested_device == "cuda_if_available_else_cpu"
        and torch.cuda.is_available()
        else "cpu"
    )
    target_fit = development_target_fit["classneg"]
    target_evaluation = development_target_evaluation["classneg"]
    print("fitting fixed ClassNeg case bank", flush=True)
    bank = _fit_bank(
        development_input_fit,
        target_fit,
        development_input_evaluation,
        target_evaluation,
        development_cells,
        config=config,
        device=device,
    )
    print("fixed ClassNeg case bank complete", flush=True)
    if bank.shared_operator is None:
        raise RuntimeError("Z1 bank did not expose shared O0")
    eligible_indices = np.asarray(
        bank.report["eligible_development_indices"], dtype=np.int64
    )
    operators = bank.eligible_operators
    operator_sources = eligible_indices.copy()
    loss_matrix = _apply_operator_bank(
        operators, development_input_evaluation, target_evaluation
    )
    shared_prediction = np.stack(
        [
            bank.shared_operator.apply(image)
            for image in development_input_evaluation
        ]
    )
    shared_loss = _rmse(shared_prediction, target_evaluation)
    oracle_loss = leave_one_case_oracle(loss_matrix, operator_sources)
    random_loss = leave_one_case_random(loss_matrix, operator_sources)
    _medoid_indices, medoid_loss = leave_one_case_medoid(
        loss_matrix, operator_sources
    )
    global_query = content_descriptor(
        development_input_evaluation,
        development_cells,
        include_spatial_cells=False,
    )
    spatial_query = content_descriptor(
        development_input_evaluation,
        development_cells,
        include_spatial_cells=True,
    )
    bank_global = global_query[eligible_indices]
    bank_spatial = spatial_query[eligible_indices]
    _global_indices, global_loss = leave_one_case_photometric(
        bank_global, global_query, loss_matrix, operator_sources
    )
    _spatial_indices, spatial_loss = leave_one_case_photometric(
        bank_spatial, spatial_query, loss_matrix, operator_sources
    )
    signatures = operator_signature_matrix(
        operators, bank.shared_operator
    )
    structure_all = bool(bank.report["gate_results"]["structure_all"])
    rank_reports: dict[str, Any] = {}
    selected_rank: int | None = None
    selected_development: Any | None = None
    for rank in config["representations"]["candidate_pca_ranks"]:
        print(f"evaluating development rank {rank}", flush=True)
        rank_report, selection = _development_rank(
            rank=int(rank),
            query_raw=spatial_query,
            operator_raw=signatures,
            loss_matrix=loss_matrix,
            shared_loss=shared_loss,
            oracle_loss=oracle_loss,
            medoid_loss=medoid_loss,
            spatial_loss=spatial_loss,
            random_loss=random_loss,
            operator_source_indices=operator_sources,
            structure_all=structure_all,
            config=config,
        )
        rank_reports[str(rank)] = rank_report
        print(
            f"development rank {rank} pass="
            f"{rank_report['all_gates_pass']}",
            flush=True,
        )
        if rank_report["all_gates_pass"]:
            selected_rank = int(rank)
            selected_development = selection
            break
    development_pass = selected_rank is not None
    fresh_report: dict[str, Any] | None = None
    fresh_payloads_accessed = 0
    if development_pass:
        print(
            "development gate passed; opening fresh confirmatory identities",
            flush=True,
        )
        (
            _fresh_input_fit,
            _fresh_target_fit,
            fresh_input_evaluation,
            fresh_target_evaluation,
            fresh_cells,
        ) = _load_partition_samples(
            contract,
            grouped,
            fresh_ids,
            config=config,
            progress_label="fresh-confirmatory",
        )
        if not np.array_equal(development_cells, fresh_cells):
            raise RuntimeError("Z1 spatial cell IDs differ across partitions")
        fresh_report = _fresh_evaluation(
            development_query_raw=spatial_query,
            development_global=bank_global,
            development_spatial=bank_spatial,
            operator_raw=signatures,
            development_loss_matrix=loss_matrix,
            operator_source_indices=operator_sources,
            selected_rank=selected_rank,
            ood_threshold=float(selected_development.ood_threshold),
            fresh_input_evaluation=fresh_input_evaluation,
            fresh_target_evaluation=fresh_target_evaluation["classneg"],
            fresh_cell_ids=fresh_cells,
            operators=operators,
            shared_operator=bank.shared_operator,
            structure_all=structure_all,
            config=config,
        )
        fresh_payloads_accessed = len(fresh_ids) * 2
        decision = fresh_report["decision_branch_before_repeat"]
    else:
        print(
            "development gate failed; fresh confirmatory targets remain sealed",
            flush=True,
        )
        decision = "development_applicability_fail"
    payloads_read = len(development_ids) * 2 + fresh_payloads_accessed
    expected_if_open = int(
        config["dataset"]["selected_payload_image_count_expected"]
    )
    if development_pass and payloads_read != expected_if_open:
        raise FilmSetManifestError("Z1 opened payload count differs")
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "activation_evidence": {
            "z0_decision_sha256": decision_sha256,
            "z0_report_sha256": config["parent_evidence"][
                "required_z0_report_sha256"
            ],
            "z0_classneg_branch": config["parent_evidence"][
                "required_z0_branch"
            ],
            "z0_velvia_loaded": False,
            "w2f1_output_only_allowed": False,
        },
        "dataset": {
            "dataset_id": config["dataset"]["dataset_id"],
            "archive_version": config["dataset"]["archive_version"],
            "internal_manifest_sha256": config["dataset"][
                "internal_dev_manifest_sha256"
            ],
            "final_manifest_sha256_verified": config["dataset"][
                "forbidden_final_manifest_sha256"
            ],
            "development_identity_count": len(development_ids),
            "fresh_confirmatory_identity_count_accessed": (
                len(fresh_ids) if development_pass else 0
            ),
            "selected_payload_image_count": payloads_read,
            "fresh_target_payloads_accessed": (
                len(fresh_ids) if development_pass else 0
            ),
            "z0_confirmatory_payloads_reused": 0,
            "unselected_payload_images_read": 0,
            "final_manifest_payload_rows_parsed": 0,
            "full_raster_outputs_generated": 0,
        },
        "partition": {
            "algorithm": partition["algorithm"],
            "seed": int(partition["seed"]),
            "development_partition_sha256": _ids_sha256(
                development_partition
            ),
            "confirmatory_partition_sha256": _ids_sha256(
                confirmatory_partition
            ),
            "stress_partition_sha256": _ids_sha256(stress_partition),
            "selected_development_sha256": _ids_sha256(development_ids),
            "sealed_z0_confirmatory_sha256": _ids_sha256(z0_ids),
            "fresh_confirmatory_sha256": _ids_sha256(fresh_ids),
            "development_fresh_intersection": 0,
            "z0_fresh_intersection": 0,
        },
        "sampling": {
            **config["sampling"],
            "fit_evaluation_indices_disjoint": True,
            "same_coordinates_across_aligned_domains": True,
        },
        "device": {
            "selected": device,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "development": {
            "case_bank": bank.report,
            "eligible_case_count": len(operators),
            "operator_source_indices": operator_sources.tolist(),
            "operator_signature_dimension": signatures.shape[1],
            "controls": {
                "global_photometric_rmse": _summary(global_loss),
                "spatial_photometric_rmse": _summary(spatial_loss),
                "random_expected_rmse": _summary(random_loss),
                "fixed_medoid_rmse": _summary(medoid_loss),
            },
            "rank_results": rank_reports,
            "selected_rank": selected_rank,
            "development_gate_pass": development_pass,
        },
        "fresh_confirmatory": fresh_report,
        "decision_branch_before_repeat": decision,
        "automatic_visual_shortlist_generated": False,
        "neural_or_direct_rgb_model_fit": False,
        "current_stock_pixels_accessed": False,
        "current_stock_training_or_operator_fitting_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2z1_asymmetric_applicability_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decision_path = (
        ROOT / config["parent_evidence"]["z0_decision_path"]
    ).resolve()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    report = run_experiment(
        config,
        decision,
        config_sha256=_sha256_file(config_path),
        decision_sha256=_sha256_file(decision_path),
        software_commit=_software_commit(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
