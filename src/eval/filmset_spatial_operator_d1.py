"""Fixed spatial explicit-operator bank for the FilmSet ClassNeg control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.roll2film.filmset_recipe_explainability import evaluate_flow_structure
from src.roll2film.hierarchical_colour_coupling import (
    fit_paired_cube_diffeomorphic_flow,
)

SCHEMA = "neuro-film.u5-r2cham3-filmset-spatial-operator-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham3-filmset-spatial-operator-d1-result.v1"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported CHAM3 contract")
    if payload["dataset"]["domain"] != "classneg":
        raise ValueError("CHAM3 domain drift")
    if payload["sampling"]["grid_rows"] * payload["sampling"]["grid_columns"] != 16:
        raise ValueError("CHAM3 fixed grid drift")
    return payload


def _fit(source: np.ndarray, target: np.ndarray, settings: Mapping[str, Any], seed: int, device: str):
    operator, _ = fit_paired_cube_diffeomorphic_flow(
        source,
        target,
        axis_size=int(settings["axis_size"]),
        integration_steps=int(settings["integration_steps"]),
        coefficient_vector_norm_cap=float(settings["coefficient_vector_norm_cap"]),
        steps=int(settings["steps"]),
        learning_rate=float(settings["learning_rate"]),
        coefficient_l2=float(settings["coefficient_l2"]),
        velocity_smoothness_l2=float(settings["velocity_smoothness_l2"]),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
        seed=seed,
        device=device,
        deterministic_algorithms=bool(settings["deterministic_algorithms"]),
        optimization_dtype=str(settings["optimization_dtype"]),
    )
    return operator


def _apply_cells(operators, values: np.ndarray, cell_ids: np.ndarray, *, offset: int = 0) -> np.ndarray:
    output = np.empty_like(values)
    count = len(operators)
    for cell in range(count):
        selected = cell_ids == cell
        output[:, selected] = np.stack(
            [operators[(cell + offset) % count].apply(row[selected]) for row in values]
        )
    return output


def _rmse(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((candidate - target) ** 2, axis=(1, 2)))


def evaluate_arrays(
    contract: Mapping[str, Any],
    development_input: np.ndarray,
    development_target: np.ndarray,
    confirmatory_input: np.ndarray,
    confirmatory_target: np.ndarray,
    cell_ids: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    settings = contract["operator"]
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    cells = np.unique(cell_ids)
    if not np.array_equal(cells, np.arange(16)):
        raise ValueError("CHAM3 cell inventory drift")
    shared = _fit(
        development_input.reshape(-1, 3),
        development_target.reshape(-1, 3),
        settings,
        int(settings["shared_seed"]),
        device,
    )
    operators = tuple(
        _fit(
            development_input[:, cell_ids == cell].reshape(-1, 3),
            development_target[:, cell_ids == cell].reshape(-1, 3),
            settings,
            int(settings["cell_seed_base"]) + int(cell),
            device,
        )
        for cell in cells
    )
    shared_out = np.stack([shared.apply(row) for row in confirmatory_input])
    candidate = _apply_cells(operators, confirmatory_input, cell_ids)
    wrong = _apply_cells(
        operators,
        confirmatory_input,
        cell_ids,
        offset=int(contract["controls"]["cyclic_wrong_cell_offset"]),
    )
    identity_rmse = _rmse(confirmatory_input, confirmatory_target)
    shared_rmse = _rmse(shared_out, confirmatory_target)
    candidate_rmse = _rmse(candidate, confirmatory_target)
    wrong_rmse = _rmse(wrong, confirmatory_target)
    improvement = 1.0 - candidate_rmse / np.maximum(shared_rmse, 1e-30)
    wrong_improvement = 1.0 - candidate_rmse / np.maximum(wrong_rmse, 1e-30)
    structure = evaluate_flow_structure(
        [shared, *operators],
        coefficient_cap=float(settings["coefficient_vector_norm_cap"]),
    )
    metrics = {
        "row_count": len(confirmatory_input),
        "candidate_rmse_mean": float(np.mean(candidate_rmse)),
        "shared_rmse_mean": float(np.mean(shared_rmse)),
        "cyclic_wrong_cell_rmse_mean": float(np.mean(wrong_rmse)),
        "identity_rmse_mean": float(np.mean(identity_rmse)),
        "improvement_over_shared_fraction": float(1.0 - np.mean(candidate_rmse) / np.mean(shared_rmse)),
        "win_fraction_over_shared": float(np.mean(candidate_rmse < shared_rmse)),
        "worst_improvement_over_shared_fraction": float(np.min(improvement)),
        "improvement_over_cyclic_wrong_cell_fraction": float(1.0 - np.mean(candidate_rmse) / np.mean(wrong_rmse)),
        "median_row_improvement_over_shared_fraction": float(np.median(improvement)),
        "median_row_improvement_over_wrong_fraction": float(np.median(wrong_improvement)),
    }
    gates = contract["gates"]
    structure_pass = (
        structure["minimum_output"] >= gates["minimum_output"]
        and structure["maximum_output"] <= gates["maximum_output"]
        and structure["minimum_jacobian_determinant"] > gates["minimum_jacobian_determinant_exclusive"]
        and structure["maximum_jacobian_spectral_norm"] <= gates["maximum_jacobian_spectral_norm"]
        and structure["maximum_inverse_error"] <= gates["maximum_inverse_error"]
        and structure["maximum_replay_error"] <= gates["maximum_replay_error"]
        and structure["maximum_coefficient_vector_norm"] <= gates["maximum_coefficient_vector_norm"]
    )
    checks = {
        "mean_improvement": metrics["improvement_over_shared_fraction"] >= gates["minimum_improvement_over_shared_fraction"],
        "wins": metrics["win_fraction_over_shared"] >= gates["minimum_win_fraction_over_shared"],
        "tail": metrics["worst_improvement_over_shared_fraction"] >= gates["minimum_worst_improvement_over_shared_fraction"],
        "wrong_cell": metrics["improvement_over_cyclic_wrong_cell_fraction"] >= gates["minimum_improvement_over_cyclic_wrong_cell_fraction"],
        "accuracy": metrics["candidate_rmse_mean"] <= gates["maximum_mean_rmse"],
        "structure": structure_pass,
    }
    passed = all(checks.values())
    rows = [
        {
            "index": index,
            "identity_rmse": float(identity_rmse[index]),
            "shared_rmse": float(shared_rmse[index]),
            "candidate_rmse": float(candidate_rmse[index]),
            "cyclic_wrong_cell_rmse": float(wrong_rmse[index]),
            "improvement_over_shared_fraction": float(improvement[index]),
        }
        for index in range(len(candidate_rmse))
    ]
    return {
        "rows": rows,
        "metrics": metrics,
        "structure": structure,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
    }


def finalize_report(contract: Mapping[str, Any], result: Mapping[str, Any], *, parent_ids: Mapping[str, str], dataset: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(canonical_json(contract)).hexdigest(),
        "parent_ids": dict(parent_ids),
        "dataset": dict(dataset),
        **result,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(canonical_json(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
