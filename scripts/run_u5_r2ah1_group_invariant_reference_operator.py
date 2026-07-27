#!/usr/bin/env python
"""Run frozen U5.R2AH1D group-invariant operator development."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    finite_difference_jacobians,
)
from src.roll2film.group_invariant_reference_operator import (  # noqa: E402
    HierarchicalReferenceOperatorPredictor,
    ReferenceEpisodePopulation,
    apply_velocity_grids_torch,
    canonicalize_reference_groups,
    generate_base_direction_grids,
    generate_episode_population,
    generate_fixed_content_controls,
    generate_owner_strength_fixture,
    parameter_state_sha256,
    vicreg_terms,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _population_sha256(population: ReferenceEpisodePopulation) -> str:
    digest = hashlib.sha256()
    for name in (
        "references",
        "target_grids",
        "direction_ids",
        "strengths",
        "content_labels",
        "nuisance_labels",
    ):
        array = np.ascontiguousarray(getattr(population, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _model_from_config(
    config: dict[str, Any],
) -> HierarchicalReferenceOperatorPredictor:
    model = config["model"]
    renderer = config["renderer"]
    return HierarchicalReferenceOperatorPredictor(
        grid_axis_size=int(renderer["velocity_grid_axis_size"]),
        point_hidden_width=int(model["point_hidden_width"]),
        reference_embedding_width=int(model["reference_embedding_width"]),
        group_hidden_width=int(model["group_hidden_width"]),
        maximum_vector_norm=float(renderer["maximum_vector_norm_per_grid_node"]),
        content_family_count=int(config["content_population"]["family_count"]),
        nuisance_family_count=len(config["nuisance_population"]["families"]),
    )


def _render_points_tensor(
    grids: torch.Tensor,
    points: torch.Tensor,
    *,
    integration_steps: int,
) -> torch.Tensor:
    expanded = points[None].expand(len(grids), -1, -1)
    return apply_velocity_grids_torch(
        expanded,
        grids,
        integration_steps=integration_steps,
    )


def _train(
    config: dict[str, Any],
    population: ReferenceEpisodePopulation,
    *,
    device: torch.device,
) -> tuple[HierarchicalReferenceOperatorPredictor, dict[str, float]]:
    training = config["training"]
    torch.manual_seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    model = _model_from_config(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > int(config["model"]["maximum_trainable_parameter_count"]):
        raise RuntimeError("model exceeds frozen parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(training["seed"]))
    batch_size = int(training["batch_size"])
    identity_fraction = float(
        config["operator_population"]["identity_episode_fraction"]
    )
    integration_steps = int(config["renderer"]["integration_steps"])
    render_points = torch.from_numpy(
        _grid(5).astype(np.float32)
    ).to(device)
    all_target_grids = torch.from_numpy(
        np.array(population.target_grids, dtype=np.float32, copy=True)
    ).to(device)
    with torch.no_grad():
        target_renders = _render_points_tensor(
            all_target_grids,
            render_points,
            integration_steps=integration_steps,
        )
    weights = training["losses"]
    last_losses: dict[str, float] = {}
    model.train()
    for _ in range(int(training["steps"])):
        operator_indices = rng.integers(
            0, population.identity_index, size=batch_size
        )
        identity = rng.random(batch_size) < identity_fraction
        operator_indices[identity] = population.identity_index
        group_indices = rng.integers(
            0, population.references.shape[1], size=batch_size
        )
        references = torch.from_numpy(
            population.references[operator_indices, group_indices]
        ).to(device)
        target_grids = all_target_grids[
            torch.from_numpy(operator_indices).to(device)
        ]
        content_labels = torch.from_numpy(
            population.content_labels[operator_indices, group_indices]
        ).to(device)
        nuisance_labels = torch.from_numpy(
            population.nuisance_labels[operator_indices, group_indices]
        ).to(device)
        result = model(references, gradient_reversal_scale=1.0)
        group_grid_loss = F.mse_loss(result["group_grid"], target_grids)
        reference_grid_loss = F.mse_loss(
            result["reference_grids"],
            target_grids[:, None].expand_as(result["reference_grids"]),
        )
        predicted_render = _render_points_tensor(
            result["group_grid"],
            render_points,
            integration_steps=integration_steps,
        )
        target_render = target_renders[
            torch.from_numpy(operator_indices).to(device)
        ]
        render_loss = F.mse_loss(predicted_render, target_render)
        invariance_loss = torch.mean(
            (
                result["reference_embeddings"]
                - result["group_embedding"][:, None, :]
            )
            ** 2
        )
        variance_loss, covariance_loss = vicreg_terms(
            result["group_embedding"]
        )
        adversary_loss = F.cross_entropy(
            result["content_logits"].reshape(
                -1, result["content_logits"].shape[-1]
            ),
            content_labels.reshape(-1),
        ) + F.cross_entropy(
            result["nuisance_logits"].reshape(
                -1, result["nuisance_logits"].shape[-1]
            ),
            nuisance_labels.reshape(-1),
        )
        loss = (
            float(weights["group_operator_grid_mse"]) * group_grid_loss
            + float(weights["single_reference_operator_grid_mse"])
            * reference_grid_loss
            + float(weights["fixed_5cube_render_mse"]) * render_loss
            + float(weights["same_operator_embedding_invariance"])
            * invariance_loss
            + float(weights["vicreg_variance"]) * variance_loss
            + float(weights["vicreg_covariance"]) * covariance_loss
            + float(weights["content_gradient_reversal"]) * adversary_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        last_losses = {
            "total": float(loss.detach().cpu()),
            "group_grid": float(group_grid_loss.detach().cpu()),
            "reference_grid": float(reference_grid_loss.detach().cpu()),
            "render": float(render_loss.detach().cpu()),
            "invariance": float(invariance_loss.detach().cpu()),
            "variance": float(variance_loss.detach().cpu()),
            "covariance": float(covariance_loss.detach().cpu()),
            "adversary": float(adversary_loss.detach().cpu()),
        }
    model.eval()
    return model, last_losses


def _predict_population(
    model: HierarchicalReferenceOperatorPredictor,
    population: ReferenceEpisodePopulation,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    groups = population.references.reshape(
        -1, *population.references.shape[2:]
    )
    predicted = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(groups), batch_size):
            batch = np.array(
                groups[start : start + batch_size], dtype=np.float32, copy=True
            )
            result = model(torch.from_numpy(batch).to(device))
            predicted.append(result["group_grid"].cpu().numpy())
            embeddings.append(result["reference_embeddings"].cpu().numpy())
    grid_shape = population.target_grids.shape[1:]
    return (
        np.concatenate(predicted).reshape(
            len(population.references),
            population.references.shape[1],
            *grid_shape,
        ),
        np.concatenate(embeddings).reshape(
            len(population.references),
            population.references.shape[1],
            population.references.shape[2],
            -1,
        ),
    )


def _render_grids(
    grids: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float32)
    flat = values.reshape(-1, *values.shape[-4:])
    point_tensor = torch.from_numpy(points.astype(np.float32)).to(device)
    rendered = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            grid_tensor = torch.from_numpy(flat[start : start + batch_size]).to(
                device
            )
            rendered.append(
                _render_points_tensor(
                    grid_tensor,
                    point_tensor,
                    integration_steps=integration_steps,
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(rendered).reshape(*values.shape[:-4], len(points), 3)


def _rmse_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((left - right) ** 2, axis=(-2, -1)))


def _probe_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    train_x = train_embeddings.reshape(-1, train_embeddings.shape[-1])
    test_x = test_embeddings.reshape(-1, test_embeddings.shape[-1])
    train_y = train_labels.reshape(-1)
    test_y = test_labels.reshape(-1)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            batch_size=256,
            random_state=seed,
            early_stopping=False,
        ),
    )
    scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for name, model in (("logistic", logistic), ("mlp", mlp)):
            model.fit(train_x, train_y)
            scores[name] = float(
                balanced_accuracy_score(test_y, model.predict(test_x))
            )
    return scores


def _structural_metrics(
    predicted_grids: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    jacobian_axis = np.linspace(
        0.05,
        0.95,
        int(config["development_evaluation"]["jacobian_grid_axis_size"]),
        dtype=np.float64,
    )
    jacobian_points = np.stack(
        np.meshgrid(jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    roundtrip_points = _grid(7)
    minimum_det = np.inf
    maximum_norm = 0.0
    minimum_output = np.inf
    maximum_output = -np.inf
    maximum_inverse = 0.0
    step = float(config["development_evaluation"]["finite_difference_step"])
    integration_steps = int(config["renderer"]["integration_steps"])
    for grid in predicted_grids.reshape(-1, *predicted_grids.shape[-4:]):
        operator = CubeDiffeomorphicColourFlow(
            np.asarray(grid, dtype=np.float64),
            integration_steps=integration_steps,
        )
        output = operator.apply(roundtrip_points)
        restored = operator.inverse(output)
        jacobians = finite_difference_jacobians(
            operator, jacobian_points, step=step
        )
        minimum_det = min(minimum_det, float(np.min(np.linalg.det(jacobians))))
        maximum_norm = max(
            maximum_norm,
            float(np.max(np.linalg.svd(jacobians, compute_uv=False)[:, 0])),
        )
        minimum_output = min(minimum_output, float(np.min(output)))
        maximum_output = max(maximum_output, float(np.max(output)))
        maximum_inverse = max(
            maximum_inverse, float(np.max(np.abs(restored - roundtrip_points)))
        )
    return {
        "output_minimum": minimum_output,
        "output_maximum": maximum_output,
        "minimum_finite_difference_jacobian_determinant": minimum_det,
        "maximum_finite_difference_jacobian_spectral_norm": maximum_norm,
        "inverse_roundtrip_maximum_absolute_error": maximum_inverse,
    }


def run(config: dict[str, Any], *, config_sha256: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("AH1D frozen contract requires local CUDA")
    device = torch.device("cuda")
    training_population = generate_episode_population(config, split="training")
    development_population = generate_episode_population(
        config, split="development"
    )
    model, final_losses = _train(
        config, training_population, device=device
    )
    train_predictions, train_embeddings = _predict_population(
        model, training_population, device=device
    )
    dev_predictions, dev_embeddings = _predict_population(
        model, development_population, device=device
    )
    integration_steps = int(config["renderer"]["integration_steps"])
    evaluation_points = _grid(
        int(config["development_evaluation"]["operator_grid_axis_size"])
    )
    nonidentity = slice(0, development_population.identity_index)
    predicted_outputs = _render_grids(
        dev_predictions[nonidentity],
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )
    target_outputs_per_instance = _render_grids(
        development_population.target_grids[nonidentity],
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )
    target_outputs = np.repeat(
        target_outputs_per_instance[:, None],
        development_population.references.shape[1],
        axis=1,
    )
    operator_errors = _rmse_rows(predicted_outputs, target_outputs)
    identity_errors = _rmse_rows(
        np.broadcast_to(evaluation_points, target_outputs.shape),
        target_outputs,
    )
    global_grid = np.mean(
        training_population.target_grids[: training_population.identity_index],
        axis=0,
    )
    global_output = _render_grids(
        global_grid[None],
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )[0]
    global_errors = _rmse_rows(
        np.broadcast_to(global_output, target_outputs.shape),
        target_outputs,
    )
    model_median = float(np.median(operator_errors))
    identity_median = float(np.median(identity_errors))
    global_median = float(np.median(global_errors))
    w1_median = 0.06631

    replicate_errors = []
    for instance_outputs in predicted_outputs:
        for left in range(len(instance_outputs)):
            for right in range(left + 1, len(instance_outputs)):
                replicate_errors.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (instance_outputs[left] - instance_outputs[right])
                                ** 2
                            )
                        )
                    )
                )
    identity_prediction = dev_predictions[development_population.identity_index]
    identity_outputs = _render_grids(
        identity_prediction,
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )
    identity_reference_errors = _rmse_rows(
        identity_outputs,
        np.broadcast_to(evaluation_points, identity_outputs.shape),
    )

    content_probes = _probe_scores(
        train_embeddings,
        training_population.content_labels,
        dev_embeddings,
        development_population.content_labels,
        seed=int(config["training"]["seed"]),
    )
    nuisance_probes = _probe_scores(
        train_embeddings,
        training_population.nuisance_labels,
        dev_embeddings,
        development_population.nuisance_labels,
        seed=int(config["training"]["seed"]) + 1,
    )

    base_dev = generate_base_direction_grids(config, split="development")
    fixed_references = generate_fixed_content_controls(
        config, base_grids=base_dev
    )
    with torch.no_grad():
        fixed_prediction = model(
            torch.from_numpy(fixed_references).to(device)
        )["group_grid"].cpu().numpy()
    fixed_outputs = _render_grids(
        fixed_prediction,
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )
    fixed_targets = _render_grids(
        base_dev,
        points=evaluation_points,
        integration_steps=integration_steps,
        device=device,
    )
    fixed_distances = np.sqrt(
        np.mean(
            (fixed_outputs[:, None] - fixed_targets[None]) ** 2,
            axis=(-2, -1),
        )
    )
    fixed_accuracy = float(
        np.mean(np.argmin(fixed_distances, axis=1) == np.arange(len(base_dev)))
    )

    owner_references, _, owner_strengths = generate_owner_strength_fixture(config)
    with torch.no_grad():
        owner_predictions = model(
            torch.from_numpy(owner_references).to(device)
        )["group_grid"].cpu().numpy()
    owner_flat = owner_predictions.reshape(len(owner_predictions), -1)
    owner_norms = np.linalg.norm(owner_flat, axis=1)
    owner_unit = owner_flat / np.maximum(owner_norms[:, None], 1e-15)
    owner_cosines = owner_unit @ owner_unit.T
    owner_minimum_cosine = float(
        np.min(owner_cosines[np.triu_indices(len(owner_cosines), k=1)])
    )
    owner_spearman = float(spearmanr(owner_strengths, owner_norms).statistic)

    canonical = np.array(
        development_population.references[
            : min(8, len(development_population.references)), 0
        ],
        dtype=np.float32,
        copy=True,
    )
    permuted = canonical[:, [2, 0, 3, 1]][:, :, ::-1]
    recanonicalized = canonicalize_reference_groups(permuted)
    with torch.no_grad():
        first_permutation = model(torch.from_numpy(canonical).to(device))[
            "group_grid"
        ].cpu().numpy()
        second_permutation = model(
            torch.from_numpy(recanonicalized).to(device)
        )["group_grid"].cpu().numpy()
    permutation_error = float(
        np.max(np.abs(first_permutation - second_permutation))
    )
    replay = _model_from_config(config).to(device)
    replay.load_state_dict(copy.deepcopy(model.state_dict()))
    replay.eval()
    with torch.no_grad():
        serialization_error = float(
            torch.max(
                torch.abs(
                    replay(torch.from_numpy(canonical).to(device))["group_grid"]
                    - model(torch.from_numpy(canonical).to(device))["group_grid"]
                )
            ).cpu()
        )

    structural = _structural_metrics(
        dev_predictions,
        config=config,
    )
    metrics = {
        "four_reference_operator_rmse_median": model_median,
        "four_reference_operator_rmse_p90": float(
            np.quantile(operator_errors, 0.9)
        ),
        "identity_baseline_operator_rmse_median": identity_median,
        "global_mean_baseline_operator_rmse_median": global_median,
        "w1_historical_four_reference_ridge_rmse_median": w1_median,
        "median_improvement_over_identity_fraction": (
            (identity_median - model_median) / identity_median
        ),
        "median_improvement_over_global_mean_fraction": (
            (global_median - model_median) / global_median
        ),
        "median_improvement_over_w1_ridge_fraction": (
            (w1_median - model_median) / w1_median
        ),
        "same_look_replicate_operator_rmse_median": float(
            np.median(replicate_errors)
        ),
        "identity_reference_grid_rmse_median": float(
            np.median(identity_reference_errors)
        ),
        "content_probe_balanced_accuracy": content_probes,
        "nuisance_probe_balanced_accuracy": nuisance_probes,
        "same_content_different_look_direction_accuracy": fixed_accuracy,
        "owner_strength_minimum_direction_cosine": owner_minimum_cosine,
        "owner_strength_spearman": owner_spearman,
        "serialization_replay_maximum_absolute_error": serialization_error,
        "set_permutation_maximum_absolute_error": permutation_error,
        **structural,
    }
    gates = config["gates"]
    checks = {
        "operator_rmse": (
            metrics["four_reference_operator_rmse_median"]
            <= gates["four_reference_operator_rmse_median_maximum"]
            and metrics["four_reference_operator_rmse_p90"]
            <= gates["four_reference_operator_rmse_p90_maximum"]
        ),
        "beats_identity": (
            metrics["median_improvement_over_identity_fraction"]
            >= gates["minimum_median_improvement_over_identity_fraction"]
        ),
        "beats_global_mean": (
            metrics["median_improvement_over_global_mean_fraction"]
            >= gates["minimum_median_improvement_over_global_mean_fraction"]
        ),
        "beats_w1_ridge": (
            metrics["median_improvement_over_w1_ridge_fraction"]
            >= gates["minimum_median_improvement_over_w1_ridge_fraction"]
        ),
        "same_look_replication": (
            metrics["same_look_replicate_operator_rmse_median"]
            <= gates["same_look_replicate_operator_rmse_median_maximum"]
        ),
        "identity_rejection": (
            metrics["identity_reference_grid_rmse_median"]
            <= gates["identity_reference_grid_rmse_median_maximum"]
        ),
        "content_probe": (
            max(content_probes.values())
            <= gates["content_family_balanced_accuracy_maximum"]
        ),
        "nuisance_probe": (
            max(nuisance_probes.values())
            <= gates["nuisance_family_balanced_accuracy_maximum"]
        ),
        "fixed_content_look_separation": (
            fixed_accuracy
            >= gates["same_content_different_look_direction_accuracy_minimum"]
        ),
        "owner_strength_path": (
            owner_minimum_cosine
            >= gates["owner_strength_direction_cosine_minimum"]
            and owner_spearman >= gates["owner_strength_spearman_minimum"]
        ),
        "range": (
            structural["output_minimum"] >= gates["output_minimum"]
            and structural["output_maximum"] <= gates["output_maximum"]
        ),
        "jacobian": (
            structural["minimum_finite_difference_jacobian_determinant"]
            >= gates["minimum_finite_difference_jacobian_determinant"]
            and structural["maximum_finite_difference_jacobian_spectral_norm"]
            <= gates["maximum_finite_difference_jacobian_spectral_norm"]
        ),
        "inverse": (
            structural["inverse_roundtrip_maximum_absolute_error"]
            <= gates["inverse_roundtrip_maximum_absolute_error"]
        ),
        "serialization": (
            serialization_error
            <= gates["serialization_replay_maximum_absolute_error"]
        ),
        "set_permutation": (
            permutation_error <= gates["set_permutation_maximum_absolute_error"]
        ),
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "training_population_sha256": _population_sha256(training_population),
        "development_population_sha256": _population_sha256(
            development_population
        ),
        "parameter_count": parameter_count,
        "parameter_state_sha256": parameter_state_sha256(model),
        "final_training_losses": final_losses,
        "metrics": metrics,
        "checks": checks,
        "all_development_checks_passed": all(checks.values()),
        "reserved_confirmation_accessed": False,
        "w1_reserved_confirmation_accessed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ah1_group_invariant_reference_operator_development_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2ah1_group_invariant_reference_operator_development_v1/report.json"
        ),
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run(json.loads(config_bytes), config_sha256=_sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_development_checks_passed": report[
                    "all_development_checks_passed"
                ],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_development_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
