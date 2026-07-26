"""Unpaired distribution objectives for fitting explicit colour flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
    _integrate_torch,
    _smoothness_loss,
)


@dataclass(frozen=True)
class DistributionLossAssets:
    kind: str
    tensors: dict[str, np.ndarray]


def make_distribution_loss_assets(spec: dict[str, Any]) -> DistributionLossAssets:
    rng = np.random.default_rng(int(spec.get("projection_seed", spec.get("feature_seed"))))
    kind = str(spec["kind"])
    if kind == "fixed_projection_sorted_quantile":
        projections = rng.normal(size=(int(spec["projection_count"]), 3))
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        return DistributionLossAssets(kind, {"projections": projections})
    if kind == "fixed_random_fourier_mmd":
        frequencies = []
        for bandwidth in spec["bandwidths"]:
            frequencies.append(
                rng.normal(
                    scale=1.0 / float(bandwidth),
                    size=(int(spec["frequencies_per_bandwidth"]), 3),
                )
            )
        frequency = np.concatenate(frequencies)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=len(frequency))
        return DistributionLossAssets(
            kind, {"frequencies": frequency, "phases": phase}
        )
    raise ValueError(f"unsupported distribution loss: {kind}")


def distribution_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    assets: DistributionLossAssets,
) -> torch.Tensor:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have equal Nx3 shape")
    if assets.kind == "fixed_projection_sorted_quantile":
        projections = torch.as_tensor(
            assets.tensors["projections"], dtype=source.dtype, device=source.device
        )
        left = torch.sort(source @ projections.T, dim=0).values
        right = torch.sort(target @ projections.T, dim=0).values
        return torch.mean((left - right) ** 2)
    if assets.kind == "fixed_random_fourier_mmd":
        frequency = torch.as_tensor(
            assets.tensors["frequencies"], dtype=source.dtype, device=source.device
        )
        phase = torch.as_tensor(
            assets.tensors["phases"], dtype=source.dtype, device=source.device
        )
        scale = np.sqrt(2.0 / len(frequency))
        left = scale * torch.cos(source @ frequency.T + phase)
        right = scale * torch.cos(target @ frequency.T + phase)
        return torch.mean((torch.mean(left, dim=0) - torch.mean(right, dim=0)) ** 2)
    raise ValueError(f"unsupported distribution loss: {assets.kind}")


def grouped_distribution_loss(
    sources: list[torch.Tensor] | tuple[torch.Tensor, ...],
    targets: list[torch.Tensor] | tuple[torch.Tensor, ...],
    assets: DistributionLossAssets,
) -> torch.Tensor:
    """Return an equal-weight loss over corresponding distribution groups."""

    if not sources or len(sources) != len(targets):
        raise ValueError("sources and targets must contain equal non-empty groups")
    losses = [
        distribution_loss(source, target, assets)
        for source, target in zip(sources, targets, strict=True)
    ]
    if len(losses) == 1:
        # Preserve the original single-distribution arithmetic exactly.
        return losses[0]
    return torch.mean(torch.stack(losses))


def _bounded_grid(raw: torch.Tensor, cap: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    return cap * raw / (1.0 + norm)


def _validate_distribution_groups(
    sources: list[np.ndarray] | tuple[np.ndarray, ...],
    targets: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not sources or len(sources) != len(targets):
        raise ValueError("sources and targets must contain equal non-empty groups")
    source_arrays, target_arrays = [], []
    for source, target in zip(sources, targets, strict=True):
        source_array = np.asarray(source, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if (
            source_array.shape != target_array.shape
            or source_array.ndim != 2
            or source_array.shape[1] != 3
            or not np.all(np.isfinite(source_array))
            or not np.all(np.isfinite(target_array))
            or np.any(source_array < 0.0)
            or np.any(source_array > 1.0)
            or np.any(target_array < 0.0)
            or np.any(target_array > 1.0)
        ):
            raise ValueError(
                "unpaired distribution groups must be equal finite [0,1] Nx3 arrays"
            )
        source_arrays.append(source_array)
        target_arrays.append(target_array)
    return source_arrays, target_arrays


def _fit_unpaired_distribution_flow_groups(
    sources: list[np.ndarray] | tuple[np.ndarray, ...],
    targets: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    loss_assets: DistributionLossAssets,
    axis_size: int,
    integration_steps: int,
    coefficient_vector_norm_cap: float,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    velocity_smoothness_l2: float,
    gradient_clip_norm: float,
    seed: int,
    device: str,
    deterministic_algorithms: bool,
    optimization_dtype: str,
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, float]]:
    source_arrays, target_arrays = _validate_distribution_groups(sources, targets)
    if optimization_dtype not in {"float32", "float64"}:
        raise ValueError("optimization_dtype must be float32 or float64")
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    dtype = torch.float32 if optimization_dtype == "float32" else torch.float64
    source_tensors = [
        torch.as_tensor(source, dtype=dtype, device=device)
        for source in source_arrays
    ]
    target_tensors = [
        torch.as_tensor(target, dtype=dtype, device=device)
        for target in target_arrays
    ]
    raw = torch.zeros(
        (axis_size, axis_size, axis_size, 3),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    with torch.no_grad():
        initial = float(
            grouped_distribution_loss(
                source_tensors, target_tensors, loss_assets
            ).cpu()
        )
    final_objective = np.inf
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        grid = _bounded_grid(raw, coefficient_vector_norm_cap)
        predictions = [
            _integrate_torch(source, grid, integration_steps=integration_steps)
            for source in source_tensors
        ]
        match = grouped_distribution_loss(
            predictions, target_tensors, loss_assets
        )
        objective = (
            match
            + coefficient_l2 * torch.mean(grid**2)
            + velocity_smoothness_l2 * _smoothness_loss(grid)
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_([raw], gradient_clip_norm)
        optimizer.step()
        final_objective = float(objective.detach().cpu())
    with torch.no_grad():
        grid = _bounded_grid(raw, coefficient_vector_norm_cap)
        predictions = [
            _integrate_torch(source, grid, integration_steps=integration_steps)
            for source in source_tensors
        ]
        final_match = float(
            grouped_distribution_loss(
                predictions, target_tensors, loss_assets
            ).cpu()
        )
    operator = CubeDiffeomorphicColourFlow(
        grid.detach().cpu().numpy(), integration_steps=integration_steps
    )
    return operator, {
        "initial_distribution_loss": initial,
        "final_distribution_loss": final_match,
        "final_regularized_objective": final_objective,
    }


def fit_unpaired_distribution_flow(
    source: np.ndarray,
    target: np.ndarray,
    *,
    loss_assets: DistributionLossAssets,
    axis_size: int,
    integration_steps: int,
    coefficient_vector_norm_cap: float,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    velocity_smoothness_l2: float,
    gradient_clip_norm: float,
    seed: int,
    device: str,
    deterministic_algorithms: bool,
    optimization_dtype: str = "float64",
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, float]]:
    return _fit_unpaired_distribution_flow_groups(
        [source],
        [target],
        loss_assets=loss_assets,
        axis_size=axis_size,
        integration_steps=integration_steps,
        coefficient_vector_norm_cap=coefficient_vector_norm_cap,
        steps=steps,
        learning_rate=learning_rate,
        coefficient_l2=coefficient_l2,
        velocity_smoothness_l2=velocity_smoothness_l2,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
        deterministic_algorithms=deterministic_algorithms,
        optimization_dtype=optimization_dtype,
    )


def fit_grouped_unpaired_distribution_flow(
    sources: list[np.ndarray] | tuple[np.ndarray, ...],
    targets: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    loss_assets: DistributionLossAssets,
    axis_size: int,
    integration_steps: int,
    coefficient_vector_norm_cap: float,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    velocity_smoothness_l2: float,
    gradient_clip_norm: float,
    seed: int,
    device: str,
    deterministic_algorithms: bool,
    optimization_dtype: str = "float64",
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, float]]:
    """Fit one explicit flow across corresponding unpaired distributions."""

    return _fit_unpaired_distribution_flow_groups(
        sources,
        targets,
        loss_assets=loss_assets,
        axis_size=axis_size,
        integration_steps=integration_steps,
        coefficient_vector_norm_cap=coefficient_vector_norm_cap,
        steps=steps,
        learning_rate=learning_rate,
        coefficient_l2=coefficient_l2,
        velocity_smoothness_l2=velocity_smoothness_l2,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
        deterministic_algorithms=deterministic_algorithms,
        optimization_dtype=optimization_dtype,
    )


def evaluate_distribution_loss(
    operator: CubeDiffeomorphicColourFlow,
    source: np.ndarray,
    target: np.ndarray,
    assets: DistributionLossAssets,
) -> tuple[float, float]:
    source_tensor = torch.as_tensor(source, dtype=torch.float64)
    target_tensor = torch.as_tensor(target, dtype=torch.float64)
    identity = float(distribution_loss(source_tensor, target_tensor, assets))
    transformed = torch.as_tensor(operator.apply(source), dtype=torch.float64)
    fitted = float(distribution_loss(transformed, target_tensor, assets))
    return identity, fitted


def evaluate_grouped_distribution_loss(
    operator: CubeDiffeomorphicColourFlow,
    sources: list[np.ndarray] | tuple[np.ndarray, ...],
    targets: list[np.ndarray] | tuple[np.ndarray, ...],
    assets: DistributionLossAssets,
) -> tuple[float, float]:
    source_arrays, target_arrays = _validate_distribution_groups(sources, targets)
    source_tensors = [
        torch.as_tensor(source, dtype=torch.float64) for source in source_arrays
    ]
    target_tensors = [
        torch.as_tensor(target, dtype=torch.float64) for target in target_arrays
    ]
    identity = float(
        grouped_distribution_loss(source_tensors, target_tensors, assets)
    )
    transformed = [
        torch.as_tensor(operator.apply(source), dtype=torch.float64)
        for source in source_arrays
    ]
    fitted = float(grouped_distribution_loss(transformed, target_tensors, assets))
    return identity, fitted


__all__ = [
    "DistributionLossAssets",
    "distribution_loss",
    "evaluate_distribution_loss",
    "evaluate_grouped_distribution_loss",
    "fit_grouped_unpaired_distribution_flow",
    "fit_unpaired_distribution_flow",
    "grouped_distribution_loss",
    "make_distribution_loss_assets",
]
