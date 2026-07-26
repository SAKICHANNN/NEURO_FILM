"""Clean-room hierarchical colour coupling for explicit-flow research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
    _integrate_torch,
    _smoothness_loss,
)


@dataclass(frozen=True)
class CoupledColourPairs:
    """One deterministic, non-reusing pseudo-pair set."""

    source: np.ndarray
    target: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray

    def __post_init__(self) -> None:
        source = _validate_rgb_rows(self.source, name="source")
        target = _validate_rgb_rows(self.target, name="target")
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        target_indices = np.asarray(self.target_indices, dtype=np.int64)
        if (
            source.shape != target.shape
            or source_indices.shape != (len(source),)
            or target_indices.shape != (len(target),)
            or len(np.unique(source_indices)) != len(source_indices)
            or len(np.unique(target_indices)) != len(target_indices)
        ):
            raise ValueError("coupled pairs must be equal, indexed and non-reusing")
        object.__setattr__(self, "source", source.copy())
        object.__setattr__(self, "target", target.copy())
        object.__setattr__(self, "source_indices", source_indices.copy())
        object.__setattr__(self, "target_indices", target_indices.copy())


def _validate_rgb_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not len(array)
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError(f"{name} must be a non-empty finite [0,1] Nx3 array")
    return array


def _random_index_pairs(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(source_indices), len(target_indices))
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    source_order = rng.permutation(source_indices)[:count]
    target_order = rng.permutation(target_indices)[:count]
    return source_order, target_order


def random_colour_coupling(
    source: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
) -> CoupledColourPairs:
    """Pair two colour samples by independent frozen permutations."""

    source_array = _validate_rgb_rows(source, name="source")
    target_array = _validate_rgb_rows(target, name="target")
    source_indices, target_indices = _random_index_pairs(
        np.arange(len(source_array), dtype=np.int64),
        np.arange(len(target_array), dtype=np.int64),
        rng=np.random.default_rng(seed),
    )
    return CoupledColourPairs(
        source_array[source_indices],
        target_array[target_indices],
        source_indices,
        target_indices,
    )


def hierarchical_colour_coupling(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_depth: int,
    seed: int,
) -> CoupledColourPairs:
    """Apply the published mean-centred RGB-octant coupling pseudocode."""

    source_array = _validate_rgb_rows(source, name="source")
    target_array = _validate_rgb_rows(target, name="target")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    rng = np.random.default_rng(seed)

    def recurse(
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        depth: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if (
            depth == maximum_depth
            or len(source_indices) == 0
            or len(target_indices) == 0
        ):
            left, right = _random_index_pairs(
                source_indices, target_indices, rng=rng
            )
            return [] if len(left) == 0 else [(left, right)]
        source_offsets = (
            source_array[source_indices]
            - np.mean(source_array[source_indices], axis=0, keepdims=True)
        )
        target_offsets = (
            target_array[target_indices]
            - np.mean(target_array[target_indices], axis=0, keepdims=True)
        )
        source_octants = np.sum(
            (source_offsets >= 0.0) * np.array([1, 2, 4]), axis=1
        )
        target_octants = np.sum(
            (target_offsets >= 0.0) * np.array([1, 2, 4]), axis=1
        )
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for octant in range(8):
            pairs.extend(
                recurse(
                    source_indices[source_octants == octant],
                    target_indices[target_octants == octant],
                    depth + 1,
                )
            )
        return pairs

    pair_groups = recurse(
        np.arange(len(source_array), dtype=np.int64),
        np.arange(len(target_array), dtype=np.int64),
        0,
    )
    if pair_groups:
        source_indices = np.concatenate([pair[0] for pair in pair_groups])
        target_indices = np.concatenate([pair[1] for pair in pair_groups])
    else:
        source_indices, target_indices = _random_index_pairs(
            np.arange(len(source_array), dtype=np.int64),
            np.arange(len(target_array), dtype=np.int64),
            rng=rng,
        )
    return CoupledColourPairs(
        source_array[source_indices],
        target_array[target_indices],
        source_indices,
        target_indices,
    )


def _bounded_grid(raw: torch.Tensor, cap: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    return cap * raw / (1.0 + norm)


def fit_paired_cube_diffeomorphic_flow(
    source: np.ndarray,
    target: np.ndarray,
    *,
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
    optimization_dtype: str = "float32",
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, float]]:
    """Fit the safe explicit flow to constructed RGB pairs."""

    source_array = _validate_rgb_rows(source, name="source")
    target_array = _validate_rgb_rows(target, name="target")
    if source_array.shape != target_array.shape:
        raise ValueError("source and target pairs must have equal shape")
    if (
        axis_size < 2
        or integration_steps < 1
        or coefficient_vector_norm_cap <= 0.0
        or steps < 1
        or learning_rate <= 0.0
        or coefficient_l2 < 0.0
        or velocity_smoothness_l2 < 0.0
        or gradient_clip_norm <= 0.0
        or optimization_dtype not in {"float32", "float64"}
    ):
        raise ValueError("invalid paired-flow fit settings")
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    dtype = torch.float32 if optimization_dtype == "float32" else torch.float64
    source_tensor = torch.as_tensor(source_array, dtype=dtype, device=device)
    target_tensor = torch.as_tensor(target_array, dtype=dtype, device=device)
    raw = torch.zeros(
        (axis_size, axis_size, axis_size, 3),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    with torch.no_grad():
        initial = float(torch.mean((source_tensor - target_tensor) ** 2).cpu())
    final_objective = np.inf
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        grid = _bounded_grid(raw, coefficient_vector_norm_cap)
        prediction = _integrate_torch(
            source_tensor, grid, integration_steps=integration_steps
        )
        match = torch.mean((prediction - target_tensor) ** 2)
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
        prediction = _integrate_torch(
            source_tensor, grid, integration_steps=integration_steps
        )
        final_match = float(torch.mean((prediction - target_tensor) ** 2).cpu())
    return (
        CubeDiffeomorphicColourFlow(
            grid.detach().cpu().numpy(), integration_steps=integration_steps
        ),
        {
            "initial_pair_mse": initial,
            "final_pair_mse": final_match,
            "final_regularized_objective": final_objective,
        },
    )


__all__ = [
    "CoupledColourPairs",
    "fit_paired_cube_diffeomorphic_flow",
    "hierarchical_colour_coupling",
    "random_colour_coupling",
]
