"""Group-invariant prediction of bounded explicit colour-flow parameters.

The neural component is discriminative and emits only a stationary velocity
grid. Final RGB values remain the responsibility of CubeDiffeomorphicColourFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from src.roll2film.histogram_case_retrieval import sample_palette
from src.roll2film.palette_score_flow import DiagonalGaussianMixturePalette


def canonicalize_reference_groups(groups: np.ndarray) -> np.ndarray:
    """Lexicographically canonicalize pixels and references exactly."""

    values = np.asarray(groups)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("groups must have finite [B,R,N,3] RGB values")
    result = np.empty_like(values)
    for batch in range(len(values)):
        references = []
        for reference in values[batch]:
            order = np.lexsort(
                (reference[:, 2], reference[:, 1], reference[:, 0])
            )
            references.append(reference[order])
        references_array = np.stack(references)
        flattened = references_array.reshape(len(references_array), -1)
        order = sorted(
            range(len(flattened)),
            key=lambda index: flattened[index].tobytes(),
        )
        result[batch] = references_array[order]
    return result


def radial_tanh_bound(
    raw_grid: torch.Tensor, *, maximum_vector_norm: float
) -> torch.Tensor:
    """Smoothly bound each RGB velocity vector by a strict radial cap."""

    if raw_grid.shape[-1] != 3 or maximum_vector_norm <= 0.0:
        raise ValueError("raw_grid or maximum_vector_norm is invalid")
    norm = torch.linalg.vector_norm(raw_grid, dim=-1, keepdim=True)
    scale = maximum_vector_norm * torch.tanh(norm / maximum_vector_norm)
    return torch.where(
        norm > 0.0,
        raw_grid * (scale / torch.clamp_min(norm, 1e-12)),
        raw_grid,
    )


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class HierarchicalReferenceOperatorPredictor(nn.Module):
    """A canonical-order Deep Sets encoder with a bounded O0 parameter head."""

    def __init__(
        self,
        *,
        grid_axis_size: int = 4,
        point_hidden_width: int = 64,
        reference_embedding_width: int = 64,
        group_hidden_width: int = 128,
        maximum_vector_norm: float = 1.7,
        content_family_count: int = 8,
        nuisance_family_count: int = 4,
    ) -> None:
        super().__init__()
        if (
            grid_axis_size < 2
            or min(
                point_hidden_width,
                reference_embedding_width,
                group_hidden_width,
                content_family_count,
                nuisance_family_count,
            )
            < 1
            or maximum_vector_norm <= 0.0
        ):
            raise ValueError("invalid predictor dimensions")
        self.grid_axis_size = int(grid_axis_size)
        self.maximum_vector_norm = float(maximum_vector_norm)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, point_hidden_width),
            nn.SiLU(),
            nn.Linear(point_hidden_width, point_hidden_width),
            nn.SiLU(),
        )
        self.reference_encoder = nn.Sequential(
            nn.Linear(2 * point_hidden_width, reference_embedding_width),
            nn.SiLU(),
            nn.Linear(reference_embedding_width, reference_embedding_width),
        )
        self.operator_head = nn.Sequential(
            nn.Linear(reference_embedding_width, group_hidden_width),
            nn.SiLU(),
            nn.Linear(
                group_hidden_width,
                grid_axis_size * grid_axis_size * grid_axis_size * 3,
            ),
        )
        self.content_head = nn.Sequential(
            nn.Linear(reference_embedding_width, reference_embedding_width),
            nn.SiLU(),
            nn.Linear(reference_embedding_width, content_family_count),
        )
        self.nuisance_head = nn.Sequential(
            nn.Linear(reference_embedding_width, reference_embedding_width),
            nn.SiLU(),
            nn.Linear(reference_embedding_width, nuisance_family_count),
        )

    def _head(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = self.operator_head(embedding)
        shape = (
            *embedding.shape[:-1],
            self.grid_axis_size,
            self.grid_axis_size,
            self.grid_axis_size,
            3,
        )
        return radial_tanh_bound(
            raw.reshape(shape),
            maximum_vector_norm=self.maximum_vector_norm,
        )

    def forward(
        self,
        references: torch.Tensor,
        *,
        gradient_reversal_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if references.ndim != 4 or references.shape[-1] != 3:
            raise ValueError("references must have shape [B,R,N,3]")
        points = self.point_encoder(references)
        summary = torch.cat(
            (torch.mean(points, dim=2), torch.amax(points, dim=2)), dim=-1
        )
        reference_embeddings = self.reference_encoder(summary)
        group_embedding = torch.mean(reference_embeddings, dim=1)
        reversed_embeddings = _GradientReversal.apply(
            reference_embeddings, gradient_reversal_scale
        )
        return {
            "reference_embeddings": reference_embeddings,
            "group_embedding": group_embedding,
            "group_grid": self._head(group_embedding),
            "reference_grids": self._head(reference_embeddings),
            "content_logits": self.content_head(reversed_embeddings),
            "nuisance_logits": self.nuisance_head(reversed_embeddings),
        }


def _sample_grid_torch_batched(
    rgb: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Trilinearly sample one velocity grid per batch row."""

    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or grid.ndim != 5
        or grid.shape[0] != rgb.shape[0]
        or grid.shape[-1] != 3
        or grid.shape[1] != grid.shape[2]
        or grid.shape[2] != grid.shape[3]
    ):
        raise ValueError("batched RGB/grid shapes are invalid")
    axis_size = int(grid.shape[1])
    coordinates = rgb * float(axis_size - 1)
    indices = torch.arange(
        axis_size, dtype=rgb.dtype, device=rgb.device
    )
    red = torch.clamp(
        1.0 - torch.abs(coordinates[..., 0, None] - indices), min=0.0
    )
    green = torch.clamp(
        1.0 - torch.abs(coordinates[..., 1, None] - indices), min=0.0
    )
    blue = torch.clamp(
        1.0 - torch.abs(coordinates[..., 2, None] - indices), min=0.0
    )
    return torch.einsum(
        "bnr,bng,bnh,brghc->bnc", red, green, blue, grid
    )


def apply_velocity_grids_torch(
    rgb: torch.Tensor,
    velocity_grids: torch.Tensor,
    *,
    integration_steps: int,
) -> torch.Tensor:
    """Differentiably integrate one stationary grid per batch row."""

    if integration_steps < 1:
        raise ValueError("integration_steps must be positive")
    step = 1.0 / integration_steps
    current = rgb
    for _ in range(integration_steps):
        k1 = current * (1.0 - current) * _sample_grid_torch_batched(
            current, velocity_grids
        )
        middle = current + 0.5 * step * k1
        k2 = middle * (1.0 - middle) * _sample_grid_torch_batched(
            middle, velocity_grids
        )
        middle = current + 0.5 * step * k2
        k3 = middle * (1.0 - middle) * _sample_grid_torch_batched(
            middle, velocity_grids
        )
        end = current + step * k3
        k4 = end * (1.0 - end) * _sample_grid_torch_batched(
            end, velocity_grids
        )
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def vicreg_terms(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return variance-hinge and off-diagonal covariance penalties."""

    if embeddings.ndim != 2 or len(embeddings) < 2:
        raise ValueError("embeddings must have shape [B,D], B>=2")
    variance = torch.mean(
        F.relu(1.0 - torch.sqrt(torch.var(embeddings, dim=0) + 1e-4))
    )
    centered = embeddings - torch.mean(embeddings, dim=0, keepdim=True)
    covariance = centered.T @ centered / float(len(embeddings) - 1)
    covariance = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = torch.sum(covariance**2) / embeddings.shape[1]
    return variance, covariance_loss


@dataclass(frozen=True)
class ReferenceEpisodePopulation:
    references: np.ndarray
    target_grids: np.ndarray
    direction_ids: np.ndarray
    strengths: np.ndarray
    content_labels: np.ndarray
    nuisance_labels: np.ndarray
    identity_index: int

    def __post_init__(self) -> None:
        arrays = {
            "references": np.asarray(self.references, dtype=np.float32),
            "target_grids": np.asarray(self.target_grids, dtype=np.float32),
            "direction_ids": np.asarray(self.direction_ids, dtype=np.int64),
            "strengths": np.asarray(self.strengths, dtype=np.float64),
            "content_labels": np.asarray(self.content_labels, dtype=np.int64),
            "nuisance_labels": np.asarray(self.nuisance_labels, dtype=np.int64),
        }
        refs = arrays["references"]
        grids = arrays["target_grids"]
        if (
            refs.ndim != 5
            or refs.shape[-1] != 3
            or grids.ndim != 5
            or len(refs) != len(grids)
            or arrays["direction_ids"].shape != (len(refs),)
            or arrays["strengths"].shape != (len(refs),)
            or arrays["content_labels"].shape != refs.shape[:3]
            or arrays["nuisance_labels"].shape != refs.shape[:3]
            or self.identity_index != len(refs) - 1
            or not np.all(np.isfinite(refs))
            or not np.all(np.isfinite(grids))
            or np.any(refs < 0.0)
            or np.any(refs > 1.0)
        ):
            raise ValueError("episode population is invalid")
        for name, value in arrays.items():
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


def _smooth_random_grids(
    *,
    count: int,
    rng: np.random.Generator,
    axis_size: int,
    smoothing_passes: int,
    minimum_norm: float,
    maximum_norm: float,
) -> np.ndarray:
    grids = []
    for _ in range(count):
        grid = rng.normal(size=(axis_size, axis_size, axis_size, 3))
        for _ in range(smoothing_passes):
            for axis in range(3):
                padded = np.concatenate(
                    (
                        np.take(grid, [0], axis=axis),
                        grid,
                        np.take(grid, [-1], axis=axis),
                    ),
                    axis=axis,
                )
                grid = (
                    0.25 * np.take(padded, np.arange(axis_size), axis=axis)
                    + 0.5
                    * np.take(padded, np.arange(1, axis_size + 1), axis=axis)
                    + 0.25
                    * np.take(padded, np.arange(2, axis_size + 2), axis=axis)
                )
        grid -= np.mean(grid, axis=(0, 1, 2), keepdims=True)
        maximum = float(np.max(np.linalg.norm(grid, axis=-1)))
        target = float(rng.uniform(minimum_norm, maximum_norm))
        grids.append(target * grid / max(maximum, 1e-15))
    return np.stack(grids)


def _palette_score_grids(
    *,
    count: int,
    rng: np.random.Generator,
    axis_size: int,
    maximum_norm: float,
) -> np.ndarray:
    from src.roll2film.histogram_case_retrieval import generate_synthetic_palette
    from src.roll2film.palette_score_flow import palette_score_velocity_grid

    grids = []
    for _ in range(count):
        palette = generate_synthetic_palette(
            rng,
            component_counts=(2, 3, 4),
            weight_dirichlet_alpha=1.5,
            mean_minimum=0.06,
            mean_maximum=0.94,
            standard_deviation_minimum=0.09,
            standard_deviation_maximum=0.24,
        )
        grids.append(
            palette_score_velocity_grid(
                palette,
                axis_size=axis_size,
                coefficient_vector_norm_cap=maximum_norm,
            )
        )
    return np.stack(grids)


def generate_base_direction_grids(
    config: dict[str, Any], *, split: str
) -> np.ndarray:
    """Generate the frozen balanced palette/random O0 direction split."""

    population = config["operator_population"]
    if split == "training":
        palette_count = int(population["training_palette_score_direction_count"])
        random_count = int(population["training_smooth_random_direction_count"])
        random_seed = int(population["training_direction_seed"])
        palette_offset = 0
    elif split == "development":
        palette_count = int(
            population["development_palette_score_direction_count"]
        )
        random_count = int(
            population["development_smooth_random_direction_count"]
        )
        random_seed = int(population["development_direction_seed"])
        palette_offset = int(population["training_palette_score_direction_count"])
    else:
        raise ValueError("split must be training or development")
    palette_total = int(population["training_palette_score_direction_count"]) + int(
        population["development_palette_score_direction_count"]
    )
    palette_all = _palette_score_grids(
        count=palette_total,
        rng=np.random.default_rng(int(population["palette_generator_seed"])),
        axis_size=int(config["renderer"]["velocity_grid_axis_size"]),
        maximum_norm=float(config["renderer"]["maximum_vector_norm_per_grid_node"]),
    )
    palette = palette_all[palette_offset : palette_offset + palette_count]
    random = _smooth_random_grids(
        count=random_count,
        rng=np.random.default_rng(random_seed),
        axis_size=int(config["renderer"]["velocity_grid_axis_size"]),
        smoothing_passes=int(population["smoothing_passes"]),
        minimum_norm=float(population["minimum_base_vector_norm"]),
        maximum_norm=float(population["maximum_base_vector_norm"]),
    )
    interleaved = []
    for index in range(max(len(palette), len(random))):
        if index < len(palette):
            interleaved.append(palette[index])
        if index < len(random):
            interleaved.append(random[index])
    return np.stack(interleaved)


def _mobius_gain(values: np.ndarray, gains: np.ndarray) -> np.ndarray:
    denominator = 1.0 + (gains[None, :] - 1.0) * values
    return gains[None, :] * values / denominator


def _apply_nuisance(
    values: np.ndarray,
    *,
    family_id: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> np.ndarray:
    if family_id == 0:
        return values
    if family_id == 1:
        limit = float(config["maximum_absolute_exposure_ev"])
        ev = float(rng.uniform(-limit, limit))
        return _mobius_gain(values, np.full(3, 2.0**ev))
    if family_id == 2:
        limit = float(config["maximum_white_balance_log_gain"])
        log_gains = rng.uniform(-limit, limit, size=3)
        log_gains -= np.mean(log_gains)
        return _mobius_gain(values, np.exp(log_gains))
    if family_id == 3:
        limit = float(config["maximum_basic_tone_deviation"])
        exponent = float(rng.uniform(1.0 - limit, 1.0 + limit))
        return values**exponent
    raise ValueError("unsupported nuisance family")


def _sample_content(
    *,
    family_id: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> np.ndarray:
    centres = np.asarray(config["family_centres"], dtype=np.float64)
    count_options = tuple(int(value) for value in config["mixture_component_counts"])
    count = count_options[int(rng.integers(0, len(count_options)))]
    palette = DiagonalGaussianMixturePalette(
        weights=rng.dirichlet(
            np.full(count, float(config["weight_dirichlet_alpha"]))
        ),
        means=np.clip(
            centres[family_id][None, :]
            + rng.normal(
                scale=float(config["component_mean_jitter_standard_deviation"]),
                size=(count, 3),
            ),
            0.01,
            0.99,
        ),
        standard_deviations=rng.uniform(
            float(config["component_standard_deviation_minimum"]),
            float(config["component_standard_deviation_maximum"]),
            size=(count, 3),
        ),
    )
    return sample_palette(
        palette,
        sample_count=int(config["samples_per_reference"]),
        rng=rng,
    )


def generate_episode_population(
    config: dict[str, Any], *, split: str
) -> ReferenceEpisodePopulation:
    """Generate a deterministic crossed reference population."""

    if split == "training":
        content_seed = int(config["content_population"]["training_content_seed"])
        nuisance_seed = int(config["nuisance_population"]["training_nuisance_seed"])
        group_count = int(
            config["content_population"]["training_groups_per_operator_instance"]
        )
    elif split == "development":
        content_seed = int(config["content_population"]["development_content_seed"])
        nuisance_seed = int(
            config["nuisance_population"]["development_nuisance_seed"]
        )
        group_count = int(
            config["content_population"]["development_groups_per_operator_instance"]
        )
    else:
        raise ValueError("split must be training or development")
    base_grids = generate_base_direction_grids(config, split=split)
    strengths = [float(value) for value in config["operator_population"]["strengths"]]
    grids = []
    direction_ids = []
    instance_strengths = []
    for direction_id, base in enumerate(base_grids):
        for strength in strengths:
            grids.append(strength * base)
            direction_ids.append(direction_id)
            instance_strengths.append(strength)
    grids.append(np.zeros_like(base_grids[0]))
    direction_ids.append(-1)
    instance_strengths.append(0.0)
    grid_array = np.stack(grids)

    content_rng = np.random.default_rng(content_seed)
    nuisance_rng = np.random.default_rng(nuisance_seed)
    references_per_group = int(
        config["content_population"]["references_per_group"]
    )
    content_count = int(config["content_population"]["family_count"])
    nuisance_count = len(config["nuisance_population"]["families"])
    all_references = []
    all_content_labels = []
    all_nuisance_labels = []
    for instance_index, grid in enumerate(grid_array):
        sources = []
        content_labels = np.empty(
            (group_count, references_per_group), dtype=np.int64
        )
        nuisance_labels = np.empty_like(content_labels)
        for group in range(group_count):
            group_sources = []
            for reference in range(references_per_group):
                content_id = (
                    instance_index + group * references_per_group + reference
                ) % content_count
                nuisance_id = (instance_index + group + reference) % nuisance_count
                source = _sample_content(
                    family_id=content_id,
                    rng=content_rng,
                    config=config["content_population"],
                )
                source = _apply_nuisance(
                    source,
                    family_id=nuisance_id,
                    rng=nuisance_rng,
                    config=config["nuisance_population"],
                )
                group_sources.append(source)
                content_labels[group, reference] = content_id
                nuisance_labels[group, reference] = nuisance_id
            sources.append(group_sources)
        source_array = np.asarray(sources, dtype=np.float64)
        operator = CubeDiffeomorphicColourFlow(
            grid,
            integration_steps=int(config["renderer"]["integration_steps"]),
        )
        styled = operator.apply(source_array.reshape(-1, 3)).reshape(
            source_array.shape
        )
        all_references.append(
            canonicalize_reference_groups(styled.astype(np.float32))
        )
        all_content_labels.append(content_labels)
        all_nuisance_labels.append(nuisance_labels)
    return ReferenceEpisodePopulation(
        references=np.stack(all_references),
        target_grids=grid_array,
        direction_ids=np.asarray(direction_ids),
        strengths=np.asarray(instance_strengths),
        content_labels=np.stack(all_content_labels),
        nuisance_labels=np.stack(all_nuisance_labels),
        identity_index=len(grid_array) - 1,
    )


def generate_fixed_content_controls(
    config: dict[str, Any],
    *,
    base_grids: np.ndarray,
) -> np.ndarray:
    """Apply several look grids to one exact four-reference content group."""

    grids = np.asarray(base_grids, dtype=np.float64)
    if (
        grids.ndim != 5
        or grids.shape[-1] != 3
        or not np.all(np.isfinite(grids))
    ):
        raise ValueError("base_grids must have shape [K,A,A,A,3]")
    rng = np.random.default_rng(
        int(config["content_population"]["same_content_different_look_control_seed"])
    )
    references = []
    for content_id in range(
        int(config["content_population"]["references_per_group"])
    ):
        references.append(
            _sample_content(
                family_id=content_id,
                rng=rng,
                config=config["content_population"],
            )
        )
    source = np.stack(references)
    styled_groups = []
    for grid in grids:
        operator = CubeDiffeomorphicColourFlow(
            grid,
            integration_steps=int(config["renderer"]["integration_steps"]),
        )
        styled_groups.append(
            operator.apply(source.reshape(-1, 3)).reshape(source.shape)
        )
    return canonicalize_reference_groups(
        np.asarray(styled_groups, dtype=np.float32)
    )


def generate_owner_strength_fixture(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact 53/55/56 references, grids, and strength values."""

    fixture = config["development_evaluation"]["owner_strength_path"]
    base = _smooth_random_grids(
        count=1,
        rng=np.random.default_rng(int(fixture["direction_seed"])),
        axis_size=int(config["renderer"]["velocity_grid_axis_size"]),
        smoothing_passes=int(config["operator_population"]["smoothing_passes"]),
        minimum_norm=float(config["operator_population"]["maximum_base_vector_norm"]),
        maximum_norm=float(config["operator_population"]["maximum_base_vector_norm"]),
    )[0]
    strengths = np.asarray(
        [float(fixture[key]) for key in ("53", "55", "56")],
        dtype=np.float64,
    )
    grids = strengths[:, None, None, None, None] * base[None]
    references = generate_fixed_content_controls(config, base_grids=grids)
    return references, grids, strengths


def parameter_state_sha256(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "HierarchicalReferenceOperatorPredictor",
    "ReferenceEpisodePopulation",
    "apply_velocity_grids_torch",
    "canonicalize_reference_groups",
    "generate_base_direction_grids",
    "generate_episode_population",
    "generate_fixed_content_controls",
    "generate_owner_strength_fixture",
    "parameter_state_sha256",
    "radial_tanh_bound",
    "vicreg_terms",
]
