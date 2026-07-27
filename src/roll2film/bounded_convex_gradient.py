"""Compact cube-bounded colour maps with an analytic positive Jacobian.

The map is the gradient of a strongly convex log-sum-exp potential:

    T(x) = (1 - alpha) x + alpha * sum_k softmax_k(x) anchor_k

Anchors lie in the RGB cube, so the result is in the cube without clipping.
For alpha < 1 the Jacobian is symmetric positive definite everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


BOUNDED_CONVEX_GRADIENT_SCHEMA = "roll2film.bounded_convex_gradient_map.v1"


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data with shape (..., 3)")
    return values


def _stable_probabilities(
    rgb_rows: np.ndarray,
    anchors: np.ndarray,
    centered_biases: np.ndarray,
    temperature: float,
) -> np.ndarray:
    logits = (rgb_rows @ anchors.T + centered_biases[None, :]) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=1, keepdims=True)


@dataclass(frozen=True)
class BoundedConvexGradientMap:
    """An explicit in-cube colour map with an analytic SPD Jacobian."""

    anchors: np.ndarray
    centered_biases: np.ndarray
    strength: float
    temperature: float = 0.15
    maximum_strength: float = 0.65
    maximum_absolute_centered_bias: float = 1.0
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchors, dtype=np.float64).copy()
        biases = np.asarray(self.centered_biases, dtype=np.float64).copy()
        if (
            anchors.ndim != 2
            or anchors.shape[1] != 3
            or anchors.shape[0] < 2
            or not np.all(np.isfinite(anchors))
            or np.any(anchors < 0.0)
            or np.any(anchors > 1.0)
        ):
            raise ValueError("anchors must have finite shape (K, 3) in [0, 1]")
        if (
            biases.shape != (anchors.shape[0],)
            or not np.all(np.isfinite(biases))
            or abs(float(np.mean(biases))) > 1e-12
            or np.max(np.abs(biases)) > self.maximum_absolute_centered_bias + 1e-12
        ):
            raise ValueError("biases must be finite, centered, bounded, and match K")
        if (
            not np.isfinite(self.strength)
            or self.strength < 0.0
            or self.strength > self.maximum_strength
            or not np.isfinite(self.temperature)
            or self.temperature <= 0.0
            or not np.isfinite(self.maximum_strength)
            or self.maximum_strength <= 0.0
            or self.maximum_strength >= 1.0
            or not np.isfinite(self.maximum_absolute_centered_bias)
            or self.maximum_absolute_centered_bias <= 0.0
        ):
            raise ValueError("invalid strength, temperature, or parameter bounds")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        anchors.setflags(write=False)
        biases.setflags(write=False)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "centered_biases", biases)

    @classmethod
    def identity(
        cls,
        *,
        anchors: np.ndarray,
        temperature: float = 0.15,
        maximum_strength: float = 0.65,
        maximum_absolute_centered_bias: float = 1.0,
    ) -> "BoundedConvexGradientMap":
        anchor_values = np.asarray(anchors, dtype=np.float64)
        return cls(
            anchors=anchor_values,
            centered_biases=np.zeros(anchor_values.shape[0], dtype=np.float64),
            strength=0.0,
            temperature=temperature,
            maximum_strength=maximum_strength,
            maximum_absolute_centered_bias=maximum_absolute_centered_bias,
        )

    @property
    def raw_fitted_scalar_count(self) -> int:
        return int(self.anchors.size + self.centered_biases.size + 1)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        if self.strength == 0.0:
            return values.copy()
        shape = values.shape
        rows = values.reshape(-1, 3)
        weights = _stable_probabilities(
            rows, self.anchors, self.centered_biases, self.temperature
        )
        barycenter = weights @ self.anchors
        result = (1.0 - self.strength) * rows + self.strength * barycenter
        if (
            not np.all(np.isfinite(result))
            or np.any(result < 0.0)
            or np.any(result > 1.0)
        ):
            raise RuntimeError("bounded convex-gradient map escaped the RGB cube")
        return result.reshape(shape)

    def jacobians(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape[:-1]
        rows = values.reshape(-1, 3)
        if self.strength == 0.0:
            identity = np.eye(3, dtype=np.float64)
            return np.broadcast_to(identity, (*shape, 3, 3)).copy()
        weights = _stable_probabilities(
            rows, self.anchors, self.centered_biases, self.temperature
        )
        means = weights @ self.anchors
        second = np.einsum(
            "nk,ki,kj->nij", weights, self.anchors, self.anchors, optimize=True
        )
        covariance = second - np.einsum("ni,nj->nij", means, means)
        jacobian = (
            (1.0 - self.strength) * np.eye(3, dtype=np.float64)[None, :, :]
            + (self.strength / self.temperature) * covariance
        )
        return jacobian.reshape(*shape, 3, 3)

    def inverse(
        self,
        rgb: np.ndarray,
        *,
        maximum_iterations: int = 40,
        residual_tolerance: float = 1e-12,
        minimum_line_search_scale: float = 1.0 / 1024.0,
    ) -> np.ndarray:
        values = _validate_rgb(rgb)
        if self.strength == 0.0:
            return values.copy()
        if (
            maximum_iterations < 1
            or not np.isfinite(residual_tolerance)
            or residual_tolerance <= 0.0
            or not np.isfinite(minimum_line_search_scale)
            or minimum_line_search_scale <= 0.0
            or minimum_line_search_scale > 1.0
        ):
            raise ValueError("invalid inverse settings")
        shape = values.shape
        target = values.reshape(-1, 3)
        current = target.copy()
        for _ in range(maximum_iterations):
            residual = self.apply(current) - target
            norms = np.max(np.abs(residual), axis=1)
            active = norms > residual_tolerance
            if not np.any(active):
                return current.reshape(shape)
            active_indices = np.flatnonzero(active)
            delta = np.linalg.solve(
                self.jacobians(current[active]),
                residual[active, ..., None],
            )[..., 0]
            scale = 1.0
            accepted = np.zeros(len(active_indices), dtype=bool)
            next_values = current[active].copy()
            while scale >= minimum_line_search_scale and not np.all(accepted):
                pending = ~accepted
                candidate = current[active][pending] - scale * delta[pending]
                in_cube = np.all((candidate >= 0.0) & (candidate <= 1.0), axis=1)
                candidate_residual = np.full(len(candidate), np.inf, dtype=np.float64)
                if np.any(in_cube):
                    candidate_residual[in_cube] = np.max(
                        np.abs(
                            self.apply(candidate[in_cube])
                            - target[active][pending][in_cube]
                        ),
                        axis=1,
                    )
                improves = candidate_residual < norms[active][pending]
                pending_indices = np.flatnonzero(pending)
                if np.any(improves):
                    accepted_indices = pending_indices[improves]
                    next_values[accepted_indices] = candidate[improves]
                    accepted[accepted_indices] = True
                scale *= 0.5
            if not np.all(accepted):
                raise RuntimeError("bounded Newton inverse line search failed")
            current[active_indices] = next_values
        residual = np.max(np.abs(self.apply(current) - target))
        if residual > residual_tolerance:
            raise RuntimeError("bounded Newton inverse did not converge")
        return current.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BOUNDED_CONVEX_GRADIENT_SCHEMA,
            "working_space": self.working_space,
            "anchors": self.anchors.tolist(),
            "centered_biases": self.centered_biases.tolist(),
            "strength": self.strength,
            "temperature": self.temperature,
            "maximum_strength": self.maximum_strength,
            "maximum_absolute_centered_bias": self.maximum_absolute_centered_bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundedConvexGradientMap":
        if payload.get("schema") != BOUNDED_CONVEX_GRADIENT_SCHEMA:
            raise ValueError("unsupported bounded convex-gradient map schema")
        return cls(
            anchors=np.asarray(payload["anchors"], dtype=np.float64),
            centered_biases=np.asarray(payload["centered_biases"], dtype=np.float64),
            strength=float(payload["strength"]),
            temperature=float(payload["temperature"]),
            maximum_strength=float(payload["maximum_strength"]),
            maximum_absolute_centered_bias=float(
                payload["maximum_absolute_centered_bias"]
            ),
            working_space=str(payload["working_space"]),
        )


def fit_bounded_convex_gradient_map(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    initial_anchors: np.ndarray,
    temperature: float,
    maximum_strength: float,
    maximum_absolute_centered_bias: float,
    seed: int,
    steps: int,
    learning_rate: float,
    bias_l2: float,
    anchor_l2_to_initial: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> BoundedConvexGradientMap:
    """Fit the 65-scalar explicit map using deterministic CPU float64 Adam."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    initial = np.asarray(initial_anchors, dtype=np.float64)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    if (
        initial.ndim != 2
        or initial.shape[1] != 3
        or initial.shape[0] < 2
        or not np.all(np.isfinite(initial))
        or np.any(initial <= 0.0)
        or np.any(initial >= 1.0)
        or temperature <= 0.0
        or maximum_strength <= 0.0
        or maximum_strength >= 1.0
        or maximum_absolute_centered_bias <= 0.0
        or steps < 1
        or learning_rate <= 0.0
        or bias_l2 < 0.0
        or anchor_l2_to_initial < 0.0
        or gradient_clip_norm <= 0.0
        or thread_count < 1
    ):
        raise ValueError("invalid fit settings")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(thread_count))
    source = torch.from_numpy(values.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(targets.reshape(-1, 3).copy())
    initial_tensor = torch.from_numpy(initial.copy())
    anchor_logits = torch.logit(initial_tensor).requires_grad_(True)
    bias_logits = torch.zeros(initial.shape[0], dtype=torch.float64, requires_grad=True)
    initial_strength = min(0.35, maximum_strength * 0.75)
    strength_fraction = initial_strength / maximum_strength
    strength_logit = torch.tensor(
        np.log(strength_fraction / (1.0 - strength_fraction)),
        dtype=torch.float64,
        requires_grad=True,
    )
    parameters = [anchor_logits, bias_logits, strength_logit]
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        anchors = torch.sigmoid(anchor_logits)
        bias_shape = torch.tanh(bias_logits)
        bias_shape = bias_shape - torch.mean(bias_shape)
        bias_scale = torch.maximum(
            torch.ones((), dtype=torch.float64),
            torch.max(torch.abs(bias_shape)),
        )
        biases = maximum_absolute_centered_bias * bias_shape / bias_scale
        strength = maximum_strength * torch.sigmoid(strength_logit)
        logits = (source @ anchors.T + biases[None, :]) / temperature
        weights = torch.softmax(logits, dim=1)
        prediction = (1.0 - strength) * source + strength * (weights @ anchors)
        loss = torch.mean((prediction - target_tensor) ** 2)
        if bias_l2:
            loss = loss + bias_l2 * torch.mean(biases**2)
        if anchor_l2_to_initial:
            loss = loss + anchor_l2_to_initial * torch.mean(
                (anchors - initial_tensor) ** 2
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm))
        optimizer.step()
    with torch.no_grad():
        anchors = torch.sigmoid(anchor_logits)
        bias_shape = torch.tanh(bias_logits)
        bias_shape = bias_shape - torch.mean(bias_shape)
        bias_scale = torch.maximum(
            torch.ones((), dtype=torch.float64),
            torch.max(torch.abs(bias_shape)),
        )
        biases = maximum_absolute_centered_bias * bias_shape / bias_scale
        strength = maximum_strength * torch.sigmoid(strength_logit)
    return BoundedConvexGradientMap(
        anchors=anchors.cpu().numpy(),
        centered_biases=biases.cpu().numpy(),
        strength=float(strength.cpu()),
        temperature=float(temperature),
        maximum_strength=float(maximum_strength),
        maximum_absolute_centered_bias=float(maximum_absolute_centered_bias),
    )


__all__ = [
    "BOUNDED_CONVEX_GRADIENT_SCHEMA",
    "BoundedConvexGradientMap",
    "fit_bounded_convex_gradient_map",
]
