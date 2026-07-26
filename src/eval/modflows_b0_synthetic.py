"""Clean-room synthetic adapter for the external ModFlows B0 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0


FLOW_PARAMETER_COUNT = 515
FLOW_INPUTS = 4
FLOW_HIDDEN = 64
FLOW_OUTPUTS = 3


@dataclass(frozen=True)
class ModFlowVelocity:
    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            ("first_weight", self.first_weight, (FLOW_HIDDEN, FLOW_INPUTS)),
            ("first_bias", self.first_bias, (FLOW_HIDDEN,)),
            ("second_weight", self.second_weight, (FLOW_OUTPUTS, FLOW_HIDDEN)),
            ("second_bias", self.second_bias, (FLOW_OUTPUTS,)),
        )
        for name, value, shape in arrays:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError("velocity parameters have invalid shape or values")
            readonly = array.copy()
            readonly.setflags(write=False)
            object.__setattr__(self, name, readonly)

    @classmethod
    def from_vector(cls, parameters: np.ndarray) -> "ModFlowVelocity":
        vector = np.asarray(parameters, dtype=np.float64)
        if (
            vector.shape != (FLOW_PARAMETER_COUNT,)
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError("flow parameter vector must contain 515 finite values")
        cursor = 0
        first_size = FLOW_HIDDEN * FLOW_INPUTS
        first_weight = vector[cursor : cursor + first_size].reshape(
            FLOW_HIDDEN,
            FLOW_INPUTS,
        )
        cursor += first_size
        first_bias = vector[cursor : cursor + FLOW_HIDDEN]
        cursor += FLOW_HIDDEN
        second_size = FLOW_OUTPUTS * FLOW_HIDDEN
        second_weight = vector[cursor : cursor + second_size].reshape(
            FLOW_OUTPUTS,
            FLOW_HIDDEN,
        )
        cursor += second_size
        second_bias = vector[cursor : cursor + FLOW_OUTPUTS]
        cursor += FLOW_OUTPUTS
        if cursor != FLOW_PARAMETER_COUNT:
            raise RuntimeError("flow parameter parsing mismatch")
        return cls(first_weight, first_bias, second_weight, second_bias)

    def __call__(self, rgb: np.ndarray, time: float) -> np.ndarray:
        points = np.asarray(rgb, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not points.size
            or not np.all(np.isfinite(points))
            or not np.isfinite(time)
        ):
            raise ValueError("velocity input must be finite Nx3 points and time")
        time_column = np.full((len(points), 1), float(time), dtype=np.float64)
        features = np.concatenate((points, time_column), axis=1)
        hidden = np.tanh(features @ self.first_weight.T + self.first_bias)
        return hidden @ self.second_weight.T + self.second_bias


def integrate_modflow_rk4(
    rgb: np.ndarray,
    velocity: ModFlowVelocity,
    *,
    start_time: float,
    end_time: float,
    steps: int,
) -> np.ndarray:
    """Integrate one frozen velocity field with fixed-step RK4."""

    points = np.asarray(rgb, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not points.size
        or not np.all(np.isfinite(points))
        or not isinstance(steps, int)
        or steps <= 0
        or not np.isfinite(start_time)
        or not np.isfinite(end_time)
    ):
        raise ValueError("invalid RK4 inputs")
    output = points.copy()
    step = (float(end_time) - float(start_time)) / steps
    time = float(start_time)
    for _ in range(steps):
        k1 = velocity(output, time)
        k2 = velocity(output + 0.5 * step * k1, time + 0.5 * step)
        k3 = velocity(output + 0.5 * step * k2, time + 0.5 * step)
        k4 = velocity(output + step * k3, time + step)
        output += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += step
    if not np.all(np.isfinite(output)):
        raise RuntimeError("RK4 integration produced non-finite values")
    return output


def transfer_modflows(
    rgb: np.ndarray,
    *,
    content_parameters: np.ndarray,
    style_parameters: np.ndarray,
    steps_per_leg: int,
) -> np.ndarray:
    """Apply content-forward then style-reverse clean-room flow composition."""

    latent = integrate_modflow_rk4(
        rgb,
        ModFlowVelocity.from_vector(content_parameters),
        start_time=0.0,
        end_time=1.0,
        steps=steps_per_leg,
    )
    return integrate_modflow_rk4(
        latent,
        ModFlowVelocity.from_vector(style_parameters),
        start_time=1.0,
        end_time=0.0,
        steps=steps_per_leg,
    )


class ModFlowsB0Encoder(nn.Module):
    """Torchvision B0 wrapper whose state names match the checkpoint prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            FLOW_PARAMETER_COUNT,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def load_modflows_b0_encoder(
    checkpoint_path: Path,
) -> tuple[ModFlowsB0Encoder, dict[str, object]]:
    """Load a pinned tensor-only checkpoint and require exact architecture."""

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("checkpoint must be a non-empty tensor state dictionary")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        raise ValueError("checkpoint contains non-tensor state")
    model = ModFlowsB0Encoder()
    expected = model.state_dict()
    if set(payload) != set(expected):
        missing = sorted(set(expected) - set(payload))
        unexpected = sorted(set(payload) - set(expected))
        raise ValueError(
            f"checkpoint architecture mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    shape_mismatches = {
        key: (tuple(payload[key].shape), tuple(expected[key].shape))
        for key in expected
        if tuple(payload[key].shape) != tuple(expected[key].shape)
    }
    if shape_mismatches:
        raise ValueError(f"checkpoint shape mismatch: {shape_mismatches}")
    model.load_state_dict(payload, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "state_key_count": len(expected),
        "tensor_parameter_count": sum(
            int(value.numel())
            for key, value in expected.items()
            if not key.endswith("num_batches_tracked")
        ),
        "classifier_weight_shape": tuple(
            expected["model.classifier.1.weight"].shape
        ),
        "classifier_bias_shape": tuple(
            expected["model.classifier.1.bias"].shape
        ),
    }
    return model, metadata


def encode_modflows_images(
    model: ModFlowsB0Encoder,
    images: np.ndarray,
    *,
    normalization_mean: tuple[float, float, float],
    normalization_std: tuple[float, float, float],
) -> np.ndarray:
    """Encode a frozen batch of float RGB images without augmentation."""

    values = np.asarray(images, dtype=np.float32)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or not values.size
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("images must be finite NHWC RGB values in [0,1]")
    tensor = torch.from_numpy(np.moveaxis(values, -1, 1).copy())
    mean = torch.tensor(normalization_mean, dtype=tensor.dtype)[None, :, None, None]
    std = torch.tensor(normalization_std, dtype=tensor.dtype)[None, :, None, None]
    with torch.inference_mode():
        output = model((tensor - mean) / std)
    result = output.detach().cpu().numpy().astype(np.float64)
    if result.shape != (len(values), FLOW_PARAMETER_COUNT) or not np.all(
        np.isfinite(result)
    ):
        raise RuntimeError("encoder output is invalid")
    return result


__all__ = [
    "FLOW_HIDDEN",
    "FLOW_INPUTS",
    "FLOW_OUTPUTS",
    "FLOW_PARAMETER_COUNT",
    "ModFlowVelocity",
    "ModFlowsB0Encoder",
    "encode_modflows_images",
    "integrate_modflow_rk4",
    "load_modflows_b0_encoder",
    "transfer_modflows",
]
