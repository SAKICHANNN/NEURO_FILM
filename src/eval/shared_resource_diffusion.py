"""Independent dimensionless shared-resource diffusion research primitive."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _heat_step(field: np.ndarray, coefficient: float) -> np.ndarray:
    padded = np.pad(field, ((1, 1), (1, 1)), mode="reflect")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * field
    )
    return field + coefficient * laplacian


def simulate(
    image: np.ndarray,
    solver: Mapping[str, Any],
    *,
    strength: float | None = None,
    diffusion: bool = True,
    shared_resource_enabled: bool = True,
) -> dict[str, np.ndarray]:
    activation = np.asarray(image, dtype=np.float64)
    if activation.ndim != 3 or activation.shape[2] != 3:
        raise ValueError("shared-resource input must be HxWx3")
    if not np.all(np.isfinite(activation)) or np.min(activation) < 0 or np.max(activation) > 1:
        raise ValueError("shared-resource input must be finite in [0,1]")
    output_strength = solver["output_strength"] if strength is None else strength
    if output_strength == 0.0:
        return {
            "output": activation.copy(),
            "deposit": np.zeros_like(activation),
            "layer_resource": np.ones_like(activation),
            "shared_resource": np.ones(activation.shape[:2], dtype=np.float64),
        }

    deposit = np.zeros_like(activation)
    layer = np.ones_like(activation)
    shared = np.ones(activation.shape[:2], dtype=np.float64)
    dt = float(solver["time_step"])
    for _ in range(int(solver["steps"])):
        reaction = (
            float(solver["reaction_rate"])
            * activation
            * layer
            * shared[..., None]
            * dt
        )
        deposit = deposit + reaction
        layer = layer - reaction
        if shared_resource_enabled:
            shared = shared - float(solver["shared_resource_consumption"]) * np.mean(
                reaction, axis=2
            )
            if diffusion:
                shared = _heat_step(shared, float(solver["diffusion_courant"]))
            shared = shared + float(solver["reservoir_mix"]) * (1.0 - shared)
    output = (1.0 - output_strength) * activation + output_strength * deposit
    return {
        "output": output,
        "deposit": deposit,
        "layer_resource": layer,
        "shared_resource": shared,
    }


def make_patterns(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    population = config["synthetic_population"]
    height, width = int(population["height"]), int(population["width"])
    patterns: dict[str, np.ndarray] = {}
    for level in population["constant_levels"]:
        patterns[f"constant_{level}"] = np.full((height, width, 3), level)
    base = float(population["constant_levels"][1])
    high = float(population["edge_levels"][1])
    for size in population["square_widths"]:
        image = np.full((height, width, 3), base)
        y0, x0 = (height - size) // 2, (width - size) // 2
        image[y0 : y0 + size, x0 : x0 + size] = high
        patterns[f"square_{size}"] = image
    low, high = population["edge_levels"]
    edge = np.full((height, width, 3), low)
    edge[:, width // 2 :] = high
    patterns["edge"] = edge
    start, end = population["gradient_endpoints"]
    ramp = np.linspace(start, end, width)
    patterns["gradient"] = np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2)
    for index, value in enumerate(population["chromatic_patch_values"]):
        image = np.zeros((height, width, 3), dtype=np.float64)
        image[height // 2 - 8 : height // 2 + 8, width // 2 - 8 : width // 2 + 8] = value
        patterns[f"primary_{index}"] = image
        isolated = np.zeros((height, width, 3), dtype=np.float64)
        isolated[
            height // 2 - 8 : height // 2 + 8,
            width // 2 - 8 : width // 2 + 8,
            index,
        ] = 0.8
        patterns[f"isolated_{index}"] = isolated
    return patterns


def evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    solver = config["solver"]
    patterns = make_patterns(config)
    outputs = {name: simulate(image, solver) for name, image in patterns.items()}
    all_state = np.concatenate(
        [
            value.reshape(-1)
            for result in outputs.values()
            for value in result.values()
        ]
    )

    reference = patterns["square_17"]
    identity_error = float(
        np.max(np.abs(simulate(reference, solver, strength=0.0)["output"] - reference))
    )
    constant_spans = [
        float(np.max(result["output"]) - np.min(result["output"]))
        for name, result in outputs.items()
        if name.startswith("constant_")
    ]
    neutral_spreads = [
        float(np.max(np.ptp(result["output"], axis=2)))
        for name, result in outputs.items()
        if name.startswith("constant_") or name.startswith("square_") or name in {"edge", "gradient"}
    ]
    rotated = np.rot90(reference)
    rotation_error = float(
        np.max(
            np.abs(
                simulate(rotated, solver)["output"]
                - np.rot90(outputs["square_17"]["output"])
            )
        )
    )
    repeat_error = float(
        np.max(
            np.abs(
                simulate(reference, solver)["output"]
                - outputs["square_17"]["output"]
            )
        )
    )
    constant_means = [
        float(np.mean(outputs[f"constant_{level}"]["output"]))
        for level in config["synthetic_population"]["constant_levels"]
    ]
    constant_deltas = np.diff(constant_means)

    spatial_names = [name for name in patterns if not name.startswith("constant_")]
    local_rmse = []
    for name in spatial_names:
        global_output = simulate(patterns[name], solver, diffusion=False)["output"]
        local_rmse.append(
            float(np.sqrt(np.mean((outputs[name]["output"] - global_output) ** 2)))
        )
    local_effect = float(np.max(local_rmse))

    edge_input = patterns["edge"][:, :, 0]
    edge_output = outputs["edge"]["output"][:, :, 0]
    input_gradient = float(np.max(np.abs(np.diff(edge_input, axis=1))))
    output_gradient = float(np.max(np.abs(np.diff(edge_output, axis=1))))
    gradient_amplification = output_gradient / input_gradient
    ringing = max(
        float(np.ptp(edge_output[:, :7])),
        float(np.ptp(edge_output[:, -7:])),
    )

    crosstalk = 0.0
    for index in range(3):
        output = outputs[f"isolated_{index}"]["output"]
        inactive = [channel for channel in range(3) if channel != index]
        crosstalk = max(crosstalk, float(np.max(np.abs(output[:, :, inactive]))))

    support_radius = int(solver["steps"])
    gates = config["gates"]
    metrics = {
        "state_minimum": float(np.min(all_state)),
        "state_maximum": float(np.max(all_state)),
        "identity_max_abs_error": identity_error,
        "constant_field_spatial_span_max": max(constant_spans),
        "neutral_axis_max_abs_channel_spread": max(neutral_spreads),
        "rotation90_max_abs_error": rotation_error,
        "repeat_max_abs_error": repeat_error,
        "constant_field_monotone_min_delta": float(np.min(constant_deltas)),
        "local_effect_rmse_vs_global_max": local_effect,
        "local_effect_rmse_per_pattern": dict(zip(spatial_names, local_rmse)),
        "edge_gradient_amplification": gradient_amplification,
        "flat_region_ringing": ringing,
        "channel_crosstalk": crosstalk,
        "finite_support_radius": support_radius,
    }
    checks = {
        "finite": bool(np.all(np.isfinite(all_state))),
        "range": metrics["state_minimum"] >= gates["state_and_output_range"][0]
        and metrics["state_maximum"] <= gates["state_and_output_range"][1],
        "identity": identity_error <= gates["identity_max_abs_error"],
        "constant_uniformity": metrics["constant_field_spatial_span_max"]
        <= gates["constant_field_spatial_span_max"],
        "neutral_axis": metrics["neutral_axis_max_abs_channel_spread"]
        <= gates["neutral_axis_max_abs_channel_spread"],
        "rotation": rotation_error <= gates["rotation90_max_abs_error"],
        "repeat": repeat_error == 0.0,
        "constant_monotone": metrics["constant_field_monotone_min_delta"]
        >= gates["constant_field_monotone_min_delta"],
        "local_effect_floor": local_effect >= gates["local_effect_rmse_vs_global_min"],
        "local_effect_cap": local_effect <= gates["local_effect_rmse_vs_global_max"],
        "gradient": gradient_amplification <= gates["maximum_edge_gradient_amplification"],
        "ringing": ringing <= gates["maximum_flat_region_ringing"],
        "crosstalk": crosstalk
        <= gates["maximum_channel_crosstalk_on_single_channel_input"],
        "support": support_radius <= gates["finite_support_radius_max"],
    }
    if not checks["local_effect_floor"] and all(
        value for key, value in checks.items() if key != "local_effect_floor"
    ):
        decision = "effect_too_weak"
    elif not all(checks.values()):
        decision = "structure_fail"
    else:
        decision = "pass"
    return {"metrics": metrics, "checks": checks, "decision": decision}
