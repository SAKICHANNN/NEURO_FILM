"""Frozen U6.P4DG synthetic image-domain cloud artifact evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from src.eval.cross_layer_cloud_row_stream import _profile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_target_density_cross_layer_cloud_rows,
)
from src.film_physics.density_conditioned_structure import counter_poisson_rate_field


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4DG parent drift")
    return contract


def _chart(shape: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = shape
    image = np.empty((h, w, 3), dtype=np.float64)
    patch_bounds = []
    for index, level in enumerate(np.linspace(0.12, 0.82, 8)):
        x0, x1 = index * w // 8, (index + 1) * w // 8
        image[: h // 3, x0:x1] = level
        patch_bounds.append((x0, x1))
    image[h // 3 : 2 * h // 3, : w // 2] = 0.18
    image[h // 3 : 2 * h // 3, w // 2 :] = 0.78
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)
    image[2 * h // 3 :, :, 0] = 0.18 + 0.58 * x
    image[2 * h // 3 :, :, 1] = 0.72 - 0.42 * x
    image[2 * h // 3 :, :, 2] = 0.24 + 0.48 * np.square(x)
    return image, patch_bounds


def _expected_density(target: np.ndarray, profile) -> np.ndarray:
    return np.stack(
        [
            gaussian_filter(
                target[..., channel],
                sigma=profile.gaussian_sigma_pixels_cmy[channel],
                mode="wrap",
                truncate=profile.gaussian_truncate,
            )
            for channel in range(3)
        ],
        axis=-1,
    )


def _independent_density(
    profile, target: np.ndarray, maximum: np.ndarray, seed: int
) -> np.ndarray:
    shape = target.shape[:2]
    layers = []
    for channel, (marginal, mark, sigma) in enumerate(
        zip(
            profile.count_profile.marginal_rates_cmy,
            profile.count_profile.mark_optical_density_cmy,
            profile.gaussian_sigma_pixels_cmy,
            strict=True,
        )
    ):
        rate = target[..., channel] / maximum[channel] * marginal
        counts = counter_poisson_rate_field(
            rate,
            shape,
            origin_yx=(0, 0),
            seed=seed + (20 + channel) * profile.count_profile.component_seed_stride,
            maximum_rate=marginal,
        )
        layers.append(
            gaussian_filter(
                counts.astype(np.float64),
                sigma=sigma,
                mode="wrap",
                truncate=profile.gaussian_truncate,
            )
            * mark
        )
    return np.stack(layers, axis=-1).astype(np.float32)


def _render_output(
    base: np.ndarray, density: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    return base * np.exp(-(density.astype(np.float64) - expected))


def _metrics(
    output: np.ndarray, patch_bounds: list[tuple[int, int]], contract: dict[str, Any]
) -> dict[str, float]:
    f = contract["fixture"]
    h, w = output.shape[:2]
    margin = f["neutral_patch_margin"]
    neutral = output[margin : h // 3 - margin]
    neutral_chroma = np.max(neutral, axis=-1) - np.min(neutral, axis=-1)
    patch_chroma = []
    for x0, x1 in patch_bounds:
        mean = np.mean(
            neutral[:, x0 + margin : x1 - margin], axis=(0, 1), dtype=np.float64
        )
        patch_chroma.append(float(np.max(mean) - np.min(mean)))
    edge = output[h // 3 + margin : 2 * h // 3 - margin]
    edge_profile = np.mean(edge, axis=(0, 2), dtype=np.float64)
    exclusion = f["edge_exclusion"]
    dark_reference = float(np.mean(edge_profile[: w // 2 - exclusion]))
    bright_reference = float(np.mean(edge_profile[w // 2 + exclusion :]))
    overshoot = max(
        0.0,
        float(
            np.max(edge_profile[w // 2 - exclusion : w // 2 + exclusion])
            - bright_reference
        ),
    )
    undershoot = max(
        0.0,
        float(
            dark_reference
            - np.min(edge_profile[w // 2 - exclusion : w // 2 + exclusion])
        ),
    )
    return {
        "neutral_chroma_p99": float(np.quantile(neutral_chroma, 0.99)),
        "maximum_neutral_patch_mean_chroma": max(patch_chroma),
        "edge_overshoot": overshoot,
        "edge_undershoot": undershoot,
        "new_boundary_fraction": float(np.mean((output < 0.0) | (output > 1.0))),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    shape = (f["height"], f["width"])
    profile = _profile(root)
    base, patches = _chart(shape)
    maximum = np.asarray(profile.count_profile.marginal_rates_cmy) * np.asarray(
        profile.count_profile.mark_optical_density_cmy
    )
    luminance = np.sum(base * np.asarray([0.2126, 0.7152, 0.0722]), axis=-1)
    scale = np.clip(0.25 + 0.5 * (1.0 - luminance), 0.05, 0.95)
    target = scale[..., None] * maximum
    rows = tuple(
        iter_target_density_cross_layer_cloud_rows(
            profile,
            target,
            maximum_developed_density_cmy=tuple(maximum),
            seed=f["seed"],
            row_tile_height=127,
        )
    )
    shared_density = np.concatenate([result.density for _, result in rows])
    repeat_rows = tuple(
        iter_target_density_cross_layer_cloud_rows(
            profile,
            target,
            maximum_developed_density_cmy=tuple(maximum),
            seed=f["seed"],
            row_tile_height=127,
        )
    )
    repeat_density = np.concatenate([result.density for _, result in repeat_rows])
    expected = _expected_density(target, profile)
    independent_density = _independent_density(profile, target, maximum, f["seed"])
    shared_output = _render_output(base, shared_density, expected)
    independent_output = _render_output(base, independent_density, expected)
    shared_metrics = _metrics(shared_output, patches, contract)
    independent_metrics = _metrics(independent_output, patches, contract)
    ratio = (
        shared_metrics["neutral_chroma_p99"] / independent_metrics["neutral_chroma_p99"]
    )
    gates = contract["gates"]
    results = {
        "neutral_chroma": shared_metrics["neutral_chroma_p99"]
        <= gates["maximum_neutral_chroma_p99"],
        "patch_chroma": shared_metrics["maximum_neutral_patch_mean_chroma"]
        <= gates["maximum_neutral_patch_mean_chroma"],
        "overshoot": shared_metrics["edge_overshoot"]
        <= gates["maximum_edge_overshoot"],
        "undershoot": shared_metrics["edge_undershoot"]
        <= gates["maximum_edge_undershoot"],
        "boundary": shared_metrics["new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "beats_independent_chroma": ratio
        <= gates["maximum_shared_to_independent_neutral_chroma_p99_ratio"],
        "repeat_identity": bool(np.array_equal(shared_density, repeat_density)),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "shared_output_sha256": hashlib.sha256(
            shared_output.astype("<f4").tobytes()
        ).hexdigest(),
        "shared": shared_metrics,
        "independent": independent_metrics,
        "shared_to_independent_neutral_chroma_p99_ratio": ratio,
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dg_cross_layer_cloud_chart_artifact_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
