"""Frozen U6.P4DF neutral-field noise-power mechanism comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_cloud_row_stream import _profile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_target_density_cross_layer_cloud_rows,
)
from src.film_physics.cross_layer_gaussian_cloud import (
    render_cross_layer_gaussian_cloud_density,
)
from src.film_physics.density_conditioned_structure import (
    counter_poisson_rate_field,
)


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
        raise RuntimeError("P4DF parent drift")
    return contract


def _noise_metrics(density: np.ndarray, margin: int) -> dict[str, Any]:
    values = density[margin:-margin, margin:-margin].astype(np.float64)
    residual = values - np.mean(values, axis=(0, 1), keepdims=True, dtype=np.float64)
    variance = np.mean(np.square(residual), axis=(0, 1), dtype=np.float64)
    normalized = residual / np.sqrt(variance)[None, None, :]
    chroma = np.stack(
        (
            normalized[..., 0] - normalized[..., 1],
            normalized[..., 0] - normalized[..., 2],
            normalized[..., 1] - normalized[..., 2],
        ),
        axis=-1,
    )
    chroma_power = float(np.mean(np.square(chroma), dtype=np.float64))
    marginal_power = float(np.mean(variance, dtype=np.float64))
    # Report coarse radial NPS fractions from the three normalized difference planes.
    power = np.mean(np.square(np.abs(np.fft.rfft2(chroma, axes=(0, 1)))), axis=2)
    fy = np.fft.fftfreq(chroma.shape[0])[:, None]
    fx = np.fft.rfftfreq(chroma.shape[1])[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    total = float(np.sum(power, dtype=np.float64))
    bands = {
        "low": float(np.sum(power[radius < 0.08], dtype=np.float64) / total),
        "mid": float(
            np.sum(power[(radius >= 0.08) & (radius < 0.25)], dtype=np.float64) / total
        ),
        "high": float(np.sum(power[radius >= 0.25], dtype=np.float64) / total),
    }
    return {
        "marginal_variance_cmy": variance.tolist(),
        "marginal_noise_power": marginal_power,
        "normalized_chroma_noise_power": chroma_power,
        "normalized_chroma_nps_fraction": bands,
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    shape = (f["height"], f["width"])
    profile = _profile(root)
    maximum = np.asarray(profile.count_profile.marginal_rates_cmy) * np.asarray(
        profile.count_profile.mark_optical_density_cmy
    )
    target = np.broadcast_to(maximum * f["neutral_scale"], (*shape, 3)).copy()
    rows = tuple(
        iter_target_density_cross_layer_cloud_rows(
            profile,
            target,
            maximum_developed_density_cmy=tuple(maximum),
            seed=f["seed"],
            row_tile_height=127,
        )
    )
    shared = np.concatenate([result.density for _, result in rows])
    rows_repeat = tuple(
        iter_target_density_cross_layer_cloud_rows(
            profile,
            target,
            maximum_developed_density_cmy=tuple(maximum),
            seed=f["seed"],
            row_tile_height=127,
        )
    )
    shared_repeat = np.concatenate([result.density for _, result in rows_repeat])
    independent_counts = np.stack(
        [
            counter_poisson_rate_field(
                np.full(shape, rate * f["neutral_scale"]),
                shape,
                origin_yx=(0, 0),
                seed=f["seed"]
                + (20 + channel) * profile.count_profile.component_seed_stride,
                maximum_rate=rate,
            )
            for channel, rate in enumerate(profile.count_profile.marginal_rates_cmy)
        ],
        axis=-1,
    )
    independent = render_cross_layer_gaussian_cloud_density(
        independent_counts,
        mark_optical_density_cmy=profile.count_profile.mark_optical_density_cmy,
        sigma_pixels_cmy=profile.gaussian_sigma_pixels_cmy,
        truncate=profile.gaussian_truncate,
    )
    shared_metrics = _noise_metrics(shared, f["analysis_margin"])
    independent_metrics = _noise_metrics(independent, f["analysis_margin"])
    rng = np.random.default_rng(f["seed"] + 9001)
    display = rng.normal(size=shared.shape).astype(np.float64)
    display *= np.sqrt(np.asarray(shared_metrics["marginal_variance_cmy"]))[
        None, None, :
    ]
    display_metrics = _noise_metrics(display.astype(np.float32), f["analysis_margin"])
    chroma_ratio = (
        shared_metrics["normalized_chroma_noise_power"]
        / independent_metrics["normalized_chroma_noise_power"]
    )
    marginal_ratio = (
        shared_metrics["marginal_noise_power"]
        / independent_metrics["marginal_noise_power"]
    )
    marginal_relative = float(
        np.max(
            np.abs(
                np.asarray(shared_metrics["marginal_variance_cmy"])
                - np.asarray(independent_metrics["marginal_variance_cmy"])
            )
            / np.asarray(independent_metrics["marginal_variance_cmy"])
        )
    )
    gates = contract["gates"]
    results = {
        "chroma_reduction": chroma_ratio
        <= gates["maximum_shared_to_independent_chroma_noise_power_ratio"],
        "marginal_power_lower": marginal_ratio
        >= gates["minimum_shared_to_independent_luminance_noise_power_ratio"],
        "marginal_power_upper": marginal_ratio
        <= gates["maximum_shared_to_independent_luminance_noise_power_ratio"],
        "marginal_variance": marginal_relative
        <= gates["maximum_marginal_variance_relative_difference"],
        "repeat_identity": bool(np.array_equal(shared, shared_repeat)),
        "finite_bounded_transmittance": bool(
            np.all(np.isfinite(shared))
            and np.all((np.exp(-shared) > 0.0) & (np.exp(-shared) <= 1.0))
        ),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "shared_density_sha256": hashlib.sha256(
            shared.astype("<f4").tobytes()
        ).hexdigest(),
        "shared": shared_metrics,
        "independent": independent_metrics,
        "display_rgb_additive": display_metrics,
        "shared_to_independent_chroma_noise_power_ratio": chroma_ratio,
        "shared_to_independent_marginal_noise_power_ratio": marginal_ratio,
        "maximum_marginal_variance_relative_difference": marginal_relative,
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4df_cross_layer_neutral_nps_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
