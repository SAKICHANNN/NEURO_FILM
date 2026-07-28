"""Development/confirmatory audit for the U6.P4 copula compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_developed_structure import load_contract as load_parent
from src.film_physics import (
    MarginalProfile,
    build_bw_silver_context,
    build_colour_dye_cloud_context,
    render_developed_structure,
    render_marginal,
    render_marginal_region,
)


SCHEMA = "neuro_film.u6_p4_structure_compiler_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4 contract")
    return payload


def _reference(parent: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(value) for value in parent["input_shape"])
    zoom = int(parent["output_zoom"])
    pitch = float(parent["output_pixel_pitch_um"])
    samples = int(parent["monte_carlo_samples"])
    colour_config = parent["colour_dye_cloud"]
    target = np.broadcast_to(
        np.asarray(colour_config["target_density_cmy"], dtype=np.float64),
        (height, width, 3),
    ).copy()
    colour = render_developed_structure(
        build_colour_dye_cloud_context(
            target,
            radius_um_cmy=tuple(colour_config["radius_um_cmy"]),
            mark_optical_density_cmy=tuple(
                colour_config["mark_optical_density_cmy"]
            ),
            output_zoom=zoom,
            output_pixel_pitch_um=pitch,
            monte_carlo_samples=samples,
            seed=seed,
        )
    ).values
    bw_config = parent["bw_metallic_silver"]
    bw = render_developed_structure(
        build_bw_silver_context(
            np.full(
                (height, width), float(bw_config["target_density"]), np.float64
            ),
            radius_um=float(bw_config["radius_um"]),
            output_zoom=zoom,
            output_pixel_pitch_um=pitch,
            monte_carlo_samples=samples,
            seed=seed + 1,
        )
    ).values[..., 0]
    return colour, bw


def _acf(values: np.ndarray, lags: list[list[int]]) -> list[float]:
    centered = values.astype(np.float64) - float(np.mean(values))
    variance = float(np.mean(np.square(centered)))
    result = []
    for dy, dx in lags:
        a = centered[
            max(0, dy) : centered.shape[0] + min(0, dy),
            max(0, dx) : centered.shape[1] + min(0, dx),
        ]
        b = centered[
            max(0, -dy) : centered.shape[0] + min(0, -dy),
            max(0, -dx) : centered.shape[1] + min(0, -dx),
        ]
        result.append(float(np.mean(a * b) / variance))
    return result


def _nps_bands(values: np.ndarray, bands: list[list[float]]) -> list[float]:
    centered = values.astype(np.float64) - float(np.mean(values))
    power = np.square(np.abs(np.fft.fft2(centered)))
    fy = np.fft.fftfreq(values.shape[0])[:, None]
    fx = np.fft.fftfreq(values.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total = float(np.sum(power))
    return [
        float(np.sum(power[(radius >= low) & (radius < high)]) / total)
        for low, high in bands
    ]


def _compile_layer(
    development: np.ndarray,
    *,
    family: str,
    sigma_grid: list[float],
    lags: list[list[int]],
    seed: int,
) -> MarginalProfile:
    mean = float(np.mean(development))
    variance = float(np.var(development))
    target_acf = np.asarray(_acf(development, lags))
    best: tuple[float, float] | None = None
    for sigma in sigma_grid:
        profile = MarginalProfile(family, mean, variance, sigma, seed)
        candidate = render_marginal(profile, development.shape)
        loss = float(np.mean(np.abs(np.asarray(_acf(candidate, lags)) - target_acf)))
        item = (loss, sigma)
        if best is None or item < best:
            best = item
    assert best is not None
    return MarginalProfile(family, mean, variance, best[1], seed)


def evaluate_structure_compiler(
    parent: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    split = contract["split"]
    development_colour, development_bw = _reference(
        parent, int(split["development_reference_seed"])
    )
    confirmation_colour, confirmation_bw = _reference(
        parent, int(split["confirmatory_reference_seed"])
    )
    margin = int(contract["metrics"]["interior_margin_pixels"])
    development_colour = development_colour[margin:-margin, margin:-margin]
    development_bw = development_bw[margin:-margin, margin:-margin]
    confirmation_colour = confirmation_colour[margin:-margin, margin:-margin]
    confirmation_bw = confirmation_bw[margin:-margin, margin:-margin]
    lags = contract["metrics"]["acf_lags"]
    bands = contract["metrics"]["radial_nps_bands_cycles_per_pixel"]
    sigma_grid = [float(value) for value in contract["candidate"]["correlation_sigma_grid_pixels"]]
    base_seed = int(split["candidate_seed"])
    profiles = [
        _compile_layer(
            development_colour[..., layer],
            family="gamma-density",
            sigma_grid=sigma_grid,
            lags=lags,
            seed=base_seed + layer,
        )
        for layer in range(3)
    ]
    profiles.append(
        _compile_layer(
            development_bw,
            family="beta-transmittance",
            sigma_grid=sigma_grid,
            lags=lags,
            seed=base_seed + 3,
        )
    )
    references = [
        confirmation_colour[..., 0],
        confirmation_colour[..., 1],
        confirmation_colour[..., 2],
        confirmation_bw,
    ]
    candidates = [
        render_marginal(profile, reference.shape)
        for profile, reference in zip(profiles, references, strict=True)
    ]
    layer_metrics = []
    for profile, reference, candidate in zip(
        profiles, references, candidates, strict=True
    ):
        ref_mean = float(np.mean(reference))
        ref_variance = float(np.var(reference))
        candidate_mean = float(np.mean(candidate))
        candidate_variance = float(np.var(candidate))
        ref_acf = _acf(reference, lags)
        candidate_acf = _acf(candidate, lags)
        ref_nps = _nps_bands(reference, bands)
        candidate_nps = _nps_bands(candidate, bands)
        layer_metrics.append(
            {
                "family": profile.family,
                "compiled_mean": profile.mean,
                "compiled_variance": profile.variance,
                "selected_sigma_pixels": profile.correlation_sigma_pixels,
                "mean_abs_error": abs(candidate_mean - ref_mean),
                "variance_ratio": candidate_variance / ref_variance,
                "acf_abs_error_max": float(
                    np.max(np.abs(np.asarray(candidate_acf) - np.asarray(ref_acf)))
                ),
                "nps_band_relative_error_max": float(
                    np.max(
                        np.abs(np.asarray(candidate_nps) - np.asarray(ref_nps))
                        / np.maximum(np.asarray(ref_nps), 1e-12)
                    )
                ),
                "reference_acf": ref_acf,
                "candidate_acf": candidate_acf,
                "reference_nps_bands": ref_nps,
                "candidate_nps_bands": candidate_nps,
            }
        )
    partition = []
    for profile, candidate in zip(profiles, candidates, strict=True):
        split_row = candidate.shape[0] // 2
        pieces = [
            render_marginal_region(
                profile,
                candidate.shape,
                origin_yx=(0, 0),
                shape=(split_row, candidate.shape[1]),
            ),
            render_marginal_region(
                profile,
                candidate.shape,
                origin_yx=(split_row, 0),
                shape=(candidate.shape[0] - split_row, candidate.shape[1]),
            ),
        ]
        partition.append(np.array_equal(np.concatenate(pieces), candidate))
    repeat = [
        np.array_equal(render_marginal(profile, candidate.shape), candidate)
        for profile, candidate in zip(profiles, candidates, strict=True)
    ]
    gates = contract["automatic_gates"]
    decisions = {
        "mean": max(item["mean_abs_error"] for item in layer_metrics)
        <= float(gates["mean_abs_error_max"]),
        "variance": min(item["variance_ratio"] for item in layer_metrics)
        >= float(gates["variance_ratio_min"])
        and max(item["variance_ratio"] for item in layer_metrics)
        <= float(gates["variance_ratio_max"]),
        "acf": max(item["acf_abs_error_max"] for item in layer_metrics)
        <= float(gates["acf_abs_error_max"]),
        "nps": max(item["nps_band_relative_error_max"] for item in layer_metrics)
        <= float(gates["nps_band_relative_error_max"]),
        "repeat": all(repeat),
        "partition": all(partition),
        "domain": bool(
            all(np.all(np.isfinite(value)) and np.all(value > 0.0) for value in candidates)
            and np.all(candidates[-1] < 1.0)
        ),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p4_structure_compiler_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "layer_metrics": layer_metrics,
        "repeat_exact": repeat,
        "partition_exact": partition,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
        "negative_control": contract["frozen_negative_control"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
