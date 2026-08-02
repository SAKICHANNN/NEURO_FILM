"""U6.P4BU spectrum, covariance and row-partition evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_nps import cosine_similarity
from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.finite_support_thomas import (
    gaussian_kernel_1d,
    render_finite_support_thomas_region,
)
from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps

SCHEMA = "neuro_film.u6_p4bu_finite_support_thomas_filter_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bu_finite_support_thomas_filter_report.v1"


class FiniteSupportThomasFilterError(RuntimeError):
    """Raised when frozen P4BU evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise FiniteSupportThomasFilterError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FiniteSupportThomasFilterError("parent must be a JSON object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    profile = contract.get("profile", {})
    kernel = contract.get("filter", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or profile.get("combined_sigma_pixels") != math.hypot(
            profile.get("particle_sigma_pixels", 0.0),
            profile.get("cluster_sigma_pixels", 0.0),
        )
        or profile.get("component_seeds")
        != [2611923443488327891, 11400714819323198485]
        or kernel.get("truncate_sigma") != 4.0
        or kernel.get("particle_radius_pixels") != 5
        or kernel.get("combined_radius_pixels") != 7
        or kernel.get("maximum_halo_pixels") != 7
        or kernel.get("realized_field_renormalization_allowed") is not False
        or evaluation.get("field_shape") != [257, 263]
        or evaluation.get("row_partitions") != [1, 7, 31, 64, 127]
        or evaluation.get("maximum_radial_signature_cosine_error") != 0.005
        or evaluation.get("maximum_covariance_absolute_error") != 0.01
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise FiniteSupportThomasFilterError("P4BU frozen contract drift")


def _frequency_response_1d(kernel: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    return np.sum(
        kernel[:, None]
        * np.cos(2.0 * math.pi * offsets[:, None] * frequencies[None, :]),
        axis=0,
    )


def _spectra(
    shape: tuple[int, int], profile: Mapping[str, Any], truncate: float
) -> tuple[np.ndarray, np.ndarray]:
    fy_axis = np.fft.fftfreq(shape[0])
    fx_axis = np.fft.fftfreq(shape[1])
    fy = fy_axis[:, None]
    fx = fx_axis[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    target = thomas_cluster_gaussian_mark_nps(
        radius,
        float(profile["particle_sigma_pixels"]),
        float(profile["cluster_sigma_pixels"]),
        float(profile["mean_offspring"]),
    )
    particle_kernel = gaussian_kernel_1d(
        float(profile["particle_sigma_pixels"]), truncate
    )
    combined_kernel = gaussian_kernel_1d(
        float(profile["combined_sigma_pixels"]), truncate
    )
    particle_y = _frequency_response_1d(particle_kernel, fy_axis)[:, None]
    particle_x = _frequency_response_1d(particle_kernel, fx_axis)[None, :]
    combined_y = _frequency_response_1d(combined_kernel, fy_axis)[:, None]
    combined_x = _frequency_response_1d(combined_kernel, fx_axis)[None, :]
    approximate = np.square(particle_y * particle_x) + float(
        profile["mean_offspring"]
    ) * np.square(combined_y * combined_x)
    target = target / float(np.mean(target, dtype=np.float64))
    approximate = approximate / float(np.mean(approximate, dtype=np.float64))
    return target, approximate


def _radial_signature(
    spectrum: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    fy = np.fft.fftfreq(spectrum.shape[0])[:, None]
    fx = np.fft.fftfreq(spectrum.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    bands = []
    for lower, upper in pairwise(edges):
        selected = spectrum[(radius >= lower) & (radius < upper)]
        if selected.size == 0:
            raise FiniteSupportThomasFilterError("empty radial band")
        bands.append(float(np.mean(selected, dtype=np.float64)))
    values = np.asarray(bands, dtype=np.float64)
    values /= float(np.sum(values, dtype=np.float64))
    values = np.log(values)
    values -= float(np.mean(values))
    values /= float(np.linalg.norm(values))
    values.setflags(write=False)
    return values


def evaluate_finite_support_thomas_filter(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4bs = _load_bound(root, parents["p4bs_decision"])
    p4bt = _load_bound(root, parents["p4bt_decision"])
    if (
        p4bs.get("decision") != parents["p4bs_decision"]["required_decision"]
        or p4bt.get("decision") != parents["p4bt_decision"]["required_decision"]
    ):
        raise FiniteSupportThomasFilterError("P4BU parent decision drift")
    profile = contract["profile"]
    filtering = contract["filter"]
    evaluation = contract["evaluation"]
    spectral_shape = tuple(int(v) for v in evaluation["spectral_grid_shape"])
    target, approximate = _spectra(
        spectral_shape, profile, float(filtering["truncate_sigma"])
    )
    edges = np.asarray(
        evaluation["radial_band_edges_cycles_per_pixel"], dtype=np.float64
    )
    target_signature = _radial_signature(target, edges)
    approximate_signature = _radial_signature(approximate, edges)
    radial_error = float(1.0 - cosine_similarity(target_signature, approximate_signature))
    target_covariance = np.fft.ifft2(target).real
    approximate_covariance = np.fft.ifft2(approximate).real
    lags = evaluation["covariance_lags_yx"]
    covariance_error = max(
        abs(
            float(
                target_covariance[dy, dx] - approximate_covariance[dy, dx]
            )
        )
        for dy, dx in lags
    )

    full_shape = tuple(int(v) for v in evaluation["field_shape"])
    border = int(evaluation["interior_border_pixels"])
    component_seeds = tuple(int(v) for v in profile["component_seeds"])
    rows: list[dict[str, Any]] = []
    means: list[float] = []
    variance_errors: list[float] = []
    parity: list[bool] = []
    repeat: list[bool] = []
    hashes: list[str] = []
    for seed in evaluation["seeds"]:
        kwargs = {
            "full_shape": full_shape,
            "particle_sigma_pixels": float(profile["particle_sigma_pixels"]),
            "cluster_sigma_pixels": float(profile["cluster_sigma_pixels"]),
            "mean_offspring": float(profile["mean_offspring"]),
            "component_seeds": component_seeds,
            "realization_seed": int(seed),
            "truncate": float(filtering["truncate_sigma"]),
        }
        full = render_finite_support_thomas_region(
            origin_yx=(0, 0), shape=full_shape, **kwargs
        )
        repeated = render_finite_support_thomas_region(
            origin_yx=(0, 0), shape=full_shape, **kwargs
        )
        seed_parity = True
        for row_height in evaluation["row_partitions"]:
            assembled = np.empty_like(full)
            for y0 in range(0, full_shape[0], int(row_height)):
                height = min(int(row_height), full_shape[0] - y0)
                assembled[y0 : y0 + height] = render_finite_support_thomas_region(
                    origin_yx=(y0, 0), shape=(height, full_shape[1]), **kwargs
                )
            seed_parity &= bool(np.array_equal(full, assembled))
        interior = full[border:-border, border:-border]
        mean = abs(float(np.mean(interior, dtype=np.float64)))
        variance = float(np.var(interior, dtype=np.float64))
        digest = hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest()
        seed_repeat = bool(np.array_equal(full, repeated))
        means.append(mean)
        variance_errors.append(abs(variance - 1.0))
        parity.append(seed_parity)
        repeat.append(seed_repeat)
        hashes.append(digest)
        rows.append(
            {
                "seed": int(seed),
                "field_sha256": digest,
                "interior_absolute_mean": mean,
                "interior_variance": variance,
                "interior_variance_absolute_error": abs(variance - 1.0),
                "row_partition_exact": seed_parity,
                "repeat_exact": seed_repeat,
            }
        )
    checks = {
        "radial_signature": radial_error
        <= evaluation["maximum_radial_signature_cosine_error"],
        "covariance": covariance_error
        <= evaluation["maximum_covariance_absolute_error"],
        "interior_variance": max(variance_errors)
        <= evaluation["maximum_interior_variance_absolute_error"],
        "interior_mean": max(means) <= evaluation["maximum_interior_absolute_mean"],
        "row_partition_exact": all(parity),
        "repeat_exact": all(repeat),
        "seed_distinct_fields": len(set(hashes)) == len(hashes),
        "finite": all(math.isfinite(value) for value in means + variance_errors),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bu_finite_support_thomas_filter_v1.json",
            "sha256",
        ),
        "profile": profile,
        "particle_radius_pixels": filtering["particle_radius_pixels"],
        "combined_radius_pixels": filtering["combined_radius_pixels"],
        "maximum_halo_pixels": filtering["maximum_halo_pixels"],
        "radial_signature_cosine_error": radial_error,
        "maximum_covariance_absolute_error": covariance_error,
        "maximum_interior_variance_absolute_error": max(variance_errors),
        "maximum_interior_absolute_mean": max(means),
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_finite_support_thomas_unit_reference_filter"
            if automatic_pass
            else "close_finite_support_thomas_filter"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FiniteSupportThomasFilterError",
    "evaluate_finite_support_thomas_filter",
    "write_report",
]
