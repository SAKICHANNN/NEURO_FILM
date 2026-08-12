"""Frozen U6.P4DD density-conditioned Gaussian cloud evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_cloud_row_stream import _profile
from src.eval.density_conditioned_cross_layer_counts import _expected
from src.film_physics.cross_layer_cloud_runtime import (
    iter_density_conditioned_cross_layer_cloud_rows,
)
from src.film_physics.cross_layer_compound_poisson import (
    density_conditioned_component_rates,
)
from src.film_physics.cross_layer_gaussian_cloud import (
    discrete_gaussian_kernel_overlap,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    for parent in contract["parents"].values():
        p = root / parent["path"]
        evidence = json.loads(p.read_text(encoding="utf-8"))
        if (
            _sha(p) != parent["sha256"]
            or evidence["decision"] != parent["required_decision"]
        ):
            raise RuntimeError("P4DD parent drift")
    return contract


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    qh, qw = fixture["quadrant_height"], fixture["quadrant_width"]
    shape = (2 * qh, 2 * qw)
    scale = np.empty((*shape, 3), dtype=np.float64)
    quadrants = (
        (slice(0, qh), slice(0, qw)),
        (slice(0, qh), slice(qw, 2 * qw)),
        (slice(qh, 2 * qh), slice(0, qw)),
        (slice(qh, 2 * qh), slice(qw, 2 * qw)),
    )
    for region, values in zip(quadrants, fixture["scale_cmy_by_quadrant"], strict=True):
        scale[region] = values
    profile = _profile(root)

    def render(tile: int) -> tuple[np.ndarray, np.ndarray]:
        rows = tuple(
            iter_density_conditioned_cross_layer_cloud_rows(
                profile, scale, seed=fixture["seed"], row_tile_height=tile
            )
        )
        return np.concatenate([x.density for _, x in rows]), np.concatenate(
            [x.transmittance for _, x in rows]
        )

    full_density, full_transmittance = render(shape[0])
    tiled_density, tiled_transmittance = render(fixture["row_partition_height"])
    repeat_density, repeat_transmittance = render(shape[0])
    overlap = np.asarray(
        [
            [
                discrete_gaussian_kernel_overlap(
                    a, b, truncate=profile.gaussian_truncate
                )
                for b in profile.gaussian_sigma_pixels_cmy
            ]
            for a in profile.gaussian_sigma_pixels_cmy
        ]
    )
    marks = np.asarray(profile.count_profile.mark_optical_density_cmy)
    margin = fixture["interior_margin"]
    metrics = []
    for index, (region, values) in enumerate(
        zip(quadrants, fixture["scale_cmy_by_quadrant"], strict=True)
    ):
        ys, xs = region
        interior = (
            full_density[
                ys.start + margin : ys.stop - margin,
                xs.start + margin : xs.stop - margin,
            ]
            .reshape(-1, 3)
            .astype(np.float64)
        )
        observed_mean = np.mean(interior, axis=0)
        observed_cov = np.cov(interior, rowvar=False, ddof=0)
        expected_mean_count, expected_count_cov = _expected(
            np.asarray(values), profile.count_profile
        )
        expected_mean = expected_mean_count * marks
        expected_cov = expected_count_cov * overlap * np.outer(marks, marks)
        observed_corr = observed_cov / np.sqrt(
            np.outer(np.diag(observed_cov), np.diag(observed_cov))
        )
        expected_corr = expected_cov / np.sqrt(
            np.outer(np.diag(expected_cov), np.diag(expected_cov))
        )
        offdiag = ~np.eye(3, dtype=bool)
        metrics.append(
            {
                "quadrant": index,
                "maximum_mean_relative_error": float(
                    np.max(np.abs(observed_mean - expected_mean) / expected_mean)
                ),
                "maximum_variance_relative_error": float(
                    np.max(
                        np.abs(np.diag(observed_cov) - np.diag(expected_cov))
                        / np.diag(expected_cov)
                    )
                ),
                "maximum_correlation_absolute_error": float(
                    np.max(np.abs(observed_corr[offdiag] - expected_corr[offdiag]))
                ),
                "minimum_component_rate": float(
                    np.min(
                        density_conditioned_component_rates(
                            profile.count_profile, np.asarray(values)[None, None, :]
                        )
                    )
                ),
            }
        )
    gates = contract["gates"]
    results = {
        "mean": max(x["maximum_mean_relative_error"] for x in metrics)
        <= gates["maximum_density_mean_relative_error"],
        "variance": max(x["maximum_variance_relative_error"] for x in metrics)
        <= gates["maximum_density_variance_relative_error"],
        "correlation": max(x["maximum_correlation_absolute_error"] for x in metrics)
        <= gates["maximum_density_correlation_absolute_error"],
        "partition_identity": bool(
            np.array_equal(full_density, tiled_density)
            and np.array_equal(full_transmittance, tiled_transmittance)
        ),
        "repeat_identity": bool(
            np.array_equal(full_density, repeat_density)
            and np.array_equal(full_transmittance, repeat_transmittance)
        ),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "density_sha256": hashlib.sha256(
            full_density.astype("<f4").tobytes()
        ).hexdigest(),
        "transmittance_sha256": hashlib.sha256(
            full_transmittance.astype("<f4").tobytes()
        ).hexdigest(),
        "quadrants": metrics,
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dd_density_conditioned_cross_layer_cloud_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
