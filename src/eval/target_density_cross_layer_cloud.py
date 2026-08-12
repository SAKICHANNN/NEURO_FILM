"""Frozen U6.P4DE typed target-density cloud evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from src.eval.cross_layer_cloud_row_stream import _profile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_target_density_cross_layer_cloud_rows,
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
        raise RuntimeError("P4DE parent drift")
    return contract


def _target(shape: tuple[int, int], maximum: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    fields = (
        0.52
        + 0.22 * np.sin(2 * np.pi * xx / shape[1])
        + 0.11 * np.cos(4 * np.pi * yy / shape[0]),
        0.48
        + 0.18 * np.cos(2 * np.pi * yy / shape[0])
        + 0.13 * np.sin(2 * np.pi * (xx + yy) / shape[1]),
        0.55
        + 0.16 * np.sin(4 * np.pi * xx / shape[1])
        - 0.12 * np.cos(2 * np.pi * yy / shape[0]),
    )
    scale = np.clip(np.stack(fields, axis=-1), 0.05, 0.95)
    return scale * maximum


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    shape = (f["height"], f["width"])
    profile = _profile(root)
    declared_maximum = np.asarray(
        f["maximum_developed_density_cmy"], dtype=np.float64
    )
    maximum = np.asarray(
        profile.count_profile.marginal_rates_cmy
    ) * np.asarray(profile.count_profile.mark_optical_density_cmy)
    if not np.allclose(declared_maximum, maximum, rtol=0.0, atol=1e-15):
        raise RuntimeError("P4DE maximum developed density drift")
    target = _target(shape, maximum)

    def render(tile: int) -> tuple[np.ndarray, np.ndarray]:
        rows = tuple(
            iter_target_density_cross_layer_cloud_rows(
                profile,
                target,
                maximum_developed_density_cmy=tuple(maximum),
                seed=f["seed"],
                row_tile_height=tile,
            )
        )
        return np.concatenate([x.density for _, x in rows]), np.concatenate(
            [x.transmittance for _, x in rows]
        )

    density, transmittance = render(shape[0])
    tiled_density, tiled_transmittance = render(f["row_partition_height"])
    expected = np.stack(
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
    margin = f["interior_margin"]
    region = np.s_[margin:-margin, margin:-margin, :]
    observed_mean = np.mean(density[region], axis=(0, 1), dtype=np.float64)
    expected_mean = np.mean(expected[region], axis=(0, 1), dtype=np.float64)
    local_observed = uniform_filter(
        density.astype(np.float64), size=(65, 65, 1), mode="wrap"
    )
    local_expected = uniform_filter(expected, size=(65, 65, 1), mode="wrap")
    mean_relative_error = float(
        np.max(np.abs(observed_mean - expected_mean) / expected_mean)
    )
    local_rmse = float(
        np.sqrt(
            np.mean(
                np.square(local_observed[region] - local_expected[region]),
                dtype=np.float64,
            )
        )
    )
    transmittance_error = float(
        np.max(
            np.abs(
                transmittance - np.exp(-density.astype(np.float64)).astype(np.float32)
            )
        )
    )
    rejected = False
    try:
        next(
            iter_target_density_cross_layer_cloud_rows(
                profile,
                target * 2.0,
                maximum_developed_density_cmy=tuple(maximum),
                seed=f["seed"],
                row_tile_height=127,
            )
        )
    except ValueError:
        rejected = True
    gates = contract["gates"]
    results = {
        "mean": mean_relative_error <= gates["maximum_interior_mean_relative_error"],
        "local_mean": local_rmse <= gates["maximum_interior_local_mean_rmse"],
        "transmittance": transmittance_error
        <= gates["maximum_transmittance_identity_error"],
        "partition_identity": bool(
            np.array_equal(density, tiled_density)
            and np.array_equal(transmittance, tiled_transmittance)
        ),
        "domain_rejection": rejected,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "density_sha256": hashlib.sha256(density.astype("<f4").tobytes()).hexdigest(),
        "transmittance_sha256": hashlib.sha256(
            transmittance.astype("<f4").tobytes()
        ).hexdigest(),
        "maximum_interior_mean_relative_error": mean_relative_error,
        "interior_local_mean_rmse": local_rmse,
        "maximum_transmittance_identity_error": transmittance_error,
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4de_target_density_cross_layer_cloud_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
