"""Evaluate cross-layer cloud covariance across square-aperture LODs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_aperture_lod import (
    block_aperture_mean,
    centered_kernel_inner_product,
    gaussian_aperture_effective_kernel,
)
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_cross_layer_poisson_region,
)
from src.film_physics.cross_layer_gaussian_cloud import (
    render_cross_layer_gaussian_cloud_density,
)
from src.film_physics.density_conditioned_structure import counter_poisson_rate_field

SCHEMA = "neuro_film.u6_p4db_cross_layer_aperture_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4db_cross_layer_aperture_lod_report.v1"


class CrossLayerApertureLODError(RuntimeError):
    pass


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    c = json.loads(path.read_text())
    p = c["parent"]
    pp = root / p["path"]
    pv = json.loads(pp.read_text())
    if (
        c.get("schema") != SCHEMA
        or c.get("status") != "contract_frozen_implementation_ready"
    ):
        raise CrossLayerApertureLODError("P4DB contract drift")
    if sha256_file(pp) != p["sha256"] or pv.get("decision") != p["required_decision"]:
        raise CrossLayerApertureLODError("P4DB parent drift")
    return c


def _profile(f: dict[str, Any], seed: int) -> CrossLayerPoissonProfile:
    return CrossLayerPoissonProfile(
        tuple(f["marginal_count_rates_cmy"]),
        f["shared_all_rate"],
        tuple(f["shared_pair_rates_cm_cy_my"]),
        tuple(f["mark_optical_density_cmy"]),
        seed,
        f["component_seed_stride"],
    )


def _counts(
    p: CrossLayerPoissonProfile, shape: tuple[int, int], independent: bool
) -> np.ndarray:
    if not independent:
        return sample_cross_layer_poisson_region(
            p, shape, origin_yx=(0, 0), shape=shape
        )
    return np.stack(
        [
            counter_poisson_rate_field(
                np.full(shape, r),
                shape,
                origin_yx=(0, 0),
                seed=p.seed + (i + 30) * p.component_seed_stride,
                maximum_rate=r,
            )
            for i, r in enumerate(p.marginal_rates_cmy)
        ],
        axis=-1,
    )


def _analytic(
    p: CrossLayerPoissonProfile,
    sigmas: tuple[float, float, float],
    factor: int,
    truncate: float,
) -> tuple[np.ndarray, np.ndarray]:
    kernels = [
        gaussian_aperture_effective_kernel(s, factor, truncate=truncate) for s in sigmas
    ]
    rates = np.array(p.marginal_rates_cmy)
    a = p.shared_all_rate
    cm, cy, my = p.shared_pair_rates_cm_cy_my
    shared = np.array(
        [
            [rates[0], a + cm, a + cy],
            [a + cm, rates[1], a + my],
            [a + cy, a + my, rates[2]],
        ]
    )
    overlap = np.array(
        [
            [centered_kernel_inner_product(kernels[i], kernels[j]) for j in range(3)]
            for i in range(3)
        ]
    )
    cov = (
        shared
        * overlap
        * np.outer(p.mark_optical_density_cmy, p.mark_optical_density_cmy)
    )
    var = np.diag(cov)
    corr = cov / np.sqrt(var[:, None] * var[None, :])
    np.fill_diagonal(corr, 1)
    return var, corr


def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = x.reshape(-1, 3)
    return np.var(f, axis=0), np.corrcoef(f, rowvar=False)


def _row(seed: int, c: dict[str, Any]) -> dict[str, Any]:
    f = c["field"]
    shape = tuple(f["shape"])
    p = _profile(f, seed)
    sigmas = tuple(f["gaussian_sigma_pixels_cmy"])
    kwargs = {
        "mark_optical_density_cmy": p.mark_optical_density_cmy,
        "sigma_pixels_cmy": sigmas,
        "truncate": f["gaussian_truncate"],
    }
    density = render_cross_layer_gaussian_cloud_density(
        _counts(p, shape, False), **kwargs
    )
    repeat = render_cross_layer_gaussian_cloud_density(
        _counts(p, shape, False), **kwargs
    )
    ind = render_cross_layer_gaussian_cloud_density(_counts(p, shape, True), **kwargs)
    rows = []
    for factor in f["aperture_factors"]:
        empirical = block_aperture_mean(density, factor)
        ivar, icorr = _stats(empirical)
        avar, acorr = _analytic(p, sigmas, factor, f["gaussian_truncate"])
        _, ncorr = _stats(block_aperture_mean(ind, factor))
        idx = np.triu_indices(3, 1)
        rows.append(
            {
                "factor": factor,
                "maximum_absolute_correlation_error": float(
                    np.max(np.abs(icorr[idx] - acorr[idx]))
                ),
                "maximum_marginal_variance_relative_error": float(
                    np.max(np.abs(ivar - avar) / avar)
                ),
                "minimum_shared_correlation": float(np.min(icorr[idx])),
                "maximum_independent_control_absolute_correlation": float(
                    np.max(np.abs(ncorr[idx]))
                ),
                "maximum_variance": float(np.max(ivar)),
            }
        )
    return {
        "seed": seed,
        "repeat_exact": bool(np.array_equal(density, repeat)),
        "apertures": rows,
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    c = load_contract(root, contract_path)
    f = c["field"]
    dev = [_row(int(s), c) for s in f["development_seeds"]]
    conf = [_row(int(s), c) for s in f["confirmation_seeds"]]
    flat = [x for r in conf for x in r["apertures"]]
    m = c["metrics"]
    worst = {
        "maximum_absolute_correlation_error": max(
            x["maximum_absolute_correlation_error"] for x in flat
        ),
        "maximum_marginal_variance_relative_error": max(
            x["maximum_marginal_variance_relative_error"] for x in flat
        ),
        "maximum_independent_control_absolute_correlation": max(
            x["maximum_independent_control_absolute_correlation"] for x in flat
        ),
        "minimum_shared_correlation": min(
            x["minimum_shared_correlation"] for x in flat
        ),
    }
    monotonic = all(
        all(
            a["maximum_variance"] >= b["maximum_variance"]
            for a, b in zip(r["apertures"], r["apertures"][1:])
        )
        for r in (*dev, *conf)
    )
    g = {
        "correlation": worst["maximum_absolute_correlation_error"]
        <= m["maximum_absolute_correlation_error"],
        "variance": worst["maximum_marginal_variance_relative_error"]
        <= m["maximum_marginal_variance_relative_error"],
        "independent": worst["maximum_independent_control_absolute_correlation"]
        <= m["maximum_independent_control_absolute_correlation"],
        "shared": worst["minimum_shared_correlation"]
        >= m["minimum_shared_correlation"],
        "monotonic": monotonic,
        "repeat": all(r["repeat_exact"] for r in (*dev, *conf)),
    }
    passed = all(g.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "confirmation_worst_metrics": worst,
        "gates": g,
        "decision": c["decision_if_pass"] if passed else c["decision_if_fail"],
        "claim_ceiling": c["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "development": dev,
        "confirmation": conf,
    }


__all__ = ["CrossLayerApertureLODError", "evaluate", "load_contract"]
