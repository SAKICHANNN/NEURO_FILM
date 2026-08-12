"""Frozen U6.P8DG cross-layer cloud row-stream evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.cross_layer_cloud_profile import (
    CrossLayerCloudReferenceProfile,
)
from src.film_physics.cross_layer_cloud_runtime import (
    estimate_cross_layer_cloud_full_live_bytes,
    estimate_cross_layer_cloud_row_stream_live_bytes,
    iter_cross_layer_cloud_profile_rows,
)
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_cross_layer_poisson_region,
)
from src.film_physics.cross_layer_gaussian_cloud import (
    render_cross_layer_gaussian_cloud_density,
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
        or evidence["profile_identity"] != contract["parent"]["profile_identity"]
    ):
        raise RuntimeError("P8DF parent drift")
    return contract


def _profile(root: Path) -> CrossLayerCloudReferenceProfile:
    compiler_contract = json.loads(
        (
            root / "configs/u6_p8df_cross_layer_cloud_profile_compiler_v1.json"
        ).read_text()
    )
    p = compiler_contract["profile"]
    return CrossLayerCloudReferenceProfile(
        CrossLayerPoissonProfile(
            tuple(p["marginal_count_rates_cmy"]),
            p["shared_all_rate"],
            tuple(p["shared_pair_rates_cm_cy_my"]),
            tuple(p["mark_optical_density_cmy"]),
            0,
            p["component_seed_stride"],
        ),
        tuple(p["gaussian_sigma_pixels_cmy"]),
        p["gaussian_truncate"],
        tuple(p["aperture_factors"]),
        tuple(item["sha256"] for item in compiler_contract["parents"].values()),
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    profile = _profile(root)
    seeded = CrossLayerPoissonProfile(
        profile.count_profile.marginal_rates_cmy,
        profile.count_profile.shared_all_rate,
        profile.count_profile.shared_pair_rates_cm_cy_my,
        profile.count_profile.mark_optical_density_cmy,
        fixture["seed"],
        profile.count_profile.component_seed_stride,
    )
    counts = sample_cross_layer_poisson_region(
        seeded, shape, origin_yx=(0, 0), shape=shape
    )
    reference_density = render_cross_layer_gaussian_cloud_density(
        counts,
        mark_optical_density_cmy=seeded.mark_optical_density_cmy,
        sigma_pixels_cmy=profile.gaussian_sigma_pixels_cmy,
        truncate=profile.gaussian_truncate,
    )
    reference_transmittance = np.exp(-reference_density.astype(np.float64)).astype(
        np.float32
    )
    partitions = []
    all_exact = True
    ordered_complete = True
    for tile_height in fixture["row_tile_heights"]:
        density_parts = []
        transmittance_parts = []
        expected_y = 0
        for y0, result in iter_cross_layer_cloud_profile_rows(
            profile, shape, seed=fixture["seed"], row_tile_height=tile_height
        ):
            ordered_complete &= y0 == expected_y
            expected_y += result.density.shape[0]
            density_parts.append(result.density)
            transmittance_parts.append(result.transmittance)
        density = np.concatenate(density_parts, axis=0)
        transmittance = np.concatenate(transmittance_parts, axis=0)
        density_exact = bool(np.array_equal(density, reference_density))
        transmittance_exact = bool(
            np.array_equal(transmittance, reference_transmittance)
        )
        all_exact &= density_exact and transmittance_exact
        ordered_complete &= expected_y == shape[0]
        live_bytes = estimate_cross_layer_cloud_row_stream_live_bytes(
            profile, full_shape=shape, row_tile_height=tile_height
        )
        full_bytes = estimate_cross_layer_cloud_full_live_bytes(shape)
        partitions.append(
            {
                "row_tile_height": tile_height,
                "density_bit_exact": density_exact,
                "transmittance_bit_exact": transmittance_exact,
                "modeled_live_bytes": live_bytes,
                "stream_to_full_modeled_ratio": live_bytes / full_bytes,
            }
        )
    repeat = b"".join(
        result.density.astype("<f4").tobytes()
        + result.transmittance.astype("<f4").tobytes()
        for _, result in iter_cross_layer_cloud_profile_rows(
            profile,
            shape,
            seed=fixture["seed"],
            row_tile_height=fixture["row_tile_heights"][0],
        )
    )
    first = b"".join(
        result.density.astype("<f4").tobytes()
        + result.transmittance.astype("<f4").tobytes()
        for _, result in iter_cross_layer_cloud_profile_rows(
            profile,
            shape,
            seed=fixture["seed"],
            row_tile_height=fixture["row_tile_heights"][0],
        )
    )
    gates = contract["gates"]
    gate_results = {
        "density_exact": all(item["density_bit_exact"] for item in partitions),
        "transmittance_exact": all(
            item["transmittance_bit_exact"] for item in partitions
        ),
        "repeat_identity": first == repeat,
        "ordered_complete_rows": ordered_complete,
        "live_bytes": max(item["modeled_live_bytes"] for item in partitions)
        <= gates["maximum_modeled_live_bytes"],
        "live_ratio": max(item["stream_to_full_modeled_ratio"] for item in partitions)
        <= gates["maximum_stream_to_full_modeled_ratio"],
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "shape": list(shape),
        "seed": fixture["seed"],
        "reference_density_sha256": hashlib.sha256(
            reference_density.astype("<f4").tobytes()
        ).hexdigest(),
        "reference_transmittance_sha256": hashlib.sha256(
            reference_transmittance.astype("<f4").tobytes()
        ).hexdigest(),
        "partitions": partitions,
        "gates": gate_results,
        "decision": contract["decision_if_pass"]
        if all(gate_results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8dg_cross_layer_cloud_row_stream_report.v1",
        "automatic_pass": all(gate_results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
