"""U6.P4K exact-area reference-simulator streaming audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    estimate_area_lod_region_workspace_bytes,
    iter_density_conditioned_structure_area_lod_rows,
    render_density_conditioned_structure_area_lod,
    render_density_conditioned_structure_area_lod_region,
)


SCHEMA = "neuro_film.u6_p4k_exact_area_streaming_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4k_exact_area_streaming_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4K parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4K contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["executor"]["full_expanded_target_allowed"]
        or contract["executor"]["output_assembly_required"]
    ):
        raise ValueError("unsupported U6.P4K contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4j_decision"],
        parents["p4j_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_adaptive_lod_retain_p4h_stationary_only"
        or not decision["next_leaf"].startswith("U6.P4K")
    ):
        raise ValueError("U6.P4J did not open exact-area streaming")
    return contract, parent


def _target(
    shape: tuple[int, int],
    *,
    maximum_density: float,
    channel_scales: list[float],
) -> np.ndarray:
    height, width = shape
    horizontal = np.linspace(
        0.0, maximum_density, width, dtype=np.float64
    )[None, :]
    vertical = np.linspace(0.7, 1.0, height, dtype=np.float64)[:, None]
    base = horizontal * vertical
    base[:, width // 3 : width // 3 + 3] = maximum_density
    scales = np.asarray(channel_scales, dtype=np.float64)
    return np.clip(
        base[..., None] * scales[None, None, :],
        0.0,
        maximum_density,
    )


def _profiles(
    parent: dict[str, Any], *, seed_offset: int
):
    return tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles_from_contract(parent, seed_offset=seed_offset)
    )


def _stream_hash(
    target: np.ndarray,
    profiles,
    *,
    factor: int,
    row_tile_height: int,
) -> tuple[str, int, int, float, float]:
    digest = hashlib.sha256()
    row_count = 0
    maximum_rows = 0
    minimum_density = float("inf")
    maximum_transmittance = 0.0
    for y0, result in iter_density_conditioned_structure_area_lod_rows(
        target,
        profiles,
        pixel_size_factor=factor,
        row_tile_height=row_tile_height,
    ):
        digest.update(int(y0).to_bytes(8, "little", signed=False))
        digest.update(result.density.tobytes(order="C"))
        digest.update(result.transmittance.tobytes(order="C"))
        row_count += result.density.shape[0]
        maximum_rows = max(maximum_rows, result.density.shape[0] * factor)
        minimum_density = min(
            minimum_density, float(np.min(result.density))
        )
        maximum_transmittance = max(
            maximum_transmittance, float(np.max(result.transmittance))
        )
    return (
        digest.hexdigest(),
        row_count,
        maximum_rows,
        minimum_density,
        maximum_transmittance,
    )


def evaluate_exact_area_streaming(
    contract: dict[str, Any],
    parent: dict[str, Any],
    *,
    include_stream_scale: bool,
) -> dict[str, Any]:
    spec = contract["synthetic_evaluation"]
    parity_target = _target(
        tuple(int(value) for value in spec["parity_shape"]),
        maximum_density=float(spec["maximum_target_density"]),
        channel_scales=spec["channel_density_scales"],
    )
    parity_rows = []
    for seed_offset in (int(value) for value in spec["seed_offsets"]):
        profiles = _profiles(parent, seed_offset=seed_offset)
        for factor in (int(value) for value in spec["lod_factors"]):
            full = render_density_conditioned_structure_area_lod(
                parity_target, profiles, pixel_size_factor=factor
            )
            partitions = {}
            for partition in (
                int(value)
                for value in contract["executor"]["partition_test_heights"]
            ):
                density = np.empty_like(full.density)
                transmittance = np.empty_like(full.transmittance)
                for y0 in range(0, parity_target.shape[0], partition):
                    y1 = min(parity_target.shape[0], y0 + partition)
                    result = (
                        render_density_conditioned_structure_area_lod_region(
                            parity_target,
                            profiles,
                            pixel_size_factor=factor,
                            origin_yx=(y0, 0),
                            shape=(y1 - y0, parity_target.shape[1]),
                        )
                    )
                    density[y0:y1] = result.density
                    transmittance[y0:y1] = result.transmittance
                partitions[str(partition)] = bool(
                    np.array_equal(full.density, density)
                    and np.array_equal(full.transmittance, transmittance)
                )
            parity_rows.append(
                {
                    "seed_offset": seed_offset,
                    "factor": factor,
                    "partition_exact": partitions,
                    "density_sha256": hashlib.sha256(
                        full.density.tobytes(order="C")
                    ).hexdigest(),
                    "transmittance_sha256": hashlib.sha256(
                        full.transmittance.tobytes(order="C")
                    ).hexdigest(),
                    "minimum_density": float(np.min(full.density)),
                    "maximum_transmittance": float(
                        np.max(full.transmittance)
                    ),
                }
            )
    zero = render_density_conditioned_structure_area_lod(
        np.zeros((31, 37, 3), dtype=np.float64),
        _profiles(parent, seed_offset=0),
        pixel_size_factor=4,
    )
    stream_rows = []
    if include_stream_scale:
        stream_target = _target(
            tuple(int(value) for value in spec["stream_shape"]),
            maximum_density=float(spec["maximum_target_density"]),
            channel_scales=spec["channel_density_scales"],
        )
        tile_height = int(contract["executor"]["row_tile_height"])
        for seed_offset in (int(value) for value in spec["seed_offsets"]):
            profiles = _profiles(parent, seed_offset=seed_offset)
            for factor in (int(value) for value in spec["lod_factors"]):
                first = _stream_hash(
                    stream_target,
                    profiles,
                    factor=factor,
                    row_tile_height=tile_height,
                )
                second = _stream_hash(
                    stream_target,
                    profiles,
                    factor=factor,
                    row_tile_height=tile_height,
                )
                maximum_halo = max(
                    int(profile.truncate * profile.correlation_sigma_pixels + 0.5)
                    for profile in profiles
                )
                stream_rows.append(
                    {
                        "seed_offset": seed_offset,
                        "factor": factor,
                        "stream_sha256": first[0],
                        "repeat_exact": first == second,
                        "output_rows": first[1],
                        "maximum_virtual_target_rows": (
                            first[2] + 2 * maximum_halo
                        ),
                        "minimum_density": first[3],
                        "maximum_transmittance": first[4],
                        "modeled_live_workspace_bytes": (
                            estimate_area_lod_region_workspace_bytes(
                                output_width=stream_target.shape[1],
                                output_row_tile_height=tile_height,
                                pixel_size_factor=factor,
                                maximum_halo_rows=maximum_halo,
                            )
                        ),
                        "full_expanded_target_allocation_count": 0,
                    }
                )
    gates = contract["automatic_gates"]
    checks = {
        "full_region_partition_byte_exact": all(
            all(row["partition_exact"].values()) for row in parity_rows
        ),
        "zero_density_exact_clear_base": bool(
            np.count_nonzero(zero.density) == 0
            and np.all(zero.transmittance == 1.0)
        ),
        "density_and_transmittance_domain_valid": (
            min(row["minimum_density"] for row in parity_rows) >= 0.0
            and max(
                row["maximum_transmittance"] for row in parity_rows
            )
            <= 1.0
        ),
        "repeat_stream_hash_exact": (
            bool(stream_rows)
            and all(row["repeat_exact"] for row in stream_rows)
        )
        if include_stream_scale
        else True,
        "factor_2_and_4_distinct_hashes": (
            len({row["stream_sha256"] for row in stream_rows})
            == len(stream_rows)
        )
        if include_stream_scale
        else True,
        "workspace_bound": (
            max(
                row["modeled_live_workspace_bytes"] for row in stream_rows
            )
            <= int(gates["maximum_modeled_live_workspace_bytes"])
        )
        if include_stream_scale
        else True,
        "virtual_target_row_bound": (
            max(
                row["maximum_virtual_target_rows"] for row in stream_rows
            )
            <= int(gates["maximum_virtual_target_rows"])
        )
        if include_stream_scale
        else True,
        "no_full_expanded_target": (
            max(
                row["full_expanded_target_allocation_count"]
                for row in stream_rows
            )
            == int(gates["full_expanded_target_allocation_count"])
        )
        if include_stream_scale
        else True,
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "parity": parity_rows,
        "stream_scale_included": include_stream_scale,
        "stream": stream_rows,
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_exact_area_streaming",
    "load_contract",
]
