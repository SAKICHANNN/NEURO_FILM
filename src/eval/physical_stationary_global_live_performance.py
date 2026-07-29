"""U6.P4O stationary global P4H live-performance audit."""

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
    compile_two_cumulant_profiles,
    iter_density_conditioned_structure_rows,
    render_density_conditioned_structure,
)


SCHEMA = (
    "neuro_film.u6_p4o_stationary_global_live_performance_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u6_p4o_stationary_global_live_performance_report.v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4O parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4O contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    executor = contract["executor"]
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or executor["full_output_assembly_allowed"]
        or executor["full_target_allocation_allowed"]
        or executor["network_allowed"]
        or executor["gpu_allowed"]
    ):
        raise ValueError("unsupported U6.P4O contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4n_decision"],
        parents["p4n_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_threshold_router_retain_global_p4h_and_exact_reference"
        or not decision["next_leaf"].startswith("U6.P4O")
    ):
        raise ValueError("U6.P4N did not open stationary performance")
    return contract, parent


def compiled_profiles(
    contract: dict[str, Any], parent: dict[str, Any]
) -> tuple:
    base = tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles_from_contract(
            parent, seed_offset=int(contract["executor"]["seed_offset"])
        )
    )
    return compile_two_cumulant_profiles(
        base,
        pixel_size_factor=int(
            contract["executor"]["pixel_size_factor"]
        ),
    )


def stationary_target(
    shape: tuple[int, int], density: list[float]
) -> np.ndarray:
    base = np.asarray(density, dtype=np.float64).reshape(1, 1, 3)
    target = np.broadcast_to(base, (*shape, 3))
    target.setflags(write=False)
    return target


def stream_hash(
    target: np.ndarray,
    profiles: tuple,
    *,
    row_tile_height: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    maximum_rows = 0
    minimum_density = float("inf")
    maximum_transmittance = 0.0
    for y0, result in iter_density_conditioned_structure_rows(
        target, profiles, row_tile_height=row_tile_height
    ):
        digest.update(int(y0).to_bytes(8, "little", signed=False))
        digest.update(result.density.tobytes(order="C"))
        digest.update(result.transmittance.tobytes(order="C"))
        rows += result.density.shape[0]
        maximum_rows = max(maximum_rows, result.density.shape[0])
        minimum_density = min(
            minimum_density, float(np.min(result.density))
        )
        maximum_transmittance = max(
            maximum_transmittance,
            float(np.max(result.transmittance)),
        )
    return {
        "stream_sha256": digest.hexdigest(),
        "output_rows": rows,
        "maximum_live_output_rows": maximum_rows,
        "minimum_density": minimum_density,
        "maximum_transmittance": maximum_transmittance,
    }


def small_partition_parity(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    executor = contract["executor"]
    target = stationary_target(
        tuple(int(value) for value in executor["small_parity_shape"]),
        executor["target_density"],
    )
    profiles = compiled_profiles(contract, parent)
    full = render_density_conditioned_structure(target, profiles)
    rows = {}
    for height in executor["partition_test_heights"]:
        streamed = stream_hash(
            target, profiles, row_tile_height=int(height)
        )
        density_parts = []
        transmittance_parts = []
        for y0, result in iter_density_conditioned_structure_rows(
            target, profiles, row_tile_height=int(height)
        ):
            density_parts.append(result.density)
            transmittance_parts.append(result.transmittance)
        density = np.concatenate(density_parts, axis=0)
        transmittance = np.concatenate(transmittance_parts, axis=0)
        rows[str(height)] = {
            "byte_exact": bool(
                np.array_equal(density, full.density)
                and np.array_equal(transmittance, full.transmittance)
            ),
            "stream_sha256": streamed["stream_sha256"],
        }
    return {"partitions": rows}


def evaluate_live_records(
    contract: dict[str, Any],
    parity: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = contract["automatic_gates"]
    expected = int(contract["executor"]["fresh_process_runs"])
    if len(runs) != expected:
        raise ValueError("U6.P4O run count does not match contract")
    hashes = [row["stream_sha256"] for row in runs]
    checks = {
        "small_partition_byte_exact": all(
            row["byte_exact"]
            for row in parity["partitions"].values()
        )
        == bool(gates["small_partition_byte_exact"]),
        "repeat_stream_hash_exact": len(set(hashes)) == 1
        and bool(hashes[0])
        == bool(gates["repeat_stream_hash_exact"]),
        "output_rows": all(
            int(row["output_rows"]) == int(gates["output_rows"])
            for row in runs
        ),
        "process_tree_peak_rss": max(
            int(row["process_tree_peak_rss_bytes"]) for row in runs
        )
        <= int(gates["maximum_process_tree_peak_rss_bytes"]),
        "worker_wall": max(float(row["wall_seconds"]) for row in runs)
        <= float(gates["maximum_worker_wall_seconds"]),
        "liveness_samples": min(
            int(row["liveness_samples"]) for row in runs
        )
        >= int(gates["minimum_liveness_samples"]),
        "exit_code": all(
            int(row["exit_code"]) == int(gates["exit_code"])
            for row in runs
        ),
        "stderr_empty": all(
            int(row["stderr_bytes"]) == 0 for row in runs
        )
        == bool(gates["stderr_empty"]),
        "owned_temp_cleanup": sum(
            int(row["owned_temp_residue_count"]) for row in runs
        )
        == int(gates["owned_temp_residue_count"]),
        "physical_domain": min(
            float(row["minimum_density"]) for row in runs
        )
        >= 0.0
        and max(float(row["maximum_transmittance"]) for row in runs)
        <= 1.0,
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "small_parity": parity,
        "runs": runs,
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
    "compiled_profiles",
    "evaluate_live_records",
    "load_contract",
    "small_partition_parity",
    "stationary_target",
    "stream_hash",
]
