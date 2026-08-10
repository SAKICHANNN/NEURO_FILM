"""Numerical gate for the multiscale scanner-glare streaming compiler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_streaming import (
    apply_scanner_glare_separable_streaming,
    compile_scanner_glare_separable,
    scanner_glare_streaming_workspace_bytes,
)


class ScannerGlareStreamingError(ValueError):
    """Raised when the frozen compiler experiment is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "neuro_film.u6_p6zc_scanner_glare_separable_streaming_contract.v1",
        "node": "U6.P6ZC",
        "status": "contract_frozen_implementation_ready",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ScannerGlareStreamingError("contract identity drift")
    fixture = payload.get("fixture", {})
    profile = payload.get("profile", {})
    if (
        fixture.get("row_partitions") != [1, 17, 64, 193, 321]
        or profile.get("kernel_size") != 177
        or profile.get("components")
        != [
            {"weight": 0.8, "sigma_pixels": 2.0},
            {"weight": 0.2, "sigma_pixels": 12.0},
        ]
    ):
        raise ScannerGlareStreamingError("contract boundary drift")
    return payload


def _profile(payload: Mapping[str, Any]) -> MultiscaleScannerGlareProfile:
    return MultiscaleScannerGlareProfile(
        components=tuple(
            ScannerGlareComponent(
                weight=float(component["weight"]),
                sigma_pixels=float(component["sigma_pixels"]),
            )
            for component in payload["components"]
        ),
        flare_fraction=float(payload["flare_fraction"]),
        truncate_sigma=float(payload["truncate_sigma"]),
    )


def _error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "maximum_absolute": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def evaluate_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    parents = contract["parents"]
    parent_hashes = {
        "p6za_contract": _sha256(root / parents["p6za_contract"]),
        "p6za_evidence": _sha256(root / parents["p6za_evidence"]),
        "scanner_glare_core_before_compiler": parents["scanner_glare_core_sha256"],
    }
    parent_hashes_exact = (
        parent_hashes["p6za_contract"] == parents["p6za_contract_sha256"]
        and parent_hashes["p6za_evidence"] == parents["p6za_evidence_sha256"]
    )
    fixture = contract["fixture"]
    profile = _profile(contract["profile"])
    kernel_size = int(contract["profile"]["kernel_size"])
    full_kernel = compile_scanner_glare_kernel(profile, kernel_size=kernel_size)
    separable = compile_scanner_glare_separable(profile, kernel_size=kernel_size)
    rng = np.random.default_rng(int(fixture["seed"]))
    source = rng.uniform(
        0.01,
        0.99,
        size=(
            int(fixture["height"]),
            int(fixture["width"]),
            int(fixture["channels"]),
        ),
    )
    reference = apply_scanner_glare(
        source, full_kernel, flare_fraction=profile.flare_fraction
    )
    partition_rows = []
    for row_chunk in fixture["row_partitions"]:
        candidate = apply_scanner_glare_separable_streaming(
            source,
            separable,
            flare_fraction=profile.flare_fraction,
            row_chunk=int(row_chunk),
        )
        partition_rows.append(
            {
                "row_chunk": int(row_chunk),
                "reference_error": _error(candidate, reference),
                "minimum": float(np.min(candidate)),
                "maximum": float(np.max(candidate)),
            }
        )
    constant = np.full((127, 139, 3), 0.42, dtype=np.float64)
    constant_output = apply_scanner_glare_separable_streaming(
        constant, separable, flare_fraction=profile.flare_fraction, row_chunk=17
    )
    impulse = np.zeros((257, 257), dtype=np.float64)
    impulse[128, 128] = 1.0
    impulse_output = apply_scanner_glare_separable_streaming(
        impulse, separable, flare_fraction=profile.flare_fraction, row_chunk=31
    )
    workspace = scanner_glare_streaming_workspace_bytes(
        width=int(fixture["workspace_probe_width"]),
        channels=int(fixture["workspace_probe_channels"]),
        row_chunk=int(fixture["workspace_probe_rows"]),
        radius=separable.radius,
    )
    maximum_absolute = max(
        row["reference_error"]["maximum_absolute"] for row in partition_rows
    )
    maximum_rmse = max(row["reference_error"]["rmse"] for row in partition_rows)
    partition_reference = apply_scanner_glare_separable_streaming(
        source, separable, flare_fraction=profile.flare_fraction, row_chunk=321
    )
    maximum_partition_error = max(
        float(
            np.max(
                np.abs(
                    apply_scanner_glare_separable_streaming(
                        source,
                        separable,
                        flare_fraction=profile.flare_fraction,
                        row_chunk=int(row_chunk),
                    )
                    - partition_reference
                )
            )
        )
        for row_chunk in fixture["row_partitions"]
    )
    measurements = {
        "maximum_reference_absolute_error": maximum_absolute,
        "maximum_reference_rmse": maximum_rmse,
        "maximum_partition_absolute_error": maximum_partition_error,
        "constant_field_error": float(np.max(np.abs(constant_output - constant))),
        "impulse_energy_error": abs(float(np.sum(impulse_output)) - 1.0),
        "minimum_output": min(row["minimum"] for row in partition_rows),
        "maximum_output": max(row["maximum"] for row in partition_rows),
        "tracked_workspace_bytes_12mp_rgb": workspace,
    }
    frozen = contract["automatic_gates"]
    gates = {
        "parent_hashes": parent_hashes_exact,
        "reference_absolute": maximum_absolute
        <= frozen["maximum_reference_absolute_error"],
        "reference_rmse": maximum_rmse <= frozen["maximum_reference_rmse"],
        "partition": maximum_partition_error
        <= frozen["maximum_partition_absolute_error"],
        "constant": measurements["constant_field_error"]
        <= frozen["constant_field_error_maximum"],
        "impulse_energy": measurements["impulse_energy_error"]
        <= frozen["impulse_energy_error_maximum"],
        "range": measurements["minimum_output"] >= frozen["minimum_output"]
        and measurements["maximum_output"] <= frozen["maximum_output"],
        "workspace": workspace <= frozen["maximum_tracked_workspace_bytes_12mp_rgb"],
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": "neuro_film.u6_p6zc_scanner_glare_separable_streaming_result.v1",
        "node": contract["node"],
        "contract": contract_path.as_posix(),
        "contract_sha256": _sha256(contract_path),
        "parent_hashes": parent_hashes,
        "profile": contract["profile"],
        "fixture": fixture,
        "partition_rows": partition_rows,
        "measurements": measurements,
        "gates": gates,
        "automatic_pass": passed,
        "decision": (
            "retain_separable_streaming_open_performance_benchmark"
            if passed
            else "close_exact_direct_separable_scanner_glare_compiler"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = dict(report)
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
