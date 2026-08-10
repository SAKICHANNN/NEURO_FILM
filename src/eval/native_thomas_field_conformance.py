"""Build and evaluate the U6.P8BS native Thomas-field primitive."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.finite_support_thomas import (
    render_finite_support_thomas_region,
)
from src.film_physics.native_thomas_field import (
    NativeThomasFieldProfile,
    load_native_thomas_field_library,
    render_native_thomas_field,
)

SCHEMA = "neuro_film.u6_p8bs_native_thomas_field_contract.v1"


class NativeThomasConformanceError(RuntimeError):
    """Raised when the frozen native conformance contract drifts."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or contract["candidate"].get("additional_model_capacity_allowed") is not False
        or contract["candidate"].get("parameter_refit_allowed") is not False
        or contract["candidate"].get("realized_variance_normalization_allowed")
        is not False
    ):
        raise NativeThomasConformanceError("P8BS contract drift")
    for binding in contract["parents"].values():
        path = root / str(binding["path"])
        if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
            raise NativeThomasConformanceError("P8BS parent hash drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decision") != binding["required_decision"]:
            raise NativeThomasConformanceError("P8BS parent decision drift")


def profile_from_contract(contract: Mapping[str, Any]) -> NativeThomasFieldProfile:
    candidate = contract["candidate"]
    return NativeThomasFieldProfile(
        particle_sigma_pixels=float(candidate["particle_sigma_samples"]),
        cluster_sigma_pixels=float(candidate["cluster_sigma_samples"]),
        mean_offspring=float(candidate["mean_offspring"]),
        truncate=float(candidate["truncate_sigma"]),
        component_seeds=tuple(int(value) for value in candidate["component_seeds"]),
        realization_seed=int(candidate["realization_seed"]),
    )


def build_llvm_mingw_dll(
    root: Path, output_dir: Path, clang: Path
) -> dict[str, str]:
    source = root / "native/film_physics/nf_thomas_field_f32_v1.c"
    header = root / "native/film_physics/nf_thomas_field_f32_v1.h"
    dll = output_dir / "nf_thomas_field_f32_llvm_v1.dll"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(clang),
            "--target=x86_64-w64-windows-gnu",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-model=strict",
            "-shared",
            str(source),
            "-o",
            str(dll),
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasConformanceError(
            "LLVM-MinGW build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-20260616-clang-22.1.8-x64",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(source),
        "header_sha256": sha256_file(header),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def _failure_atomic(library: ctypes.CDLL, profile: NativeThomasFieldProfile) -> bool:
    abi = profile.as_abi()
    workspace = np.full(48, np.float32(-17.0))
    output = np.full(16, np.float32(-13.0))
    raw_mean = ctypes.c_double(-11.0)
    workspace_before = workspace.tobytes()
    output_before = output.tobytes()
    status = library.nf_thomas_field_f32_apply_v1(
        ctypes.byref(abi),
        4,
        4,
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        workspace.size - 1,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.size,
        ctypes.byref(raw_mean),
    )
    return (
        status != 0
        and workspace.tobytes() == workspace_before
        and output.tobytes() == output_before
        and raw_mean.value == -11.0
    )


def evaluate_conformance(
    root: Path,
    contract: Mapping[str, Any],
    *,
    output_dir: Path,
    clang: Path,
) -> dict[str, Any]:
    validate_contract(root, contract)
    msvc = build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_thomas_field_f32_v1.c",
        header_relative="native/film_physics/nf_thomas_field_f32_v1.h",
        basename="nf_thomas_field_f32_msvc_v1",
    )
    llvm = build_llvm_mingw_dll(root, output_dir, clang)
    profile = profile_from_contract(contract)
    shape = tuple(int(value) for value in contract["conformance"]["shape"])
    libraries = {
        "msvc": load_native_thomas_field_library(Path(msvc["dll_path"])),
        "llvm_mingw": load_native_thomas_field_library(Path(llvm["dll_path"])),
    }
    reference = render_finite_support_thomas_region(
        shape,
        origin_yx=(0, 0),
        shape=shape,
        particle_sigma_pixels=profile.particle_sigma_pixels,
        cluster_sigma_pixels=profile.cluster_sigma_pixels,
        mean_offspring=profile.mean_offspring,
        component_seeds=profile.component_seeds,
        realization_seed=profile.realization_seed,
        truncate=profile.truncate,
    )
    reference = reference - np.mean(reference, dtype=np.float64)
    rows: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, library in libraries.items():
        first, raw_mean, workspace_bytes = render_native_thomas_field(
            library, profile, shape
        )
        second, second_mean, _ = render_native_thomas_field(library, profile, shape)
        outputs[name] = first
        delta = first.astype(np.float64) - reference
        rows[name] = {
            "output_sha256": sha256_bytes(first.tobytes()),
            "repeat_exact": bool(np.array_equal(first, second)),
            "raw_mean_repeat_exact": raw_mean == second_mean,
            "raw_mean": raw_mean,
            "maximum_absolute_error": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "correlation": float(np.corrcoef(first.ravel(), reference.ravel())[0, 1]),
            "absolute_projected_mean": abs(float(np.mean(first, dtype=np.float64))),
            "variance_ratio": float(np.var(first) / np.var(reference)),
            "failure_atomic": _failure_atomic(library, profile),
            "workspace_bytes": workspace_bytes,
        }
    cross = float(np.max(np.abs(outputs["msvc"] - outputs["llvm_mingw"])))
    gates = contract["conformance"]
    gate_results = {
        "reference_maximum": max(row["maximum_absolute_error"] for row in rows.values())
        <= float(gates["maximum_absolute_error_vs_float64_reference"]),
        "reference_rmse": max(row["rmse"] for row in rows.values())
        <= float(gates["maximum_rmse_vs_float64_reference"]),
        "reference_correlation": min(row["correlation"] for row in rows.values())
        >= float(gates["minimum_correlation_vs_float64_reference"]),
        "projected_mean": max(row["absolute_projected_mean"] for row in rows.values())
        <= float(gates["maximum_absolute_projected_mean"]),
        "variance": all(
            float(gates["minimum_variance_ratio"])
            <= row["variance_ratio"]
            <= float(gates["maximum_variance_ratio"])
            for row in rows.values()
        ),
        "cross_compiler": cross <= float(gates["maximum_cross_compiler_absolute_error"]),
        "repeat": all(row["repeat_exact"] for row in rows.values()),
        "failure_atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    stable = {
        "schema": "neuro_film.u6_p8bs_native_thomas_field_conformance.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(
            root / "configs/u6_p8bs_native_thomas_field_v1.json"
        ),
        "profile": profile.__dict__,
        "shape": list(shape),
        "reference_sha256": sha256_bytes(
            np.ascontiguousarray(reference, dtype="<f8").tobytes()
        ),
        "toolchains": {"msvc": msvc, "llvm_mingw": llvm},
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = sha256_bytes(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return {**stable, "stable_evidence_id": stable_id}


__all__ = [
    "NativeThomasConformanceError",
    "build_llvm_mingw_dll",
    "canonical_bytes",
    "evaluate_conformance",
    "profile_from_contract",
    "validate_contract",
]
