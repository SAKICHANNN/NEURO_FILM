"""Build and evaluate U6.P8BT native density-domain composition."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.native_thomas_field_conformance import (
    profile_from_contract as p8bs_profile_from_contract,
)
from src.film_physics.finite_support_thomas import render_finite_support_thomas_region
from src.film_physics.native_thomas_density import (
    load_native_thomas_density_library,
    render_native_thomas_transmittance,
)

SCHEMA = "neuro_film.u6_p8bt_native_density_thomas_transmittance_contract.v1"
SOURCE_PATHS = (
    "native/film_physics/nf_thomas_field_f32_v1.c",
    "native/film_physics/nf_thomas_density_f32_v1.c",
)
HEADER_PATHS = (
    "native/film_physics/nf_thomas_field_f32_v1.h",
    "native/film_physics/nf_thomas_density_f32_v1.h",
)


class NativeThomasDensityConformanceError(RuntimeError):
    """Raised when the P8BT contract or native conformance fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or contract["candidate"].get("p8bs_profile_unchanged") is not True
        or contract["candidate"].get("p4bw_amplitude_refit_allowed") is not False
        or contract["candidate"].get("realized_field_normalization_allowed")
        is not False
        or contract["candidate"].get("display_rgb_noise_allowed") is not False
    ):
        raise NativeThomasDensityConformanceError("P8BT contract drift")
    for binding in contract["parents"].values():
        path = root / str(binding["path"])
        if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
            raise NativeThomasDensityConformanceError("P8BT parent hash drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise NativeThomasDensityConformanceError("P8BT parent decision drift")


def _p8bs_contract(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "configs/u6_p8bs_native_thomas_field_v1.json").read_text(
            encoding="utf-8"
        )
    )


def fixture_arrays(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    base = (
        0.45
        + 0.60 * x[None, :]
        + 0.25 * y[:, None]
        + 0.05
        * np.sin(2.0 * np.pi * x[None, :])
        * np.cos(2.0 * np.pi * y[:, None])
    )
    sigma = 0.010 + 0.006 * x[None, :] + 0.004 * y[:, None]
    return (
        np.ascontiguousarray(base, dtype=np.float32),
        np.ascontiguousarray(sigma, dtype=np.float32),
    )


def float64_reference(
    root: Path, base: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    p8bs = _p8bs_contract(root)
    profile = p8bs_profile_from_contract(p8bs)
    shape = base.shape
    field = render_finite_support_thomas_region(
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
    field = field - np.mean(field, dtype=np.float64)
    density = base.astype(np.float64) + sigma.astype(np.float64) * field
    if np.any(density < 0.0):
        raise NativeThomasDensityConformanceError("reference density left domain")
    return np.ascontiguousarray(np.power(10.0, -density), dtype=np.float32)


def _build_msvc(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_thomas_density_msvc_v1.dll").resolve()
    import_library = (output_dir / "nf_thomas_density_msvc_v1.lib").resolve()
    batch = output_dir / "build_nf_thomas_density_msvc_v1.bat"
    sources = " ".join(f'"{(root / path).resolve()}"' for path in SOURCE_PATHS)
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD {sources} '
        f'/link /Brepro /OUT:"{dll}" /IMPLIB:"{import_library}"\r\n',
        encoding="ascii",
        newline="",
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(batch.resolve())],
        cwd=output_dir,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasDensityConformanceError(
            "MSVC P8BT build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "msvc-x64-c11",
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_thomas_density_llvm_v1.dll"
    command = [
        str(clang),
        "--target=x86_64-w64-windows-gnu",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-model=strict",
        "-shared",
        *(str(root / path) for path in SOURCE_PATHS),
        "-o",
        str(dll),
        "-Wl,--no-insert-timestamp",
    ]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasDensityConformanceError(
            "LLVM P8BT build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-20260616-clang-22.1.8-x64",
        "clang_sha256": sha256_file(clang),
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def _failure_atomic(library: ctypes.CDLL, root: Path) -> bool:
    profile = p8bs_profile_from_contract(_p8bs_contract(root)).as_abi()
    base = np.zeros((4, 4), dtype=np.float32)
    sigma = np.full((4, 4), 100.0, dtype=np.float32)
    workspace = np.full(64, np.float32(-17.0))
    output = np.full(16, np.float32(-13.0))
    raw_mean = ctypes.c_double(-11.0)
    status = library.nf_thomas_density_f32_apply_v1(
        ctypes.byref(profile), 4, 4,
        base.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), base.size,
        sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), sigma.size,
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), workspace.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
        ctypes.byref(raw_mean),
    )
    return status != 0 and np.all(output == -13.0) and raw_mean.value == -11.0


def evaluate_conformance(
    root: Path,
    contract: Mapping[str, Any],
    *,
    output_dir: Path,
    clang: Path,
) -> dict[str, Any]:
    validate_contract(root, contract)
    builds = {
        "msvc": _build_msvc(root, output_dir / "msvc"),
        "llvm_mingw": _build_llvm(root, output_dir / "llvm", clang),
    }
    profile = p8bs_profile_from_contract(_p8bs_contract(root))
    shape = tuple(int(value) for value in contract["fixture"]["shape"])
    base, sigma = fixture_arrays(shape)
    reference = float64_reference(root, base, sigma)
    rows: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    offset = float(contract["fixture"]["density_offset_probe"])
    expected_ratio = 10.0**-offset
    for name, build in builds.items():
        library = load_native_thomas_density_library(Path(build["dll_path"]))
        first, raw_mean, workspace_bytes = render_native_thomas_transmittance(
            library, profile, base, sigma
        )
        repeat, repeat_mean, _ = render_native_thomas_transmittance(
            library, profile, base, sigma
        )
        shifted, _, _ = render_native_thomas_transmittance(
            library, profile, base + np.float32(offset), sigma
        )
        delta = first.astype(np.float64) - reference.astype(np.float64)
        outputs[name] = first
        rows[name] = {
            "output_sha256": _sha256_bytes(first.tobytes()),
            "repeat_exact": bool(np.array_equal(first, repeat)),
            "raw_mean_repeat_exact": raw_mean == repeat_mean,
            "maximum_absolute_error": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "maximum_density_offset_ratio_error": float(
                np.max(np.abs(shifted.astype(np.float64) / first - expected_ratio))
            ),
            "minimum_transmittance": float(np.min(first)),
            "maximum_transmittance": float(np.max(first)),
            "failure_atomic": _failure_atomic(library, root),
            "workspace_bytes": workspace_bytes,
        }
    cross = float(np.max(np.abs(outputs["msvc"] - outputs["llvm_mingw"])))
    gates = contract["conformance"]
    gate_results = {
        "reference_maximum": max(row["maximum_absolute_error"] for row in rows.values())
        <= float(gates["maximum_absolute_error_vs_float64_reference"]),
        "reference_rmse": max(row["rmse"] for row in rows.values())
        <= float(gates["maximum_rmse_vs_float64_reference"]),
        "cross_compiler": cross
        <= float(gates["maximum_cross_compiler_absolute_error"]),
        "density_offset": max(
            row["maximum_density_offset_ratio_error"] for row in rows.values()
        )
        <= float(gates["maximum_density_offset_ratio_error"]),
        "transmittance_domain": all(
            row["minimum_transmittance"] > 0.0
            and row["maximum_transmittance"] <= 1.0
            for row in rows.values()
        ),
        "repeat": all(row["repeat_exact"] for row in rows.values()),
        "failure_atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    report = {
        "schema": "neuro_film.u6_p8bt_native_density_thomas_conformance.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(
            root / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
        ),
        "shape": list(shape),
        "reference_sha256": _sha256_bytes(reference.tobytes()),
        "toolchains": builds,
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = json.loads(json.dumps(report))
    for toolchain in identity["toolchains"].values():
        toolchain.pop("dll_path", None)
        toolchain.pop("dll_sha256", None)
    stable_id = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return {**report, "stable_evidence_id": stable_id}


__all__ = [
    "NativeThomasDensityConformanceError",
    "evaluate_conformance",
    "fixture_arrays",
    "float64_reference",
    "validate_contract",
]
