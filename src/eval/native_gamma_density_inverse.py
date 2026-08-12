"""P4HO dual-compiler evaluation for native inverse-Gamma density."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.native_gamma_density import (
    NativeGammaDensityDiagnosticsV1,
    apply_native_gamma_density,
    load_native_gamma_density_library,
)

SCHEMA = "neuro-film.u6-p4ho-native-gamma-density-inverse-contract.v1"


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, str]:
    source = root / "native/film_physics/nf_gamma_density_f64_v1.c"
    header = root / "native/film_physics/nf_gamma_density_f64_v1.h"
    dll = output_dir / "nf_gamma_density_f64_llvm_v1.dll"
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
        raise RuntimeError(
            "P4HO LLVM-MinGW build failed:\n"
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


def _inputs(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(candidate["uniform_grid_count"])
    tail = 1.0 / float(candidate["uniform_tail_denominator"])
    uniforms = np.linspace(tail, 1.0 - tail, count, dtype=np.float64)
    rows = [
        (probability, shape, mean / shape)
        for shape in candidate["shape_values"]
        for mean in candidate["mean_density_values"]
        for probability in uniforms
    ]
    return tuple(np.asarray(column, dtype=np.float64) for column in zip(*rows))  # type: ignore[return-value]


def _failure_atomic(
    library: ctypes.CDLL,
    uniforms: np.ndarray,
    shapes: np.ndarray,
    scales: np.ndarray,
    iterations: int,
) -> bool:
    bad = uniforms.copy()
    bad[len(bad) // 2] = 1.0
    output = np.full(uniforms.shape, -17.0, dtype=np.float64)
    diagnostics = NativeGammaDensityDiagnosticsV1()
    diagnostics.struct_size = ctypes.sizeof(diagnostics)
    diagnostics.abi_version = 1
    output_before = output.tobytes()
    diagnostics_before = ctypes.string_at(ctypes.byref(diagnostics), ctypes.sizeof(diagnostics))
    status = library.nf_gamma_density_f64_apply_v1(
        bad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        shapes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bad.size,
        iterations,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.size,
        ctypes.byref(diagnostics),
    )
    return (
        status != 0
        and output.tobytes() == output_before
        and ctypes.string_at(ctypes.byref(diagnostics), ctypes.sizeof(diagnostics))
        == diagnostics_before
    )


def _stable_payload(report: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(report))
    stable.pop("stable_evidence_id", None)
    for toolchain in stable["toolchains"].values():
        toolchain.pop("dll_path", None)
        toolchain.pop("dll_sha256", None)
        toolchain.pop("compiler_output", None)
    return stable


def evaluate(*, root: Path, contract: dict[str, Any], output_dir: Path, clang: Path) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HO contract")
    for name in ("p4hn_evidence", "profile"):
        binding = contract["parents"][name]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HO parent drift: {name}")
    evidence = json.loads(
        (root / contract["parents"]["p4hn_evidence"]["path"]).read_text("utf-8")
    )
    if evidence.get("decision") != contract["parents"]["p4hn_evidence"][
        "required_decision"
    ]:
        raise ValueError("P4HO P4HN decision drift")
    candidate = contract["candidate"]
    uniforms, shapes, scales = _inputs(candidate)
    reference = gamma.ppf(uniforms, a=shapes, scale=scales)
    msvc = build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_gamma_density_f64_v1.c",
        header_relative="native/film_physics/nf_gamma_density_f64_v1.h",
        basename="nf_gamma_density_f64_msvc_v1",
    )
    llvm = _build_llvm(root, output_dir, clang)
    rows: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    iterations = int(candidate["inverse_iterations"])
    group_size = int(candidate["uniform_grid_count"])
    for name, build in {"msvc": msvc, "llvm_mingw": llvm}.items():
        library = load_native_gamma_density_library(Path(build["dll_path"]))
        first, diagnostics = apply_native_gamma_density(
            library, uniforms, shapes, scales, inverse_iterations=iterations
        )
        second, second_diagnostics = apply_native_gamma_density(
            library, uniforms, shapes, scales, inverse_iterations=iterations
        )
        delta = first - reference
        monotone = all(
            np.all(np.diff(first[start : start + group_size]) > 0.0)
            for start in range(0, first.size, group_size)
        )
        outputs[name] = first
        rows[name] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "repeat_exact": bool(np.array_equal(first, second)),
            "diagnostics_repeat_exact": diagnostics == second_diagnostics,
            "positive_finite": bool(np.all(np.isfinite(first)) and np.all(first > 0.0)),
            "monotone": bool(monotone),
            "failure_atomic": _failure_atomic(
                library, uniforms, shapes, scales, iterations
            ),
            "maximum_absolute_developed_density_error": float(np.max(np.abs(delta))),
            "maximum_relative_developed_density_error": float(
                np.max(np.abs(delta) / np.maximum(np.abs(reference), 1e-15))
            ),
            "diagnostics": diagnostics,
        }
    cross = float(np.max(np.abs(outputs["msvc"] - outputs["llvm_mingw"])))
    gates = contract["automatic_gates"]
    checks = {
        "absolute_error": max(
            row["maximum_absolute_developed_density_error"] for row in rows.values()
        )
        <= gates["maximum_absolute_developed_density_error"],
        "relative_error": max(
            row["maximum_relative_developed_density_error"] for row in rows.values()
        )
        <= gates["maximum_relative_developed_density_error"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "repeat": all(
            row["repeat_exact"] and row["diagnostics_repeat_exact"]
            for row in rows.values()
        ),
        "positive_finite": all(row["positive_finite"] for row in rows.values()),
        "monotone": all(row["monotone"] for row in rows.values()),
        "failure_atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.u6-p4ho-native-gamma-density-inverse-result.v1",
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "profile_bundle_sha256": contract["parents"]["profile"]["bundle_sha256"],
        "sample_count": int(uniforms.size),
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "toolchains": {"msvc": msvc, "llvm_mingw": llvm},
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(_stable_payload(report), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


__all__ = ["evaluate"]
