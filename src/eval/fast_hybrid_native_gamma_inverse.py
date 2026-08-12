"""P4HS accuracy and throughput evaluation for fast native Gamma inversion."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma

from src.eval.native_gamma_density_inverse import _inputs
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.native_fast_gamma_density import (
    NativeFastGammaDensityDiagnosticsV1,
    apply_native_fast_gamma_density,
    load_native_fast_gamma_density_library,
)

SCHEMA = "neuro-film.u6-p4hs-fast-hybrid-native-gamma-inverse-contract.v1"


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, str]:
    source = root / "native/film_physics/nf_gamma_density_fast_f64_v1.c"
    header = root / "native/film_physics/nf_gamma_density_fast_f64_v1.h"
    dll = output_dir / "nf_gamma_density_fast_f64_llvm_v1.dll"
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
            (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-20260616-clang-22.1.8-x64",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(source),
        "header_sha256": sha256_file(header),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def _call(
    library: ctypes.CDLL,
    u: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float]:
    started = time.perf_counter()
    output, diagnostics = apply_native_fast_gamma_density(
        library,
        u,
        a,
        s,
        direct_iterations=int(candidate["direct_inverse_iterations"]),
        newton_iterations=int(candidate["newton_iterations"]),
        direct_shape_upper=float(candidate["direct_shape_upper_exclusive"]),
        newton_shape_upper=float(candidate["newton_shape_upper_exclusive"]),
    )
    return output, diagnostics, time.perf_counter() - started


def _atomic(
    library: ctypes.CDLL,
    u: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    candidate: dict[str, Any],
) -> bool:
    bad = u.copy()
    bad[len(bad) // 2] = 1.0
    output = np.full(u.shape, -29.0)
    diagnostics = NativeFastGammaDensityDiagnosticsV1()
    diagnostics.struct_size = ctypes.sizeof(diagnostics)
    diagnostics.abi_version = 1
    before = output.tobytes()
    diag_before = ctypes.string_at(
        ctypes.byref(diagnostics), ctypes.sizeof(diagnostics)
    )
    status = library.nf_gamma_density_fast_f64_apply_v1(
        bad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bad.size,
        int(candidate["direct_inverse_iterations"]),
        int(candidate["newton_iterations"]),
        float(candidate["direct_shape_upper_exclusive"]),
        float(candidate["newton_shape_upper_exclusive"]),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.size,
        ctypes.byref(diagnostics),
    )
    return (
        status != 0
        and output.tobytes() == before
        and ctypes.string_at(ctypes.byref(diagnostics), ctypes.sizeof(diagnostics))
        == diag_before
    )


def _stable(report: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(report))
    payload.pop("stable_evidence_id", None)
    for build in payload["toolchains"].values():
        for key in ("dll_path", "dll_sha256", "compiler_output"):
            build.pop(key, None)
    for row in payload["results"].values():
        row.pop("kernel_seconds", None)
    return payload


def evaluate(
    *, root: Path, contract: dict[str, Any], output_dir: Path, clang: Path
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HS contract")
    for name in ("p4hr_evidence", "p4hp_evidence"):
        binding = contract["parents"][name]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HS parent drift: {name}")
        if (
            json.loads(path.read_text("utf-8")).get("decision")
            != binding["required_decision"]
        ):
            raise ValueError(f"P4HS decision drift: {name}")
    candidate = contract["candidate"]
    u, a, s = _inputs(candidate)
    reference = gamma.ppf(u, a=a, scale=s)
    msvc = build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_gamma_density_fast_f64_v1.c",
        header_relative="native/film_physics/nf_gamma_density_fast_f64_v1.h",
        basename="nf_gamma_density_fast_f64_msvc_v1",
    )
    llvm = _build_llvm(root, output_dir, clang)
    rows = {}
    outputs = {}
    group = int(candidate["uniform_grid_count"])
    for name, build in {"msvc": msvc, "llvm_mingw": llvm}.items():
        library = load_native_fast_gamma_density_library(Path(build["dll_path"]))
        first, diagnostics, _ = _call(library, u, a, s, candidate)
        second, second_diagnostics, _ = _call(library, u, a, s, candidate)
        delta = first - reference
        outputs[name] = first
        rng = np.random.default_rng(2026081341)
        perf_count = int(candidate["performance_fixture_samples"])
        perf_shape = np.power(
            10.0,
            rng.uniform(
                float(candidate["performance_shape_log10_min"]),
                float(candidate["performance_shape_log10_max"]),
                perf_count,
            ),
        )
        perf_uniform = rng.uniform(1.0 / 131072.0, 1.0 - 1.0 / 131072.0, perf_count)
        perf_scale = rng.uniform(1e-8, 0.01, perf_count)
        _, _, seconds = _call(library, perf_uniform, perf_shape, perf_scale, candidate)
        rows[name] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "repeat_exact": bool(np.array_equal(first, second)),
            "diagnostics_repeat_exact": diagnostics == second_diagnostics,
            "kernel_seconds": seconds,
            "maximum_absolute_developed_density_error": float(np.max(np.abs(delta))),
            "maximum_relative_developed_density_error": float(
                np.max(np.abs(delta) / np.maximum(np.abs(reference), 1e-15))
            ),
            "monotone": bool(
                all(
                    np.all(np.diff(first[i : i + group]) > 0)
                    for i in range(0, first.size, group)
                )
            ),
            "failure_atomic": _atomic(library, u, a, s, candidate),
            "diagnostics": diagnostics,
        }
    cross = float(np.max(np.abs(outputs["msvc"] - outputs["llvm_mingw"])))
    gates = contract["automatic_gates"]
    checks = {
        "absolute_error": max(
            r["maximum_absolute_developed_density_error"] for r in rows.values()
        )
        <= gates["maximum_absolute_developed_density_error"],
        "relative_error": max(
            r["maximum_relative_developed_density_error"] for r in rows.values()
        )
        <= gates["maximum_relative_developed_density_error"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "kernel_time": max(r["kernel_seconds"] for r in rows.values())
        <= gates["maximum_kernel_seconds_per_131793_samples"],
        "three_branches": all(
            r["diagnostics"][key] > 0
            for r in rows.values()
            for key in (
                "direct_branch_count",
                "newton_branch_count",
                "asymptotic_branch_count",
            )
        ),
        "repeat": all(
            r["repeat_exact"] and r["diagnostics_repeat_exact"] for r in rows.values()
        ),
        "monotone": all(r["monotone"] for r in rows.values()),
        "failure_atomic": all(r["failure_atomic"] for r in rows.values()),
    }
    passed = all(checks.values())
    report = {
        "schema": "neuro-film.u6-p4hs-fast-hybrid-native-gamma-inverse-result.v1",
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sample_count": int(u.size),
        "toolchains": {"msvc": msvc, "llvm_mingw": llvm},
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(_stable(report), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


__all__ = ["evaluate"]
