"""P4HM dual-compiler conformance for native histogram-copula transport."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.native_histogram_copula import (
    NativeHistogramCopulaDiagnosticsV1,
    apply_native_histogram_copula,
    load_native_histogram_copula_library,
)

SCHEMA = "neuro-film.u6-p4hm-native-histogram-copula-transport-contract.v1"


def _acklam(p: np.ndarray) -> np.ndarray:
    a = np.asarray(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.asarray(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.asarray(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.asarray(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    values = np.asarray(p, dtype=np.float64)
    result = np.empty_like(values)
    lower = values < 0.02425
    upper = values > 0.97575
    center = ~(lower | upper)
    q = np.sqrt(-2.0 * np.log(values[lower]))
    result[lower] = (
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )
    q = np.sqrt(-2.0 * np.log(1.0 - values[upper]))
    result[upper] = -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )
    q = values[center] - 0.5
    r = q * q
    result[center] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )
    return result


def _quantize(values: np.ndarray, bins: int) -> np.ndarray:
    minimum = np.min(values)
    maximum = np.max(values)
    if maximum <= minimum:
        raise ValueError("degenerate histogram input")
    return np.floor(
        (values - minimum) * float(bins - 1) / (maximum - minimum) + 0.5
    ).astype(np.uint16)


def _midpoint_table(counts: np.ndarray, sample_count: int) -> np.ndarray:
    cumulative = np.cumsum(counts, dtype=np.uint64)
    before = cumulative - counts
    return (before.astype(np.float64) + 0.5 * counts) / float(sample_count)


def python_reference(
    fields: np.ndarray, target_correlation: np.ndarray, bins: int
) -> np.ndarray:
    sample_count = fields.shape[0]
    raw_bins = np.empty(fields.shape, dtype=np.uint16)
    normal = np.empty(fields.shape, dtype=np.float64)
    for channel in range(3):
        raw_bins[:, channel] = _quantize(fields[:, channel].astype(np.float64), bins)
        counts = np.bincount(raw_bins[:, channel], minlength=bins).astype(np.uint64)
        midpoint = _midpoint_table(counts, sample_count)
        occupied = counts > 0
        table = np.zeros(bins, dtype=np.float64)
        table[occupied] = _acklam(midpoint[occupied])
        normal[:, channel] = table[raw_bins[:, channel]]
    mean = np.mean(normal, axis=0)
    correlation = np.corrcoef(normal, rowvar=False)
    input_lower = np.linalg.cholesky(correlation)
    target_lower = np.linalg.cholesky(target_correlation)
    whitened = np.linalg.solve(input_lower, (normal - mean).T).T
    correlated = whitened @ target_lower.T
    output = np.empty(fields.shape, dtype=np.float32)
    for channel in range(3):
        quantized = _quantize(correlated[:, channel], bins)
        counts = np.bincount(quantized, minlength=bins).astype(np.uint64)
        output[:, channel] = _midpoint_table(counts, sample_count)[quantized]
    return output


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, str]:
    source = root / "native/film_physics/nf_histogram_copula_f32_v1.c"
    header = root / "native/film_physics/nf_histogram_copula_f32_v1.h"
    dll = output_dir / "nf_histogram_copula_f32_llvm_v1.dll"
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
            "P4HM LLVM-MinGW build failed:\n"
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


def _failure_atomic(
    library: ctypes.CDLL, fields: np.ndarray, target: np.ndarray, bins: int
) -> bool:
    workspace_bytes = ctypes.c_size_t()
    if (
        library.nf_histogram_copula_f32_workspace_bytes_v1(
            fields.shape[0], bins, ctypes.byref(workspace_bytes)
        )
        != 0
    ):
        return False
    workspace = ctypes.create_string_buffer(workspace_bytes.value)
    output = np.full(fields.shape, np.float32(-13.0))
    diagnostics = NativeHistogramCopulaDiagnosticsV1()
    diagnostics.struct_size = ctypes.sizeof(diagnostics)
    diagnostics.abi_version = 1
    output_before = output.tobytes()
    diagnostics_before = ctypes.string_at(
        ctypes.byref(diagnostics), ctypes.sizeof(diagnostics)
    )
    status = library.nf_histogram_copula_f32_apply_v1(
        fields.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fields.shape[0],
        target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bins,
        workspace,
        workspace_bytes.value - 1,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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


def evaluate(
    *, root: Path, contract: dict[str, Any], output_dir: Path, clang: Path
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HM contract")
    for name, binding in contract["parents"].items():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HM parent drift: {name}")
    evidence = json.loads(
        (root / contract["parents"]["p4hl_evidence"]["path"]).read_text("utf-8")
    )
    if evidence.get("decision") != contract["parents"]["p4hl_evidence"][
        "required_decision"
    ]:
        raise ValueError("P4HM P4HL decision drift")
    profile = json.loads(
        (root / contract["parents"]["profile"]["path"]).read_text("utf-8")
    )
    target = np.ascontiguousarray(
        profile["execution"]["correlation_matrix"], dtype=np.float64
    )
    fixture = contract["fixture"]
    count = int(fixture["height"]) * int(fixture["width"])
    rng = np.random.default_rng(int(fixture["field_seed"]))
    latent = rng.standard_normal((count, 4))
    fields = np.empty((count, 3), dtype=np.float32)
    fields[:, 0] = latent[:, 0] + 0.19 * latent[:, 3] ** 2
    fields[:, 1] = 0.23 * latent[:, 0] + latent[:, 1] - 0.11 * latent[:, 3] ** 2
    fields[:, 2] = -0.17 * latent[:, 0] + 0.29 * latent[:, 1] + latent[:, 2]
    bins = int(contract["candidate"]["rank_bins"])
    reference = python_reference(fields, target, bins)
    msvc = build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_histogram_copula_f32_v1.c",
        header_relative="native/film_physics/nf_histogram_copula_f32_v1.h",
        basename="nf_histogram_copula_f32_msvc_v1",
    )
    llvm = _build_llvm(root, output_dir, clang)
    rows: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, build in {"msvc": msvc, "llvm_mingw": llvm}.items():
        library = load_native_histogram_copula_library(Path(build["dll_path"]))
        first, diagnostics = apply_native_histogram_copula(
            library, fields, target_correlation=target, rank_bins=bins
        )
        second, second_diagnostics = apply_native_histogram_copula(
            library, fields, target_correlation=target, rank_bins=bins
        )
        delta = first.astype(np.float64) - reference.astype(np.float64)
        outputs[name] = first
        expected_uniform_correlation = (6.0 / math.pi) * np.arcsin(target / 2.0)
        np.fill_diagonal(expected_uniform_correlation, 1.0)
        observed_correlation = np.asarray(
            diagnostics["output_uniform_correlation"], dtype=np.float64
        )
        rows[name] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "repeat_exact": bool(np.array_equal(first, second)),
            "diagnostics_repeat_exact": diagnostics == second_diagnostics,
            "maximum_absolute_error_vs_python": float(np.max(np.abs(delta))),
            "rmse_vs_python": float(np.sqrt(np.mean(delta * delta))),
            "maximum_expected_uniform_correlation_error": float(
                np.max(np.abs(observed_correlation - expected_uniform_correlation))
            ),
            "maximum_uniform_mean_error": float(
                np.max(np.abs(np.asarray(diagnostics["output_uniform_mean"]) - 0.5))
            ),
            "finite_unit_output": bool(
                np.all(np.isfinite(first))
                and np.all(first >= 0.0)
                and np.all(first <= 1.0)
            ),
            "failure_atomic": _failure_atomic(library, fields, target, bins),
            "workspace_bytes": diagnostics["workspace_bytes"],
            "diagnostics": diagnostics,
        }
    cross = float(
        np.max(
            np.abs(
                outputs["msvc"].astype(np.float64)
                - outputs["llvm_mingw"].astype(np.float64)
            )
        )
    )
    gates = contract["automatic_gates"]
    checks = {
        "python_maximum": max(
            row["maximum_absolute_error_vs_python"] for row in rows.values()
        )
        <= gates["maximum_absolute_error_vs_python_same_algorithm"],
        "python_rmse": max(row["rmse_vs_python"] for row in rows.values())
        <= gates["maximum_rmse_vs_python_same_algorithm"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "target_correlation": max(
            row["maximum_expected_uniform_correlation_error"] for row in rows.values()
        )
        <= gates["maximum_target_correlation_error"],
        "uniform_mean": max(
            row["maximum_uniform_mean_error"] for row in rows.values()
        )
        <= gates["maximum_uniform_mean_error"],
        "workspace": max(row["workspace_bytes"] for row in rows.values())
        <= gates["maximum_workspace_bytes"],
        "repeat": all(
            row["repeat_exact"] and row["diagnostics_repeat_exact"]
            for row in rows.values()
        ),
        "finite_unit": all(row["finite_unit_output"] for row in rows.values()),
        "failure_atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    report = {
        "schema": contract["schema"].replace("contract", "result"),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "profile_bundle_sha256": profile["bundle_sha256"],
        "fixture_sha256": hashlib.sha256(fields.tobytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "toolchains": {"msvc": msvc, "llvm_mingw": llvm},
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            _stable_payload(report), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    return report


__all__ = ["evaluate", "python_reference"]
