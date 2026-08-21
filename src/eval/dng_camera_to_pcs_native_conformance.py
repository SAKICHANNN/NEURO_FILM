"""Dual-compiler conformance for the P96 DNG camera-to-PCS C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file

SCHEMA = "neuro_film.p96_dng_camera_to_pcs_portable_abi_result.v1"
SOURCE_RELATIVE = "native/preprocess/nf_dng_camera_to_pcs_v1.c"
HEADER_RELATIVE = "native/preprocess/nf_dng_camera_to_pcs_v1.h"
BASENAME = "nf_dng_camera_to_pcs_v1"


class P96Error(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise P96Error(f"bound JSON mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P96Error(f"bound JSON root must be an object: {path}")
    return value


def build_msvc(root: Path, output_dir: Path) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE_RELATIVE,
        header_relative=HEADER_RELATIVE,
        basename=BASENAME,
    )


def build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    source = root / SOURCE_RELATIVE
    header = root / HEADER_RELATIVE
    dll = output_dir / f"{BASENAME}.dll"
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
        raise P96Error(
            "LLVM-MinGW P96 build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-x64-c11-strict",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(source),
        "header_sha256": sha256_file(header),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def build_probes() -> np.ndarray:
    probes = np.empty((257, 3), dtype=np.float64)
    for index in range(257):
        probes[index, 0] = ((index * 37 + 3) % 257) / 64.0
        probes[index, 1] = ((index * 73 + 11) % 257) / 96.0
        probes[index, 2] = ((index * 109 + 17) % 257) / 128.0
    probes[:8] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.18, 0.18, 0.18],
            [4.0, 4.0, 4.0],
            [0.125, 2.0, 3.75],
        ],
        dtype=np.float64,
    )
    return probes


def python_oracle(matrix: np.ndarray, probes: np.ndarray) -> np.ndarray:
    output = np.empty_like(probes)
    for index, (x, y, z) in enumerate(probes):
        output[index, 0] = (matrix[0, 0] * x + matrix[0, 1] * y) + matrix[0, 2] * z
        output[index, 1] = (matrix[1, 0] * x + matrix[1, 1] * y) + matrix[1, 2] * z
        output[index, 2] = (matrix[2, 0] * x + matrix[2, 1] * y) + matrix[2, 2] * z
    return output


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_dng_camera_to_pcs_abi_version_v1.argtypes = []
    library.nf_dng_camera_to_pcs_abi_version_v1.restype = ctypes.c_uint32
    library.nf_dng_camera_to_pcs_apply_v1.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    library.nf_dng_camera_to_pcs_apply_v1.restype = ctypes.c_int
    if library.nf_dng_camera_to_pcs_abi_version_v1() != 1:
        raise P96Error("native P96 ABI version drift")
    return library


def _apply(library: ctypes.CDLL, matrix: np.ndarray, probes: np.ndarray) -> np.ndarray:
    output = np.full_like(probes, -17.0)
    status = library.nf_dng_camera_to_pcs_apply_v1(
        matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        probes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(probes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output.size,
    )
    if status != 0:
        raise P96Error(f"native P96 apply failed with status {status}")
    return output


def _failure_atomicity(
    library: ctypes.CDLL, matrix: np.ndarray, probes: np.ndarray
) -> dict[str, bool]:
    matrix_pointer = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def invoke(
        test_matrix: np.ndarray, test_input: np.ndarray, output_count: int
    ) -> bool:
        output = np.full_like(probes, -1234.5)
        before = output.tobytes()
        status = library.nf_dng_camera_to_pcs_apply_v1(
            test_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            test_input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(probes),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output_count,
        )
        return status != 0 and output.tobytes() == before

    nonfinite_matrix = matrix.copy()
    nonfinite_matrix[0, 0] = np.nan
    singular_matrix = np.zeros((3, 3), dtype=np.float64)
    nonfinite_input = probes.copy()
    nonfinite_input[-1, -1] = np.inf
    storage = np.arange(probes.size + 1, dtype=np.float64)
    storage_before = storage.tobytes()
    partial_status = library.nf_dng_camera_to_pcs_apply_v1(
        matrix_pointer,
        storage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(probes),
        ctypes.cast(storage.ctypes.data + 8, ctypes.POINTER(ctypes.c_double)),
        probes.size,
    )
    return {
        "nonfinite_matrix": invoke(nonfinite_matrix, probes, probes.size),
        "singular_matrix": invoke(singular_matrix, probes, probes.size),
        "nonfinite_input": invoke(matrix, nonfinite_input, probes.size),
        "undersized_output": invoke(matrix, probes, probes.size - 1),
        "partial_overlap": partial_status != 0 and storage.tobytes() == storage_before,
    }


def run_loaded(
    dll_path: Path, profiles: list[dict[str, Any]], probes: np.ndarray
) -> dict[str, Any]:
    library = _load_library(dll_path)
    rows = []
    output_arrays = {}
    maximum_error = 0.0
    for profile in profiles:
        matrix = np.ascontiguousarray(profile["camera_to_pcs"], dtype=np.float64)
        expected = python_oracle(matrix, probes)
        output = _apply(library, matrix, probes)
        repeat = _apply(library, matrix, probes)
        inplace = probes.copy()
        status = library.nf_dng_camera_to_pcs_apply_v1(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            inplace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inplace),
            inplace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            inplace.size,
        )
        error = float(np.max(np.abs(output - expected)))
        maximum_error = max(maximum_error, error)
        if status != 0:
            raise P96Error("native P96 in-place apply failed")
        output_arrays[profile["source_id"]] = output
        rows.append(
            {
                "camera_make": profile["camera_make"],
                "inplace_byte_exact": inplace.tobytes() == output.tobytes(),
                "maximum_absolute_error": error,
                "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "repeat_byte_exact": repeat.tobytes() == output.tobytes(),
                "source_id": profile["source_id"],
            }
        )
    first_matrix = np.ascontiguousarray(profiles[0]["camera_to_pcs"], dtype=np.float64)
    return {
        "failure_atomicity": _failure_atomicity(library, first_matrix, probes),
        "maximum_python_error": maximum_error,
        "output_arrays": output_arrays,
        "rows": rows,
    }


def evaluate(
    *, root: Path, config: dict[str, Any], output_dir: Path, order: str
) -> dict[str, Any]:
    for binding in config["bindings"].values():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise P96Error(f"P96 binding mismatch: {binding['path']}")
    parent = _load_bound_json(
        root / config["bindings"]["p94_report"]["path"],
        config["bindings"]["p94_report"]["sha256"],
    )
    if parent["decision"] != "PASS_PRIVATE_DNG_FORWARD_MATRIX_MECHANICS":
        raise P96Error("P94 parent decision drift")
    profiles = list(parent["rows"])
    profiles.sort(key=lambda item: item["source_id"], reverse=order == "reverse")
    probes = build_probes()
    clang = root / config["bindings"]["clang"]["path"]
    builds = {
        "msvc_a": build_msvc(root, output_dir / "msvc_a"),
        "msvc_b": build_msvc(root, output_dir / "msvc_b"),
        "llvm_a": build_llvm(root, output_dir / "llvm_a", clang),
        "llvm_b": build_llvm(root, output_dir / "llvm_b", clang),
    }
    independent_builds_exact = (
        builds["msvc_a"]["dll_sha256"] == builds["msvc_b"]["dll_sha256"]
        and builds["llvm_a"]["dll_sha256"] == builds["llvm_b"]["dll_sha256"]
    )
    loaded = {
        name: run_loaded(Path(build["dll_path"]), profiles, probes)
        for name, build in builds.items()
    }
    canonical_rows = {
        name: {row["source_id"]: row for row in result["rows"]}
        for name, result in loaded.items()
    }
    all_source_ids = sorted(canonical_rows["msvc_a"])
    maximum_cross_compiler_error = 0.0
    cross_compiler_hash_exact = True
    for source_id in all_source_ids:
        msvc = canonical_rows["msvc_a"][source_id]
        llvm = canonical_rows["llvm_a"][source_id]
        cross_compiler_hash_exact &= msvc["output_sha256"] == llvm["output_sha256"]
        maximum_cross_compiler_error = max(
            maximum_cross_compiler_error,
            float(
                np.max(
                    np.abs(
                        loaded["msvc_a"]["output_arrays"][source_id]
                        - loaded["llvm_a"]["output_arrays"][source_id]
                    )
                )
            ),
        )
    gates = config["gates"]
    metrics = {
        "cross_compiler_output_hash_exact": cross_compiler_hash_exact,
        "independent_builds_exact": independent_builds_exact,
        "maximum_cross_compiler_error": maximum_cross_compiler_error,
        "maximum_python_error": max(
            result["maximum_python_error"] for result in loaded.values()
        ),
        "profile_count": len(profiles),
        "probe_count": len(probes),
        "raster_sample_rgb_reads": 0,
        "transformed_triplet_count": len(profiles) * len(probes),
    }
    gate_results = {
        "failure_atomicity": all(
            all(result["failure_atomicity"].values()) for result in loaded.values()
        ),
        "cross_compiler_error": metrics["maximum_cross_compiler_error"]
        <= gates["maximum_absolute_error"],
        "independent_builds": metrics["independent_builds_exact"],
        "inplace": all(
            row["inplace_byte_exact"]
            for result in loaded.values()
            for row in result["rows"]
        ),
        "profile_count": metrics["profile_count"] == gates["required_profiles"],
        "python_error": metrics["maximum_python_error"]
        <= gates["maximum_absolute_error"],
        "repeat": all(
            row["repeat_byte_exact"]
            for result in loaded.values()
            for row in result["rows"]
        ),
        "triplet_count": metrics["transformed_triplet_count"]
        == gates["required_transformed_triplets"],
        "zero_raster_sample_reads": metrics["raster_sample_rgb_reads"]
        <= gates["maximum_raster_sample_rgb_reads"],
    }
    build_facts = {
        name: {
            "dll_sha256": build["dll_sha256"],
            "header_sha256": build["header_sha256"],
            "source_sha256": build["source_sha256"],
            "toolchain": build["toolchain"],
        }
        for name, build in builds.items()
    }
    scientific = {
        "builds": build_facts,
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_CAMERA_TO_PCS_PORTABLE_ABI"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_CAMERA_TO_PCS_PORTABLE_ABI",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": metrics,
        "probe_sha256": hashlib.sha256(probes.tobytes()).hexdigest(),
        "rows": sorted(loaded["msvc_a"]["rows"], key=lambda item: item["source_id"]),
        "schema": SCHEMA,
    }
    return {
        **scientific,
        "stable_identity_sha256": hashlib.sha256(
            _canonical_bytes(scientific)
        ).hexdigest(),
    }


__all__ = [
    "P96Error",
    "build_probes",
    "evaluate",
    "python_oracle",
]
