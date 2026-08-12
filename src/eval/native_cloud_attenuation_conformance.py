"""Build and execute the frozen P4DZ native attenuation kernel."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file

SOURCE = "native/film_physics/nf_cloud_attenuation_f32_v1.c"
HEADER = "native/film_physics/nf_cloud_attenuation_f32_v1.h"


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_cloud_attenuation_llvm_v1.dll"
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
            str(root / SOURCE),
            "-o",
            str(dll),
            "-Wl,--no-insert-timestamp",
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
        "toolchain": "llvm-mingw-clang-22.1.8",
        "dll_path": str(dll.resolve()),
        "dll_sha256": sha256_file(dll),
    }


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.nf_cloud_attenuation_f32_apply_v1.argtypes = [
        pointer,
        pointer,
        ctypes.c_size_t,
        pointer,
        pointer,
        pointer,
    ]
    library.nf_cloud_attenuation_f32_apply_v1.restype = ctypes.c_int
    return library


def _apply(
    library: ctypes.CDLL, expected: np.ndarray, base: np.ndarray, gain: np.ndarray
):
    density = np.full_like(expected, np.float32(-7.0))
    output = np.full_like(expected, np.float32(-9.0))
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_cloud_attenuation_f32_apply_v1(
        expected.ctypes.data_as(pointer),
        base.ctypes.data_as(pointer),
        expected.size // 3,
        gain.ctypes.data_as(pointer),
        density.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
    )
    return status, density, output


def evaluate(
    root: Path, contract_path: Path, output_dir: Path, clang: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4DZ parent drift")
    if (
        json.loads(parent.read_text())["decision"]
        != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4DZ parent decision drift")
    builds = {
        "msvc": build_msvc_c11_dll(
            root=root,
            output_dir=output_dir / "msvc",
            source_relative=SOURCE,
            header_relative=HEADER,
            basename="nf_cloud_attenuation_msvc_v1",
        ),
        "llvm": _build_llvm(root, output_dir / "llvm", clang),
    }
    rng = np.random.default_rng(contract["fixture"]["seed"])
    count = contract["fixture"]["sample_count"]
    expected = np.ascontiguousarray(
        rng.uniform(0.15, 0.95, (count, 3)), dtype=np.float32
    )
    base = np.ascontiguousarray(
        expected + rng.uniform(-0.1, 0.02, (count, 3)), dtype=np.float32
    )
    gain = np.asarray(contract["fixture"]["channel_residual_gain"], dtype=np.float32)
    reference_t = expected.astype(np.float64) + gain.astype(np.float64) * (
        base.astype(np.float64) - expected.astype(np.float64)
    )
    reference_d = -np.log10(reference_t)
    outputs = {}
    rows = {}
    for name, build in builds.items():
        library = _load(Path(build["dll_path"]))
        status, density, transmittance = _apply(library, expected, base, gain)
        repeat_status, repeat_density, repeat_t = _apply(library, expected, base, gain)
        invalid = base.copy()
        invalid[-1, -1] = np.nan
        bad_status, bad_density, bad_t = _apply(library, expected, invalid, gain)
        outputs[name] = (density, transmittance)
        rows[name] = {
            "status": status,
            "repeat_status": repeat_status,
            "transmittance_sha256": hashlib.sha256(transmittance.tobytes()).hexdigest(),
            "density_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
            "maximum_transmittance_absolute_error": float(
                np.max(np.abs(transmittance.astype(np.float64) - reference_t))
            ),
            "maximum_density_absolute_error": float(
                np.max(np.abs(density.astype(np.float64) - reference_d))
            ),
            "repeat_exact": bool(
                np.array_equal(density, repeat_density)
                and np.array_equal(transmittance, repeat_t)
            ),
            "failure_atomic": bool(
                bad_status != 0
                and np.all(bad_density == -7.0)
                and np.all(bad_t == -9.0)
            ),
        }
    cross = max(
        float(np.max(np.abs(outputs["msvc"][0] - outputs["llvm"][0]))),
        float(np.max(np.abs(outputs["msvc"][1] - outputs["llvm"][1]))),
    )
    gates = contract["gates"]
    decisions = {
        "transmittance": max(
            row["maximum_transmittance_absolute_error"] for row in rows.values()
        )
        <= gates["maximum_transmittance_absolute_error_vs_python"],
        "density": max(row["maximum_density_absolute_error"] for row in rows.values())
        <= gates["maximum_density_absolute_error_vs_python"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "repeat": all(row["repeat_exact"] for row in rows.values()),
        "atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dz_native_cloud_attenuation_conformance.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
