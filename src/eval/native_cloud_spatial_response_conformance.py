"""P4EH native separable cloud spatial-response conformance."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file

SOURCE = "native/film_physics/nf_cloud_spatial_response_f32_v1.c"
HEADER = "native/film_physics/nf_cloud_spatial_response_f32_v1.h"


class _Profile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("sigma", ctypes.c_double * 3),
        ("mark", ctypes.c_double * 3),
        ("truncate", ctypes.c_double),
    ]


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_cloud_spatial_response_llvm_v1.dll"
    completed = subprocess.run(
        [
            str(clang), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2",
            "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared",
            str(root / SOURCE), "-o", str(dll), "-Wl,--no-insert-timestamp",
        ], capture_output=True, check=False, timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise RuntimeError((completed.stdout + completed.stderr).decode(errors="replace"))
    return {"dll_path": str(dll.resolve()), "dll_sha256": sha256_file(dll)}


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_cloud_spatial_response_f32_apply_v1.argtypes = [
        ctypes.POINTER(_Profile), ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.nf_cloud_spatial_response_f32_apply_v1.restype = ctypes.c_int
    return library


def _apply(library: ctypes.CDLL, profile: _Profile, counts: np.ndarray, core_height: int, halo: int):
    density = np.full((core_height, counts.shape[1], 3), -7.0, dtype=np.float32)
    transmittance = np.full_like(density, -9.0)
    workspace = np.empty(2 * counts.shape[0] * counts.shape[1], dtype=np.float64)
    status = library.nf_cloud_spatial_response_f32_apply_v1(
        ctypes.byref(profile), counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        core_height, counts.shape[1], halo,
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), workspace.size,
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        transmittance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), density.size,
    )
    return status, density, transmittance


def evaluate(root: Path, contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4EH parent drift")
    if evidence["decision"] != contract["parent"]["required_decision"]:
        raise RuntimeError("P4EH parent decision drift")
    builds = {
        "msvc": build_msvc_c11_dll(root=root, output_dir=output_dir / "msvc", source_relative=SOURCE, header_relative=HEADER, basename="nf_cloud_spatial_response_msvc_v1"),
        "llvm": _build_llvm(root, output_dir / "llvm", clang),
    }
    fixture = contract["fixture"]
    core_height, width, halo = fixture["core_height"], fixture["width"], fixture["halo"]
    rng = np.random.default_rng(fixture["seed"])
    counts = np.ascontiguousarray(rng.poisson(8.0, (core_height + 2 * halo, width, 3)), dtype=np.uint16)
    sigma = tuple(fixture["sigma_pixels_cmy"])
    mark = tuple(fixture["mark_optical_density_cmy"])
    profile = _Profile(ctypes.sizeof(_Profile), 1, (ctypes.c_double * 3)(*sigma), (ctypes.c_double * 3)(*mark), fixture["truncate"])
    reference_channels = []
    for channel in range(3):
        filtered = gaussian_filter(counts[..., channel].astype(np.float64), sigma=sigma[channel], mode=("nearest", "wrap"), truncate=fixture["truncate"])
        reference_channels.append(filtered[halo : halo + core_height] * mark[channel])
    reference_density = np.stack(reference_channels, axis=-1)
    reference_t = np.exp(-reference_density)
    outputs = {}
    results = {}
    for name, build in builds.items():
        library = _load(Path(build["dll_path"]))
        try:
            status, density, transmittance = _apply(library, profile, counts, core_height, halo)
            repeat_status, repeat_density, repeat_t = _apply(library, profile, counts, core_height, halo)
            invalid = _Profile.from_buffer_copy(profile)
            invalid.sigma[0] = 100.0
            bad_status, bad_density, bad_t = _apply(library, invalid, counts, core_height, halo)
        finally:
            _ctypes.FreeLibrary(library._handle)
        outputs[name] = (density, transmittance)
        results[name] = {
            "status": status,
            "density_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
            "transmittance_sha256": hashlib.sha256(transmittance.tobytes()).hexdigest(),
            "maximum_density_absolute_error": float(np.max(np.abs(density.astype(np.float64) - reference_density))),
            "maximum_transmittance_absolute_error": float(np.max(np.abs(transmittance.astype(np.float64) - reference_t))),
            "repeat_exact": bool(repeat_status == 0 and np.array_equal(density, repeat_density) and np.array_equal(transmittance, repeat_t)),
            "failure_atomic": bool(bad_status != 0 and np.all(bad_density == -7.0) and np.all(bad_t == -9.0)),
        }
    cross = max(float(np.max(np.abs(outputs["msvc"][0] - outputs["llvm"][0]))), float(np.max(np.abs(outputs["msvc"][1] - outputs["llvm"][1]))))
    gates = contract["gates"]
    decisions = {
        "density": max(row["maximum_density_absolute_error"] for row in results.values()) <= gates["maximum_density_absolute_error"],
        "transmittance": max(row["maximum_transmittance_absolute_error"] for row in results.values()) <= gates["maximum_transmittance_absolute_error"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "repeat": all(row["repeat_exact"] for row in results.values()),
        "atomic": all(row["failure_atomic"] for row in results.values()),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path), "source_sha256": sha256_file(root / SOURCE), "header_sha256": sha256_file(root / HEADER),
        "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(), "results": results,
        "maximum_cross_compiler_absolute_error": cross, "gates": decisions,
        "decision": contract["decision_if_pass"] if all(decisions.values()) else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"],
    }
    return {"schema": "neuro_film.u6_p4eh_native_cloud_spatial_response.v1", "automatic_pass": all(decisions.values()), "stable": stable, "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


__all__ = ["evaluate"]
