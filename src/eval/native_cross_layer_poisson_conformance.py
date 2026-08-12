"""Build and execute the frozen P4EF native cross-layer Poisson kernel."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_cross_layer_poisson_region,
)

SOURCE = "native/film_physics/nf_cross_layer_poisson_u16_v1.c"
HEADER = "native/film_physics/nf_cross_layer_poisson_u16_v1.h"


class _Profile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("marginal_rates_cmy", ctypes.c_double * 3),
        ("shared_all_rate", ctypes.c_double),
        ("shared_pair_rates_cm_cy_my", ctypes.c_double * 3),
        ("seed", ctypes.c_uint64),
        ("component_seed_stride", ctypes.c_uint64),
    ]


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_cross_layer_poisson_llvm_v1.dll"
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
    library.nf_cross_layer_poisson_u16_sample_region_v1.argtypes = [
        ctypes.POINTER(_Profile),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    library.nf_cross_layer_poisson_u16_sample_region_v1.restype = ctypes.c_int
    return library


def _unload(library: ctypes.CDLL) -> None:
    if library._handle:
        _ctypes.FreeLibrary(library._handle)


def _native_profile(profile: CrossLayerPoissonProfile) -> _Profile:
    return _Profile(
        ctypes.sizeof(_Profile),
        1,
        (ctypes.c_double * 3)(*profile.marginal_rates_cmy),
        profile.shared_all_rate,
        (ctypes.c_double * 3)(*profile.shared_pair_rates_cm_cy_my),
        profile.seed,
        profile.component_seed_stride,
    )


def _apply(
    library: ctypes.CDLL,
    profile: _Profile,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, np.ndarray]:
    output = np.full((*shape, 3), np.uint16(65535), dtype=np.uint16)
    pointer = ctypes.POINTER(ctypes.c_uint16)
    status = library.nf_cross_layer_poisson_u16_sample_region_v1(
        ctypes.byref(profile),
        *full_shape,
        *origin_yx,
        *shape,
        output.ctypes.data_as(pointer),
        output.size,
    )
    return status, output


def evaluate(
    root: Path, contract_path: Path, output_dir: Path, clang: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4EF parent drift")
    if parent_payload["decision"] != contract["parent"]["required_decision"]:
        raise RuntimeError("P4EF parent decision drift")
    builds = {
        "msvc": build_msvc_c11_dll(
            root=root,
            output_dir=output_dir / "msvc",
            source_relative=SOURCE,
            header_relative=HEADER,
            basename="nf_cross_layer_poisson_msvc_v1",
        ),
        "llvm": _build_llvm(root, output_dir / "llvm", clang),
    }
    fixture = contract["fixture"]
    full_shape = (fixture["height"], fixture["width"])
    profile = CrossLayerPoissonProfile(
        tuple(fixture["marginal_rates_cmy"]),
        fixture["shared_all_rate"],
        tuple(fixture["shared_pair_rates_cm_cy_my"]),
        (0.01, 0.01, 0.01),
        fixture["seed"],
        fixture["component_seed_stride"],
    )
    reference = sample_cross_layer_poisson_region(
        profile, full_shape, origin_yx=(0, 0), shape=full_shape
    )
    native_profile = _native_profile(profile)
    outputs: dict[str, np.ndarray] = {}
    rows: dict[str, Any] = {}
    for name, build in builds.items():
        library = _load(Path(build["dll_path"]))
        try:
            status, output = _apply(
                library, native_profile, full_shape, (0, 0), full_shape
            )
            repeat_status, repeat = _apply(
                library, native_profile, full_shape, (0, 0), full_shape
            )
            partitions = {}
            for partition_height in fixture["partition_heights"]:
                parts = []
                for y in range(0, full_shape[0], partition_height):
                    part_shape = (
                        min(partition_height, full_shape[0] - y),
                        full_shape[1],
                    )
                    part_status, part = _apply(
                        library, native_profile, full_shape, (y, 0), part_shape
                    )
                    if part_status != 0:
                        raise RuntimeError("native partition failed")
                    parts.append(part)
                partitions[str(partition_height)] = bool(
                    np.array_equal(np.concatenate(parts, axis=0), output)
                )
            invalid = _native_profile(profile)
            invalid.shared_all_rate = -1.0
            bad_status, bad_output = _apply(
                library, invalid, full_shape, (0, 0), full_shape
            )
        finally:
            _unload(library)
        outputs[name] = output
        rows[name] = {
            "status": status,
            "repeat_status": repeat_status,
            "count_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "python_exact": bool(np.array_equal(output, reference)),
            "repeat_exact": bool(np.array_equal(output, repeat)),
            "partition_exact": partitions,
            "failure_atomic": bool(
                bad_status != 0 and np.all(bad_output == np.uint16(65535))
            ),
        }
    decisions = {
        "python_count_identity": all(row["python_exact"] for row in rows.values()),
        "cross_compiler_identity": bool(
            np.array_equal(outputs["msvc"], outputs["llvm"])
        ),
        "partition_identity": all(
            all(row["partition_exact"].values()) for row in rows.values()
        ),
        "repeat_identity": all(row["repeat_exact"] for row in rows.values()),
        "failure_atomicity": all(row["failure_atomic"] for row in rows.values()),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "reference_count_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "results": rows,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4ef_native_cross_layer_poisson_counts.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
