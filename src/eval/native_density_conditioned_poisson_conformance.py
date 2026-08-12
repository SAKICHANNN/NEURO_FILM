"""P4EG native density-conditioned shared-count conformance."""

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
    sample_density_conditioned_cross_layer_poisson_region,
)

SOURCE = "native/film_physics/nf_density_conditioned_poisson_u16_v2.c"
HEADER = "native/film_physics/nf_density_conditioned_poisson_u16_v2.h"


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
    dll = output_dir / "nf_density_conditioned_poisson_llvm_v2.dll"
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
    return {"dll_path": str(dll.resolve()), "dll_sha256": sha256_file(dll)}


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_density_conditioned_poisson_u16_sample_region_v2.argtypes = [
        ctypes.POINTER(_Profile),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    library.nf_density_conditioned_poisson_u16_sample_region_v2.restype = ctypes.c_int
    return library


def _profile(value: CrossLayerPoissonProfile) -> _Profile:
    return _Profile(
        ctypes.sizeof(_Profile),
        2,
        (ctypes.c_double * 3)(*value.marginal_rates_cmy),
        value.shared_all_rate,
        (ctypes.c_double * 3)(*value.shared_pair_rates_cm_cy_my),
        value.seed,
        value.component_seed_stride,
    )


def _apply(
    library: ctypes.CDLL,
    profile: _Profile,
    scale: np.ndarray,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
) -> tuple[int, np.ndarray]:
    values = np.ascontiguousarray(scale, dtype=np.float32)
    output = np.full(values.shape, np.uint16(65535), dtype=np.uint16)
    status = library.nf_density_conditioned_poisson_u16_sample_region_v2(
        ctypes.byref(profile),
        *full_shape,
        *origin_yx,
        *values.shape[:2],
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
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
        raise RuntimeError("P4EG parent drift")
    if parent_payload["decision"] != contract["parent"]["required_decision"]:
        raise RuntimeError("P4EG parent decision drift")
    builds = {
        "msvc": build_msvc_c11_dll(
            root=root,
            output_dir=output_dir / "msvc",
            source_relative=SOURCE,
            header_relative=HEADER,
            basename="nf_density_conditioned_poisson_msvc_v2",
        ),
        "llvm": _build_llvm(root, output_dir / "llvm", clang),
    }
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    rng = np.random.default_rng(fixture["scale_seed"])
    scale = np.ascontiguousarray(rng.uniform(0.0, 1.0, (*shape, 3)), dtype=np.float32)
    scale[0, :, :] = 0.0
    python_profile = CrossLayerPoissonProfile(
        tuple(fixture["marginal_rates_cmy"]),
        fixture["shared_all_rate"],
        tuple(fixture["shared_pair_rates_cm_cy_my"]),
        (0.01, 0.01, 0.01),
        fixture["profile_seed"],
        fixture["component_seed_stride"],
    )
    reference = sample_density_conditioned_cross_layer_poisson_region(
        python_profile, scale, shape, origin_yx=(0, 0)
    )
    native_profile = _profile(python_profile)
    outputs = {}
    results = {}
    for name, build in builds.items():
        library = _load(Path(build["dll_path"]))
        try:
            status, output = _apply(library, native_profile, scale, shape, (0, 0))
            partitions = {}
            for height in fixture["partition_heights"]:
                parts = []
                for y in range(0, shape[0], height):
                    part_scale = scale[y : min(y + height, shape[0])]
                    part_status, part = _apply(
                        library, native_profile, part_scale, shape, (y, 0)
                    )
                    if part_status != 0:
                        raise RuntimeError("native conditioned partition failed")
                    parts.append(part)
                partitions[str(height)] = bool(
                    np.array_equal(np.concatenate(parts), output)
                )
            invalid = scale.copy()
            invalid[-1, -1, -1] = np.nan
            bad_status, bad_output = _apply(
                library, native_profile, invalid, shape, (0, 0)
            )
        finally:
            _ctypes.FreeLibrary(library._handle)
        outputs[name] = output
        results[name] = {
            "status": status,
            "count_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "python_exact": bool(np.array_equal(output, reference)),
            "partition_exact": partitions,
            "zero_scale_zero_counts": bool(np.all(output[0] == 0)),
            "failure_atomic": bool(
                bad_status != 0 and np.all(bad_output == np.uint16(65535))
            ),
        }
    decisions = {
        "python_count_identity": all(row["python_exact"] for row in results.values()),
        "cross_compiler_identity": bool(np.array_equal(outputs["msvc"], outputs["llvm"])),
        "partition_identity": all(all(row["partition_exact"].values()) for row in results.values()),
        "zero_scale_zero_counts": all(row["zero_scale_zero_counts"] for row in results.values()),
        "failure_atomicity": all(row["failure_atomic"] for row in results.values()),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "reference_count_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "results": results,
        "gates": decisions,
        "decision": contract["decision_if_pass"] if all(decisions.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4eg_native_density_conditioned_poisson.v2",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
