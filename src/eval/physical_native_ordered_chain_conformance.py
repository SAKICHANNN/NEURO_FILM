"""Compose the native physical primitives in the canonical domain order."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.eval.physical_native_domains_conformance import (
    build_msvc_native_domains_dll,
)
from src.eval.physical_native_print_conformance import (
    NativePrintProfileV1,
    profile_struct_from_payload,
)
from src.eval.physical_native_spatial_conformance import (
    NativeGaussianProfileV1,
    build_msvc_native_gaussian_dll,
    gaussian_profile_struct,
)
from src.film_physics.native_adjacency_profile import (
    build_native_ordered_chain_oracle,
    compile_native_adjacency_profile_payload,
    native_adjacency_payload_sha256,
)
from src.film_physics.native_profile import (
    compile_native_domains_profile_payload,
)
from src.film_physics.native_spatial_profile import (
    compile_native_gaussian_profile_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


class NativeAdjacencyProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("gain_rgb", ctypes.c_double * 3),
        ("maximum_absolute_transmittance_delta", ctypes.c_double),
        ("maximum_absolute_density_delta", ctypes.c_double),
        ("black_reference_density", ctypes.c_double * 3),
        ("white_reference_density", ctypes.c_double * 3),
    ]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_exact_json(
    root: Path, relative: str, expected_sha256: str
) -> dict[str, Any]:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"hash mismatch: {relative}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {relative}")
    return value


def adjacency_profile_struct(
    payload: dict[str, Any],
) -> NativeAdjacencyProfileV1:
    profile = NativeAdjacencyProfileV1()
    profile.struct_size = ctypes.sizeof(NativeAdjacencyProfileV1)
    profile.abi_version = 1
    profile.source_component_sha256 = payload["source_component"][
        "sha256"
    ].encode("ascii")
    for channel in range(3):
        profile.gain_rgb[channel] = float(
            payload["development_adjacency_gain_rgb"][channel]
        )
        profile.black_reference_density[channel] = float(
            payload["black_reference_density"][channel]
        )
        profile.white_reference_density[channel] = float(
            payload["white_reference_density"][channel]
        )
    profile.maximum_absolute_transmittance_delta = float(
        payload["maximum_absolute_transmittance_delta"]
    )
    profile.maximum_absolute_density_delta = float(
        payload["maximum_absolute_density_delta"]
    )
    return profile


def build_msvc_native_adjacency_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_bounded_adjacency_v1.c",
        header_relative="native/film_physics/nf_bounded_adjacency_v1.h",
        basename="nf_bounded_adjacency_v1",
    )


def _load_domains(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    common = [
        ctypes.POINTER(NativePrintProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_physical_sensitometry_apply_v1.argtypes = common
    library.nf_physical_sensitometry_apply_v1.restype = ctypes.c_int
    library.nf_physical_interpretation_apply_v1.argtypes = common
    library.nf_physical_interpretation_apply_v1.restype = ctypes.c_int
    return library


def _load_gaussian(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_gaussian_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_gaussian_apply_v1.restype = ctypes.c_int
    return library


def _load_adjacency(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_bounded_adjacency_apply_v1.argtypes = [
        ctypes.POINTER(NativeAdjacencyProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_bounded_adjacency_apply_v1.restype = ctypes.c_int
    return library


def run_loaded_ordered_chain(
    *,
    domains_dll: Path,
    gaussian_dll: Path,
    adjacency_dll: Path,
    domains_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    adjacency_payload: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    domains = _load_domains(domains_dll)
    gaussian = _load_gaussian(gaussian_dll)
    adjacency = _load_adjacency(adjacency_dll)
    domains_profile = profile_struct_from_payload(domains_payload)
    adjacency_profile = adjacency_profile_struct(adjacency_payload)
    stages = {
        row["stage"]: row for row in spatial_payload["stages"]
    }
    source = np.ascontiguousarray(
        oracle["input_scene_linear_f64"], dtype=np.float64
    )
    height, width, channels = source.shape
    count = height * width
    if channels != 3:
        raise RuntimeError("ordered-chain oracle shape drift")
    tolerance = float(
        oracle["comparison"]["stage_max_abs_tolerance"]
    )
    observed: dict[str, np.ndarray] = {}

    def blur(values: np.ndarray, stage_name: str) -> np.ndarray:
        profile = gaussian_profile_struct(
            spatial_payload, stages[stage_name]
        )
        workspace = np.empty_like(values)
        output = np.empty_like(values)
        status = gaussian.nf_gaussian_apply_v1(
            ctypes.byref(profile),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            height,
            width,
            workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            workspace.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"native blur failed: {stage_name}")
        return output

    forward = blur(source, "forward_scatter")
    observed["forward_scatter"] = forward
    density = np.empty_like(source)
    if domains.nf_physical_sensitometry_apply_v1(
        ctypes.byref(domains_profile),
        forward.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        count,
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ) != 0:
        raise RuntimeError("native ordered sensitometry failed")
    observed["developed_density"] = density
    blurred_density = blur(density, "development_adjacency")
    adjacent = np.empty_like(source)
    if adjacency.nf_bounded_adjacency_apply_v1(
        ctypes.byref(adjacency_profile),
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        blurred_density.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
        count,
        adjacent.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ) != 0:
        raise RuntimeError("native bounded adjacency failed")
    observed["bounded_adjacency"] = adjacent
    diffused = blur(adjacent, "dye_diffusion")
    observed["dye_diffusion"] = diffused
    interpreted = np.empty_like(source)
    if domains.nf_physical_interpretation_apply_v1(
        ctypes.byref(domains_profile),
        diffused.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        count,
        interpreted.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ) != 0:
        raise RuntimeError("native ordered interpretation failed")
    observed["interpretation"] = interpreted
    scanned = blur(interpreted, "scanner_mtf")
    observed["scanner_mtf"] = scanned

    rows = []
    for name in (
        "forward_scatter",
        "developed_density",
        "bounded_adjacency",
        "dye_diffusion",
        "interpretation",
        "scanner_mtf",
    ):
        expected = np.asarray(
            oracle["expected_by_stage_f64"][name], dtype=np.float64
        )
        error = float(np.max(np.abs(observed[name] - expected)))
        if error > tolerance:
            raise RuntimeError(
                f"ordered native stage {name} error {error} exceeds "
                f"{tolerance}"
            )
        rows.append(
            {
                "stage": name,
                "maximum_absolute_error": error,
                "tolerance": tolerance,
                "array_sha256": hashlib.sha256(
                    observed[name].tobytes()
                ).hexdigest(),
            }
        )
    return {
        "status": "pass",
        "shape": list(source.shape),
        "stages": rows,
        "final_array_sha256": rows[-1]["array_sha256"],
    }


def run_native_ordered_chain_conformance(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AA"):
        raise ValueError("P8AA parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    oracle = build_native_ordered_chain_oracle(
        artifact, adjacency_payload
    )
    if (
        native_adjacency_payload_sha256(adjacency_payload)
        != config["expected_adjacency_payload_sha256"]
        or oracle["oracle_sha256"] != config["expected_oracle_sha256"]
    ):
        raise ValueError("P8AA frozen payload or oracle identity drift")

    builds = {}
    for name, builder in (
        ("domains", build_msvc_native_domains_dll),
        ("gaussian", build_msvc_native_gaussian_dll),
        ("adjacency", build_msvc_native_adjacency_dll),
    ):
        first = builder(root=root, output_dir=output_dir / f"{name}_a")
        second = builder(root=root, output_dir=output_dir / f"{name}_b")
        expected = config["component_sources"][name]
        if (
            first["source_sha256"] != expected["source_sha256"]
            or first["header_sha256"] != expected["header_sha256"]
            or second["source_sha256"] != expected["source_sha256"]
            or second["header_sha256"] != expected["header_sha256"]
            or first["dll_sha256"] != second["dll_sha256"]
        ):
            raise ValueError(f"P8AA {name} build identity drift")
        builds[name] = (first, second)

    replays = []
    for index in range(2):
        replays.append(
            run_loaded_ordered_chain(
                domains_dll=Path(builds["domains"][index]["dll_path"]),
                gaussian_dll=Path(builds["gaussian"][index]["dll_path"]),
                adjacency_dll=Path(
                    builds["adjacency"][index]["dll_path"]
                ),
                domains_payload=domains_payload,
                spatial_payload=spatial_payload,
                adjacency_payload=adjacency_payload,
                oracle=oracle,
            )
        )
    if replays[0] != replays[1]:
        raise RuntimeError("independent ordered-chain replay drift")
    stable_core = {
        "schema": "neuro_film.u6_p8aa_native_ordered_chain.v1",
        "adjacency_payload_sha256": native_adjacency_payload_sha256(
            adjacency_payload
        ),
        "oracle_sha256": oracle["oracle_sha256"],
        "component_dll_sha256": {
            name: rows[0]["dll_sha256"] for name, rows in builds.items()
        },
        "independent_build_dll_sha_exact": True,
        "replays": replays,
        "decision": (
            "pass the fixed small-image native physical chain in canonical "
            "domain order; neutral gauge, AO6 display look, full-resolution "
            "performance and product integration remain open"
        ),
        "claim_ceiling": adjacency_payload["claim_ceiling"],
        "production_default_changed": False,
    }
    return {
        **stable_core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
