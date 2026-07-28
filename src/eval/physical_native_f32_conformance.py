"""Validate the native Standard float32 physical chain against Reference."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.eval.physical_native_ordered_chain_conformance import (
    NativeAdjacencyProfileV1,
    adjacency_profile_struct,
)
from src.eval.physical_native_print_conformance import (
    NativePrintProfileV1,
    profile_struct_from_payload,
)
from src.eval.physical_native_spatial_conformance import (
    NativeGaussianProfileV1,
    gaussian_profile_struct,
)
from src.film_physics.native_adjacency_profile import (
    build_native_ordered_chain_oracle,
    compile_native_adjacency_profile_payload,
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


def _build_component(
    *,
    root: Path,
    output_dir: Path,
    component: dict[str, str],
    basename: str,
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=component["source"],
        header_relative=component["header"],
        basename=basename,
    )


def _load_domains(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    arguments = [
        ctypes.POINTER(NativePrintProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_physical_sensitometry_f32_apply_v1.argtypes = arguments
    library.nf_physical_sensitometry_f32_apply_v1.restype = ctypes.c_int
    library.nf_physical_interpretation_f32_apply_v1.argtypes = arguments
    library.nf_physical_interpretation_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_gaussian(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_gaussian_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_gaussian_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_adjacency(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_bounded_adjacency_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeAdjacencyProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_bounded_adjacency_f32_apply_v1.restype = ctypes.c_int
    return library


def _float_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _render_f32_chain(
    *,
    domains_dll: Path,
    gaussian_dll: Path,
    adjacency_dll: Path,
    domains_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    adjacency_payload: dict[str, Any],
    source: np.ndarray,
    audit_failure_atomic: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    domains = _load_domains(domains_dll)
    gaussian = _load_gaussian(gaussian_dll)
    adjacency = _load_adjacency(adjacency_dll)
    domains_profile = profile_struct_from_payload(domains_payload)
    adjacency_profile = adjacency_profile_struct(adjacency_payload)
    stages = {row["stage"]: row for row in spatial_payload["stages"]}
    source = np.ascontiguousarray(source, dtype=np.float32)
    height, width, channels = source.shape
    if channels != 3:
        raise ValueError("float32 chain requires interleaved RGB")
    count = height * width

    def blur(values: np.ndarray, stage: str) -> np.ndarray:
        profile = gaussian_profile_struct(
            spatial_payload, stages[stage]
        )
        workspace = np.empty_like(values)
        output = np.empty_like(values)
        status = gaussian.nf_gaussian_f32_apply_v1(
            ctypes.byref(profile),
            _float_pointer(values),
            height,
            width,
            _float_pointer(workspace),
            workspace.size,
            _float_pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"float32 blur failed: {stage}: {status}")
        return output

    observed: dict[str, np.ndarray] = {}
    forward = blur(source, "forward_scatter")
    observed["forward_scatter"] = forward
    density = np.empty_like(source)
    if domains.nf_physical_sensitometry_f32_apply_v1(
        ctypes.byref(domains_profile),
        _float_pointer(forward),
        count,
        _float_pointer(density),
    ) != 0:
        raise RuntimeError("float32 sensitometry failed")
    observed["developed_density"] = density
    blurred_density = blur(density, "development_adjacency")
    adjacent = np.empty_like(source)
    if adjacency.nf_bounded_adjacency_f32_apply_v1(
        ctypes.byref(adjacency_profile),
        _float_pointer(density),
        _float_pointer(blurred_density),
        count,
        _float_pointer(adjacent),
    ) != 0:
        raise RuntimeError("float32 bounded adjacency failed")
    observed["bounded_adjacency"] = adjacent
    diffused = blur(adjacent, "dye_diffusion")
    observed["dye_diffusion"] = diffused
    interpreted = np.empty_like(source)
    if domains.nf_physical_interpretation_f32_apply_v1(
        ctypes.byref(domains_profile),
        _float_pointer(diffused),
        count,
        _float_pointer(interpreted),
    ) != 0:
        raise RuntimeError("float32 interpretation failed")
    observed["interpretation"] = interpreted
    observed["scanner_mtf"] = blur(interpreted, "scanner_mtf")

    failure_atomic: dict[str, bool] = {}
    if not audit_failure_atomic:
        return observed, failure_atomic
    sentinel = np.full_like(source, np.float32(-19.0))
    invalid = source.copy()
    invalid.reshape(-1)[0] = np.float32(np.nan)
    before = sentinel.tobytes()
    status = domains.nf_physical_sensitometry_f32_apply_v1(
        ctypes.byref(domains_profile),
        _float_pointer(invalid),
        count,
        _float_pointer(sentinel),
    )
    failure_atomic["domains"] = status != 0 and sentinel.tobytes() == before

    gaussian_profile = gaussian_profile_struct(
        spatial_payload, stages["forward_scatter"]
    )
    invalid = source.copy()
    invalid.reshape(-1)[0] = np.float32(-1.0)
    workspace = np.full_like(source, np.float32(-17.0))
    output = np.full_like(source, np.float32(-13.0))
    before_workspace = workspace.tobytes()
    before_output = output.tobytes()
    status = gaussian.nf_gaussian_f32_apply_v1(
        ctypes.byref(gaussian_profile),
        _float_pointer(invalid),
        height,
        width,
        _float_pointer(workspace),
        workspace.size,
        _float_pointer(output),
    )
    failure_atomic["gaussian"] = (
        status != 0
        and workspace.tobytes() == before_workspace
        and output.tobytes() == before_output
    )

    invalid_density = density.copy()
    invalid_density.reshape(-1)[0] = np.float32(np.nan)
    output = np.full_like(source, np.float32(-11.0))
    before = output.tobytes()
    status = adjacency.nf_bounded_adjacency_f32_apply_v1(
        ctypes.byref(adjacency_profile),
        _float_pointer(invalid_density),
        _float_pointer(blurred_density),
        count,
        _float_pointer(output),
    )
    failure_atomic["adjacency"] = (
        status != 0 and output.tobytes() == before
    )
    return observed, failure_atomic


def render_tiled_f32_chain(
    *,
    domains_dll: Path,
    gaussian_dll: Path,
    adjacency_dll: Path,
    domains_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    adjacency_payload: dict[str, Any],
    source: np.ndarray,
    tile_rows: int,
    reverse: bool = False,
) -> np.ndarray:
    """Render the Standard chain with the canonical summed vertical halo."""
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    source = np.ascontiguousarray(source, dtype=np.float32)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or source.size == 0
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("float32 tiled source must be finite [0,1] HxWx3")
    height = int(source.shape[0])
    halo = sum(
        int(row["maximum_radius"]) for row in spatial_payload["stages"]
    )
    starts = list(range(0, height, tile_rows))
    if reverse:
        starts.reverse()
    output = np.empty_like(source)
    for y0 in starts:
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        stages, _ = _render_f32_chain(
            domains_dll=domains_dll,
            gaussian_dll=gaussian_dll,
            adjacency_dll=adjacency_dll,
            domains_payload=domains_payload,
            spatial_payload=spatial_payload,
            adjacency_payload=adjacency_payload,
            source=source[source_y0:source_y1],
            audit_failure_atomic=False,
        )
        output[y0:y1] = stages["scanner_mtf"][
            y0 - source_y0 : y1 - source_y0
        ]
    return output


def run_native_f32_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AD"):
        raise ValueError("P8AD parent decision drift")
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

    basenames = {
        "domains": "nf_physical_domains_f32_v1",
        "gaussian": "nf_gaussian_rgb_f32_v1",
        "adjacency": "nf_bounded_adjacency_f32_v1",
    }
    builds: dict[str, list[dict[str, Any]]] = {}
    for name, basename in basenames.items():
        component = config["component_sources"][name]
        rows = [
            _build_component(
                root=root,
                output_dir=output_dir / f"{name}_{index}",
                component=component,
                basename=basename,
            )
            for index in range(2)
        ]
        for row in rows:
            if (
                row["source_sha256"] != component["source_sha256"]
                or row["header_sha256"] != component["header_sha256"]
            ):
                raise ValueError(f"P8AD {name} source identity drift")
        if rows[0]["dll_sha256"] != rows[1]["dll_sha256"]:
            raise RuntimeError(f"P8AD {name} build reproducibility failed")
        builds[name] = rows

    source = np.asarray(
        oracle["input_scene_linear_f64"], dtype=np.float32
    )
    replays = []
    for index in range(2):
        observed, failure_atomic = _render_f32_chain(
            domains_dll=Path(builds["domains"][index]["dll_path"]),
            gaussian_dll=Path(builds["gaussian"][index]["dll_path"]),
            adjacency_dll=Path(
                builds["adjacency"][index]["dll_path"]
            ),
            domains_payload=domains_payload,
            spatial_payload=spatial_payload,
            adjacency_payload=adjacency_payload,
            source=source,
        )
        rows = []
        for stage in (
            "forward_scatter",
            "developed_density",
            "bounded_adjacency",
            "dye_diffusion",
            "interpretation",
            "scanner_mtf",
        ):
            expected = np.asarray(
                oracle["expected_by_stage_f64"][stage],
                dtype=np.float64,
            )
            actual = observed[stage]
            rows.append(
                {
                    "stage": stage,
                    "maximum_absolute_error_vs_float64": float(
                        np.max(np.abs(actual.astype(np.float64) - expected))
                    ),
                    "array_sha256": hashlib.sha256(
                        actual.tobytes()
                    ).hexdigest(),
                }
            )
        replays.append(
            {
                "stages": rows,
                "failure_atomic": failure_atomic,
                "final_array_sha256": rows[-1]["array_sha256"],
            }
        )
    if replays[0] != replays[1]:
        raise RuntimeError("P8AD independent replay drift")
    if not all(replays[0]["failure_atomic"].values()):
        raise RuntimeError("P8AD invalid input mutated native output")
    tolerance = float(
        config["gates"]["maximum_stage_absolute_error_vs_float64"]
    )
    worst = max(
        row["maximum_absolute_error_vs_float64"]
        for row in replays[0]["stages"]
    )
    if worst > tolerance:
        raise RuntimeError(
            f"P8AD float32 error {worst} exceeds {tolerance}"
        )
    stable = {
        "schema": "neuro_film.u6_p8ad_native_standard_f32.v1",
        "component_dll_sha256": {
            name: rows[0]["dll_sha256"]
            for name, rows in builds.items()
        },
        "independent_build_dll_sha_exact": True,
        "sample_storage": "float32",
        "profile_parameters": "float64",
        "maximum_absolute_error_vs_float64": worst,
        "tolerance": tolerance,
        "replay": replays[0],
        "decision": (
            "pass the separate native Standard float32 ABIs for the fixed "
            "small-image physical chain; full-resolution resources, display "
            "completion, device runtimes and product integration remain open"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AE native Standard float32 12MP tiled resource comparison"
        ),
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
