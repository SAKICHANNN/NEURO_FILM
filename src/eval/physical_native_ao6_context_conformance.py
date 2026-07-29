"""Conformance audit for native streaming AO6 source-context reduction."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.eval.physical_native_ao6_base_conformance import (
    _load as _load_base,
    _pointer,
    build_msvc_native_ao6_base_dll,
)
from src.film_physics.display_look import (
    build_density_source_context_row_staged,
    build_source_context_display_look_stages,
)
from src.film_physics.native_ao6_base_profile import (
    NativeAo6BaseContextF32V1,
    NativeAo6BaseProfileF32V1,
    native_ao6_base_display_payload_sha256,
    native_ao6_base_profile_struct,
)
from src.film_physics.native_ao6_context_profile import (
    NativeAo6ContextStateF32V1,
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


def build_msvc_native_ao6_context_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_ao6_context_f32_v1.c",
        header_relative="native/film_physics/nf_ao6_context_f32_v1.h",
        basename="nf_ao6_context_f32_v1",
    )


def _load_context(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_context_f32_init_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1)
    ]
    library.nf_ao6_context_f32_init_v1.restype = ctypes.c_int
    library.nf_ao6_context_f32_update_v1.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.nf_ao6_context_f32_update_v1.restype = ctypes.c_int
    library.nf_ao6_context_f32_finalize_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
    ]
    library.nf_ao6_context_f32_finalize_v1.restype = ctypes.c_int
    return library


def _native_context(
    *,
    library: ctypes.CDLL,
    profile: NativeAo6BaseProfileF32V1,
    source: np.ndarray,
    tile_rows: int,
) -> tuple[NativeAo6BaseContextF32V1, str]:
    state = NativeAo6ContextStateF32V1()
    if library.nf_ao6_context_f32_init_v1(ctypes.byref(state)) != 0:
        raise RuntimeError("P8AK context init failed")
    height, width, _ = source.shape
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        rows = np.ascontiguousarray(source[y0:y1].reshape(-1, 3))
        status = library.nf_ao6_context_f32_update_v1(
            ctypes.byref(profile),
            ctypes.byref(state),
            _pointer(rows),
            rows.shape[0],
        )
        if status != 0:
            raise RuntimeError(
                f"P8AK context update {tile_rows} failed: {status}"
            )
    context = NativeAo6BaseContextF32V1()
    if library.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise RuntimeError("P8AK context finalize failed")
    return context, hashlib.sha256(bytes(context)).hexdigest()


def run_native_ao6_context_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AK"):
        raise ValueError("P8AK parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    display = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    display_sha = native_ao6_base_display_payload_sha256(display)
    if display_sha != config["expected_display_payload_sha256"]:
        raise ValueError("P8AK display payload drift")
    profile = native_ao6_base_profile_struct(display)
    context_builds = [
        build_msvc_native_ao6_context_dll(
            root=root, output_dir=output_dir / f"context_{index}"
        )
        for index in range(2)
    ]
    base_build = build_msvc_native_ao6_base_dll(
        root=root, output_dir=output_dir / "base"
    )
    expected_source = config["native_source"]
    for build in context_builds:
        if (
            build["source_sha256"] != expected_source["source_sha256"]
            or build["header_sha256"] != expected_source["header_sha256"]
        ):
            raise ValueError("P8AK source identity drift")
    if context_builds[0]["dll_sha256"] != context_builds[1]["dll_sha256"]:
        raise RuntimeError("P8AK independent context build drift")

    height = int(config["oracle"]["height"])
    width = int(config["oracle"]["width"])
    random = np.random.default_rng(int(config["oracle"]["seed"]))
    source = random.random((height, width, 3), dtype=np.float32)
    encoded = random.random((height, width, 3), dtype=np.float32)
    source[0, 0] = (0.0, 0.0, 0.0)
    source[0, 1] = (1.0, 1.0, 1.0)
    encoded[0, 0] = (0.0, 0.5, 1.0)
    encoded[0, 1] = (1.0, 0.0, 0.75)
    python_context = build_density_source_context_row_staged(
        display, source, tile_rows=7
    )
    apply_base, _ = build_source_context_display_look_stages(
        display, source
    )
    python_output = np.asarray(apply_base(encoded), dtype=np.float32)

    context_library = _load_context(
        Path(context_builds[0]["dll_path"])
    )
    partition_rows = []
    contexts = []
    for tile_rows in config["tile_rows"]:
        context, context_sha = _native_context(
            library=context_library,
            profile=profile,
            source=source,
            tile_rows=int(tile_rows),
        )
        contexts.append(context)
        partition_rows.append(
            {
                "tile_rows": int(tile_rows),
                "context_sha256": context_sha,
            }
        )
    if len({row["context_sha256"] for row in partition_rows}) != 1:
        raise RuntimeError("P8AK context partition drift")
    context = contexts[0]
    native_mean = np.asarray(list(context.source_mean), dtype=np.float32)
    native_std = np.asarray(list(context.source_std), dtype=np.float32)
    python_mean = np.asarray(python_context.lab_mean, dtype=np.float32)
    python_std = np.asarray(python_context.lab_std, dtype=np.float32)
    mean_error = float(
        np.max(np.abs(native_mean.astype(np.float64) - python_mean))
    )
    std_error = float(
        np.max(np.abs(native_std.astype(np.float64) - python_std))
    )

    base_library = _load_base(Path(base_build["dll_path"]))
    flat_input = np.ascontiguousarray(encoded.reshape(-1, 3))
    native_output = np.empty_like(flat_input)
    status = base_library.nf_ao6_base_f32_apply_v1(
        ctypes.byref(profile),
        ctypes.byref(context),
        _pointer(flat_input),
        flat_input.shape[0],
        _pointer(native_output),
    )
    if status != 0:
        raise RuntimeError(f"P8AK native base failed: {status}")
    difference = np.abs(
        native_output.astype(np.float64) -
        python_output.reshape(-1, 3).astype(np.float64)
    )
    maximum_output_error = float(np.max(difference))
    gates = config["gates"]
    if mean_error > float(gates["maximum_context_mean_error"]):
        raise RuntimeError(f"P8AK mean error exceeds gate: {mean_error}")
    if std_error > float(gates["maximum_context_std_error"]):
        raise RuntimeError(f"P8AK std error exceeds gate: {std_error}")
    if maximum_output_error > float(gates["maximum_output_error"]):
        raise RuntimeError(
            f"P8AK output error exceeds gate: {maximum_output_error}"
        )

    invalid = source[:1].copy().reshape(-1, 3)
    invalid[0, 0] = np.float32(np.nan)
    state = NativeAo6ContextStateF32V1()
    context_library.nf_ao6_context_f32_init_v1(ctypes.byref(state))
    before = bytes(state)
    invalid_status = context_library.nf_ao6_context_f32_update_v1(
        ctypes.byref(profile),
        ctypes.byref(state),
        _pointer(np.ascontiguousarray(invalid)),
        invalid.shape[0],
    )
    invalid_unchanged = invalid_status != 0 and bytes(state) == before
    if not invalid_unchanged:
        raise RuntimeError("P8AK invalid update changed reducer state")

    stable = {
        "schema": "neuro_film.u6_p8ak_native_ao6_context.v1",
        "display_payload_sha256": display_sha,
        "context_dll_sha256": context_builds[0]["dll_sha256"],
        "base_dll_sha256": base_build["dll_sha256"],
        "independent_context_build_dll_sha_exact": True,
        "partition_rows": partition_rows,
        "partition_context_byte_exact": True,
        "invalid_update_state_unchanged": True,
        "oracle": {
            "shape": [height, width, 3],
            "native_mean": native_mean.tolist(),
            "python_mean": python_mean.tolist(),
            "maximum_context_mean_error": mean_error,
            "native_std": native_std.tolist(),
            "python_std": python_std.tolist(),
            "maximum_context_std_error": std_error,
            "maximum_output_error": maximum_output_error,
            "p999_output_error": float(np.quantile(difference, 0.999)),
            "native_output_sha256": hashlib.sha256(
                native_output.tobytes()
            ).hexdigest(),
            "python_output_sha256": hashlib.sha256(
                python_output.tobytes()
            ).hexdigest(),
        },
        "decision": (
            "pass the native row-major streaming source-context reducer; "
            "the six-float Python image boundary is removed and native base "
            "output remains within the frozen Python-reference tolerance"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AL compose native context, AO6 base, explicit sRGB "
            "transfer and native t15/c35 residual as one row-streamed "
            "display-look execution with Python-reference and partition gates"
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
