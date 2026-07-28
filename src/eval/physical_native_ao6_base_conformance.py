"""Conformance audit for the active native AO6 source-context base."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.film_physics.display_look import (
    build_density_source_context_row_staged,
    build_source_context_display_look_stages,
)
from src.film_physics.native_ao6_base_profile import (
    NativeAo6BaseContextF32V1,
    NativeAo6BaseProfileF32V1,
    native_ao6_base_context_struct,
    native_ao6_base_display_payload_sha256,
    native_ao6_base_profile_struct,
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


def build_msvc_native_ao6_base_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_ao6_base_f32_v1.c",
        header_relative="native/film_physics/nf_ao6_base_f32_v1.h",
        basename="nf_ao6_base_f32_v1",
    )


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_base_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_base_f32_apply_v1.restype = ctypes.c_int
    return library


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_native_ao6_base_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AJ"):
        raise ValueError("P8AJ parent decision drift")
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
        raise ValueError("P8AJ display payload drift")
    profile = native_ao6_base_profile_struct(display)
    builds = [
        build_msvc_native_ao6_base_dll(
            root=root, output_dir=output_dir / f"build_{index}"
        )
        for index in range(2)
    ]
    expected_source = config["native_source"]
    for build in builds:
        if (
            build["source_sha256"] != expected_source["source_sha256"]
            or build["header_sha256"] != expected_source["header_sha256"]
        ):
            raise ValueError("P8AJ source identity drift")
    if builds[0]["dll_sha256"] != builds[1]["dll_sha256"]:
        raise RuntimeError("P8AJ native build drift")

    height = int(config["oracle"]["height"])
    width = int(config["oracle"]["width"])
    random = np.random.default_rng(int(config["oracle"]["seed"]))
    source = random.random((height, width, 3), dtype=np.float32)
    encoded = random.random((height, width, 3), dtype=np.float32)
    source[0, 0] = (0.0, 0.0, 0.0)
    source[0, 1] = (1.0, 1.0, 1.0)
    encoded[0, 0] = (0.0, 0.5, 1.0)
    encoded[0, 1] = (1.0, 0.0, 0.75)
    source_context = build_density_source_context_row_staged(
        display, source, tile_rows=7
    )
    context = native_ao6_base_context_struct(source_context)
    apply_base, _ = build_source_context_display_look_stages(
        display, source
    )
    expected = np.asarray(apply_base(encoded), dtype=np.float32)
    flat_input = np.ascontiguousarray(encoded.reshape(-1, 3))
    expected_flat = np.ascontiguousarray(expected.reshape(-1, 3))
    replays = []
    for build in builds:
        library = _load(Path(build["dll_path"]))
        output = np.empty_like(flat_input)
        status = library.nf_ao6_base_f32_apply_v1(
            ctypes.byref(profile),
            ctypes.byref(context),
            _pointer(flat_input),
            flat_input.shape[0],
            _pointer(output),
        )
        if status != 0:
            raise RuntimeError(f"P8AJ native base failed: {status}")
        inplace = flat_input.copy()
        status = library.nf_ao6_base_f32_apply_v1(
            ctypes.byref(profile),
            ctypes.byref(context),
            _pointer(inplace),
            inplace.shape[0],
            _pointer(inplace),
        )
        if status != 0 or inplace.tobytes() != output.tobytes():
            raise RuntimeError("P8AJ in-place drift")
        partition_rows = []
        for tile_rows in (1, 7, 32, height):
            partitioned = np.empty_like(flat_input)
            for y0 in range(0, height, tile_rows):
                y1 = min(height, y0 + tile_rows)
                start = y0 * width
                stop = y1 * width
                input_rows = np.ascontiguousarray(
                    flat_input[start:stop]
                )
                status = library.nf_ao6_base_f32_apply_v1(
                    ctypes.byref(profile),
                    ctypes.byref(context),
                    _pointer(input_rows),
                    input_rows.shape[0],
                    _pointer(partitioned[start:stop]),
                )
                if status != 0:
                    raise RuntimeError(
                        f"P8AJ partition {tile_rows} failed: {status}"
                    )
            partition_rows.append(
                {
                    "tile_rows": tile_rows,
                    "byte_exact": (
                        partitioned.tobytes() == output.tobytes()
                    ),
                }
            )
        if not all(row["byte_exact"] for row in partition_rows):
            raise RuntimeError("P8AJ partition drift")
        invalid = flat_input.copy()
        invalid[0, 0] = np.float32(np.nan)
        sentinel = np.full_like(flat_input, np.float32(-9.0))
        before = sentinel.tobytes()
        status = library.nf_ao6_base_f32_apply_v1(
            ctypes.byref(profile),
            ctypes.byref(context),
            _pointer(invalid),
            invalid.shape[0],
            _pointer(sentinel),
        )
        if status == 0 or sentinel.tobytes() != before:
            raise RuntimeError("P8AJ invalid input changed output")
        difference = np.abs(
            output.astype(np.float64) -
            expected_flat.astype(np.float64)
        )
        replays.append(
            {
                "maximum_absolute_error": float(np.max(difference)),
                "p999_absolute_error": float(
                    np.quantile(difference, 0.999)
                ),
                "output_sha256": hashlib.sha256(
                    output.tobytes()
                ).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    expected_flat.tobytes()
                ).hexdigest(),
                "in_place_byte_exact": True,
                "partition_rows": partition_rows,
                "invalid_input_output_unchanged": True,
                "minimum_output": float(np.min(output)),
                "maximum_output": float(np.max(output)),
            }
        )
    if replays[0] != replays[1]:
        raise RuntimeError("P8AJ independent replay drift")
    tolerance = float(config["gates"]["maximum_absolute_error"])
    if replays[0]["maximum_absolute_error"] > tolerance:
        raise RuntimeError(
            "P8AJ native base error "
            f"{replays[0]['maximum_absolute_error']} exceeds {tolerance}"
        )
    stable = {
        "schema": "neuro_film.u6_p8aj_native_ao6_base.v1",
        "display_payload_sha256": display_sha,
        "dll_sha256": builds[0]["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "source_context": {
            "pixel_count": source_context.pixel_count,
            "lab_mean": list(source_context.lab_mean),
            "lab_std": list(source_context.lab_std),
            "boundary": (
                "six externally supplied float32 Lab statistics; image "
                "pixels are not consumed by Python after reduction"
            ),
        },
        "oracle": {
            "shape": [height, width, 3],
            **replays[0],
            "tolerance": tolerance,
        },
        "active_contract": (
            "density strength 0.5 -> Lab affine statistics transfer -> "
            "14-iteration fixed-L chroma gamut compression -> margin 4"
        ),
        "excluded_zero_features": [
            "grain",
            "tone_rolloff",
            "local_luma_detail",
            "chroma_curve",
            "neutral_protect",
            "skin_protect",
            "chroma_caps",
            "dither",
        ],
        "decision": (
            "pass the minimal active native AO6 source-context base; do not "
            "port inactive legacy safe-Lab features"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AK replace the external six-float source-context "
            "reduction with a deterministic native streaming reducer and "
            "compose the complete native display-look chain"
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
