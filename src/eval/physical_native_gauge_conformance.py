"""Conformance audit for the post-scan native float32 neutral gauge."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
)
from src.film_physics.native_gauge_profile import (
    NativeGaugeProfileF32V1,
    native_gauge_payload_sha256,
    native_gauge_profile_struct,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
    reconstruct_standalone_runtime,
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


def build_msvc_native_gauge_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=(
            "native/film_physics/nf_neutral_gauge_f32_v1.c"
        ),
        header_relative=(
            "native/film_physics/nf_neutral_gauge_f32_v1.h"
        ),
        basename="nf_neutral_gauge_f32_v1",
    )


def _load_gauge(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_neutral_gauge_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaugeProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_neutral_gauge_f32_apply_v1.restype = ctypes.c_int
    return library


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_native_gauge_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AG"):
        raise ValueError("P8AG parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    payload = artifact["component_payloads"]["neutral-axis-gauge"]
    payload_sha = native_gauge_payload_sha256(payload)
    if payload_sha != config["expected_gauge_payload_sha256"]:
        raise ValueError("P8AG gauge payload drift")
    profile = native_gauge_profile_struct(payload)
    builds = [
        build_msvc_native_gauge_dll(
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
            raise ValueError("P8AG native gauge source drift")
    if builds[0]["dll_sha256"] != builds[1]["dll_sha256"]:
        raise RuntimeError("P8AG native gauge build drift")

    _, python_gauge = reconstruct_standalone_runtime(artifact)
    random = np.random.default_rng(int(config["oracle"]["seed"]))
    source = random.random(
        (1, int(config["oracle"]["rgb_count"]), 3),
        dtype=np.float32,
    )
    source[0, 0] = (0.0, 0.0, 0.0)
    source[0, 1] = (1.0, 1.0, 1.0)
    source[0, 2] = (0.0, 0.5, 1.0)
    expected = apply_gauge_to_intermediate(
        source.astype(np.float64), python_gauge
    ).astype(np.float32)
    replays = []
    for build in builds:
        library = _load_gauge(Path(build["dll_path"]))
        observed = np.empty_like(source)
        status = library.nf_neutral_gauge_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(source),
            source.shape[1],
            _pointer(observed),
        )
        if status != 0:
            raise RuntimeError(f"P8AG native gauge failed: {status}")
        inplace = source.copy()
        status = library.nf_neutral_gauge_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(inplace),
            source.shape[1],
            _pointer(inplace),
        )
        if status != 0 or inplace.tobytes() != observed.tobytes():
            raise RuntimeError("P8AG native gauge in-place drift")
        invalid = source.copy()
        invalid.reshape(-1)[0] = np.float32(np.nan)
        sentinel = np.full_like(source, np.float32(-13.0))
        before = sentinel.tobytes()
        status = library.nf_neutral_gauge_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(invalid),
            source.shape[1],
            _pointer(sentinel),
        )
        if status == 0 or sentinel.tobytes() != before:
            raise RuntimeError("P8AG invalid input changed output")
        replays.append(
            {
                "maximum_absolute_error_vs_python_float32": float(
                    np.max(
                        np.abs(
                            observed.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                ),
                "output_sha256": hashlib.sha256(
                    observed.tobytes()
                ).hexdigest(),
                "in_place_byte_exact": True,
                "invalid_input_output_unchanged": True,
            }
        )
    if replays[0] != replays[1]:
        raise RuntimeError("P8AG replay drift")
    tolerance = float(config["gates"]["maximum_absolute_error"])
    if replays[0][
        "maximum_absolute_error_vs_python_float32"
    ] > tolerance:
        raise RuntimeError("P8AG gauge tolerance failed")
    stable = {
        "schema": "neuro_film.u6_p8ag_native_neutral_gauge.v1",
        "gauge_payload_sha256": payload_sha,
        "knot_counts": [
            int(profile.knot_count[channel]) for channel in range(3)
        ],
        "dll_sha256": builds[0]["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "oracle": {
            "rgb_count": int(source.shape[1]),
            **replays[0],
            "tolerance": tolerance,
        },
        "domain_order": config["domain_order"],
        "double_counting_control": config["double_counting_control"],
        "decision": (
            "pass the native Standard post-scan neutral-axis gauge against "
            "the artifact-only Python float32 oracle; AO6 base/residual "
            "native execution and real decoder/encoder integration remain open"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AH native AO6 residual-only ABI after an externally "
            "provided source-context base, preserving the fixed t15/c35 "
            "analytical gamut guard"
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
