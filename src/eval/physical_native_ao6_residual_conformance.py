"""Conformance audit for the native fixed AO6 t15/c35 residual."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.density_witness_frontier import encoded_srgb_to_linear
from src.eval.native_msvc import build_msvc_c11_dll
from src.film_physics.native_ao6_residual_profile import (
    NativeAo6ResidualProfileF32V1,
    native_ao6_display_payload_sha256,
    native_ao6_residual_profile_struct,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.roll2film.positive_film import PositiveFilmResponseOperator


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


def build_msvc_native_ao6_residual_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_ao6_residual_f32_v1.c",
        header_relative="native/film_physics/nf_ao6_residual_f32_v1.h",
        basename="nf_ao6_residual_f32_v1",
    )


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_residual_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeAo6ResidualProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_residual_f32_apply_v1.restype = ctypes.c_int
    return library


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_native_ao6_residual_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AH"):
        raise ValueError("P8AH parent decision drift")
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
    display_sha = native_ao6_display_payload_sha256(display)
    if display_sha != config["expected_display_payload_sha256"]:
        raise ValueError("P8AH display payload drift")
    profile = native_ao6_residual_profile_struct(display)
    builds = [
        build_msvc_native_ao6_residual_dll(
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
            raise ValueError("P8AH source identity drift")
    if builds[0]["dll_sha256"] != builds[1]["dll_sha256"]:
        raise RuntimeError("P8AH native build drift")

    random = np.random.default_rng(int(config["oracle"]["seed"]))
    encoded = random.random(
        (int(config["oracle"]["rgb_count"]), 3),
        dtype=np.float32,
    )
    encoded[0] = (0.0, 0.0, 0.0)
    encoded[1] = (1.0, 1.0, 1.0)
    encoded[2] = (0.0, 0.5, 1.0)
    encoded[3] = (1.0 / 65535.0, 0.25, 1.0 - 1.0 / 65535.0)
    source = np.ascontiguousarray(
        encoded_srgb_to_linear(
            encoded.astype(np.float64)[None, ...]
        )[0],
        dtype=np.float32,
    )
    residual = display["residual"]
    expected = apply_factorized_boundary_guard(
        PositiveFilmResponseOperator.from_dict(residual["operator"]),
        source.astype(np.float64),
        tone_strength=float(residual["tone_strength"]),
        chroma_strength=float(residual["chroma_strength"]),
        luma_weights=np.asarray(
            residual["luma_weights"], dtype=np.float64
        ),
        hard_boundary_epsilon_encoded_srgb=float(
            residual["hard_boundary_epsilon_encoded_srgb"]
        ),
        guard_boundary_epsilon_encoded_srgb=float(
            residual["guard_boundary_epsilon_encoded_srgb"]
        ),
    )
    expected_output = expected.output.astype(np.float32)
    expected_tone = expected.tone_scale.astype(np.float32)
    expected_chroma = expected.chroma_scale.astype(np.float32)
    replays = []
    for build in builds:
        library = _load(Path(build["dll_path"]))
        output = np.empty_like(source)
        tone = np.empty(source.shape[0], dtype=np.float32)
        chroma = np.empty(source.shape[0], dtype=np.float32)
        status = library.nf_ao6_residual_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(source),
            source.shape[0],
            _pointer(output),
            _pointer(tone),
            _pointer(chroma),
        )
        if status != 0:
            raise RuntimeError(f"P8AH native residual failed: {status}")
        inplace = source.copy()
        status = library.nf_ao6_residual_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(inplace),
            source.shape[0],
            _pointer(inplace),
            None,
            None,
        )
        if status != 0 or inplace.tobytes() != output.tobytes():
            raise RuntimeError("P8AH in-place or null-diagnostic drift")
        invalid = source.copy()
        invalid.reshape(-1)[0] = np.float32(np.nan)
        sentinel = np.full_like(source, np.float32(-9.0))
        before = sentinel.tobytes()
        status = library.nf_ao6_residual_f32_apply_v1(
            ctypes.byref(profile),
            _pointer(invalid),
            source.shape[0],
            _pointer(sentinel),
            None,
            None,
        )
        if status == 0 or sentinel.tobytes() != before:
            raise RuntimeError("P8AH invalid input changed output")
        replays.append(
            {
                "maximum_output_absolute_error": float(
                    np.max(
                        np.abs(
                            output.astype(np.float64)
                            - expected_output.astype(np.float64)
                        )
                    )
                ),
                "maximum_tone_scale_absolute_error": float(
                    np.max(
                        np.abs(
                            tone.astype(np.float64)
                            - expected_tone.astype(np.float64)
                        )
                    )
                ),
                "maximum_chroma_scale_absolute_error": float(
                    np.max(
                        np.abs(
                            chroma.astype(np.float64)
                            - expected_chroma.astype(np.float64)
                        )
                    )
                ),
                "output_sha256": hashlib.sha256(
                    output.tobytes()
                ).hexdigest(),
                "tone_scale_sha256": hashlib.sha256(
                    tone.tobytes()
                ).hexdigest(),
                "chroma_scale_sha256": hashlib.sha256(
                    chroma.tobytes()
                ).hexdigest(),
                "in_place_byte_exact": True,
                "invalid_input_output_unchanged": True,
                "minimum_output": float(np.min(output)),
                "maximum_output": float(np.max(output)),
            }
        )
    if replays[0] != replays[1]:
        raise RuntimeError("P8AH independent replay drift")
    tolerance = float(config["gates"]["maximum_absolute_error"])
    worst = max(
        replays[0]["maximum_output_absolute_error"],
        replays[0]["maximum_tone_scale_absolute_error"],
        replays[0]["maximum_chroma_scale_absolute_error"],
    )
    if worst > tolerance:
        raise RuntimeError(
            f"P8AH native residual error {worst} exceeds {tolerance}"
        )
    if (
        replays[0]["minimum_output"] < 0.0
        or replays[0]["maximum_output"] > 1.0
    ):
        raise RuntimeError("P8AH output escaped relative linear RGB")
    stable = {
        "schema": "neuro_film.u6_p8ah_native_ao6_residual.v1",
        "display_payload_sha256": display_sha,
        "dll_sha256": builds[0]["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "tone_strength": float(profile.tone_strength),
        "chroma_strength": float(profile.chroma_strength),
        "oracle": {
            "rgb_count": int(source.shape[0]),
            **replays[0],
            "tolerance": tolerance,
        },
        "input_boundary": (
            "externally prepared AO6 source-context base in relative "
            "linear-sRGB; this ABI does not compute or refit source context"
        ),
        "decision": (
            "pass the native fixed AO6 t15/c35 factorized residual and "
            "analytical source-inclusive gamut guard; the source-context "
            "safe-Lab base remains an independent explicit stage"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AI compose the native float32 physical chain, neutral "
            "gauge and AO6 residual around a frozen external base fixture; "
            "prove ordered row-stream parity and no-double-count receipts"
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
