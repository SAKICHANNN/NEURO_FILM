"""Conformance audit for the complete native AO6 display-look execution."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.physical_native_ao6_base_conformance import _pointer
from src.eval.physical_native_ao6_context_conformance import (
    _load_context,
    _native_context,
    build_msvc_native_ao6_context_dll,
)
from src.film_physics.display_look import (
    build_source_context_display_look,
)
from src.film_physics.native_ao6_base_profile import (
    NativeAo6BaseContextF32V1,
    NativeAo6BaseProfileF32V1,
    native_ao6_base_display_payload_sha256,
    native_ao6_base_profile_struct,
)
from src.film_physics.native_ao6_residual_profile import (
    NativeAo6ResidualProfileF32V1,
    native_ao6_residual_profile_struct,
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


def build_msvc_native_ao6_display_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    source_relatives = [
        "native/film_physics/nf_ao6_base_f32_v1.c",
        "native/film_physics/nf_ao6_residual_f32_v1.c",
        "native/film_physics/nf_ao6_display_f32_v1.c",
    ]
    header_relatives = [
        "native/film_physics/nf_ao6_base_f32_v1.h",
        "native/film_physics/nf_ao6_residual_f32_v1.h",
        "native/film_physics/nf_ao6_display_f32_v1.h",
    ]
    sources = [(root / relative).resolve() for relative in source_relatives]
    headers = [(root / relative).resolve() for relative in header_relatives]
    if (
        not vcvars.is_file()
        or not all(path.is_file() for path in sources)
        or not all(path.is_file() for path in headers)
    ):
        raise RuntimeError("native source or MSVC environment is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_ao6_display_f32_v1.dll").resolve()
    import_library = (
        output_dir / "nf_ao6_display_f32_v1.lib"
    ).resolve()
    batch = output_dir / "build_nf_ao6_display_f32_v1.bat"
    quoted_sources = " ".join(f'"{path}"' for path in sources)
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f"cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD "
        f"{quoted_sources} /link /Brepro /OUT:\"{dll}\" "
        f'/IMPLIB:"{import_library}"\r\n',
        encoding="ascii",
        newline="",
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(batch.resolve())],
        capture_output=True,
        timeout=120,
        check=False,
        cwd=output_dir,
    )
    compiler_output = (
        completed.stdout + completed.stderr
    ).decode("utf-8", errors="replace")
    if completed.returncode != 0 or not dll.is_file():
        raise RuntimeError(
            "MSVC native display build failed:\n" + compiler_output
        )
    return {
        "toolchain": "msvc-x64-c11",
        "sources": {
            relative: sha256_file(path)
            for relative, path in zip(source_relatives, sources)
        },
        "headers": {
            relative: sha256_file(path)
            for relative, path in zip(header_relatives, headers)
        },
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
        "compiler_output": compiler_output.strip(),
    }


def _load_display(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_display_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(NativeAo6ResidualProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_display_f32_apply_v1.restype = ctypes.c_int
    return library


def run_native_ao6_display_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AL"):
        raise ValueError("P8AL parent decision drift")
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
        raise ValueError("P8AL display payload drift")
    base_profile = native_ao6_base_profile_struct(display)
    residual_profile = native_ao6_residual_profile_struct(display)

    display_builds = [
        build_msvc_native_ao6_display_dll(
            root=root, output_dir=output_dir / f"display_{index}"
        )
        for index in range(2)
    ]
    expected_sources = config["native_sources"]
    for build in display_builds:
        if (
            build["sources"] != expected_sources["sources"]
            or build["headers"] != expected_sources["headers"]
        ):
            raise ValueError("P8AL native source identity drift")
    if display_builds[0]["dll_sha256"] != display_builds[1]["dll_sha256"]:
        raise RuntimeError("P8AL independent display build drift")
    context_build = build_msvc_native_ao6_context_dll(
        root=root, output_dir=output_dir / "context"
    )

    height = int(config["oracle"]["height"])
    width = int(config["oracle"]["width"])
    random = np.random.default_rng(int(config["oracle"]["seed"]))
    source = random.random((height, width, 3), dtype=np.float32)
    encoded = random.random((height, width, 3), dtype=np.float32)
    source[0, 0] = (0.0, 0.0, 0.0)
    source[0, 1] = (1.0, 1.0, 1.0)
    encoded[0, 0] = (0.0, 0.5, 1.0)
    encoded[0, 1] = (1.0, 0.0, 0.75)
    python_output = np.asarray(
        build_source_context_display_look(display, source)(encoded),
        dtype=np.float32,
    )

    context_library = _load_context(Path(context_build["dll_path"]))
    context, context_sha = _native_context(
        library=context_library,
        profile=base_profile,
        source=source,
        tile_rows=7,
    )
    library = _load_display(Path(display_builds[0]["dll_path"]))
    partition_rows = []
    partition_outputs = []
    for tile_rows in config["tile_rows"]:
        output = np.empty_like(encoded)
        for y0 in range(0, height, int(tile_rows)):
            y1 = min(height, y0 + int(tile_rows))
            input_rows = np.ascontiguousarray(
                encoded[y0:y1].reshape(-1, 3)
            )
            scratch = np.empty_like(input_rows)
            output_rows = np.ascontiguousarray(
                output[y0:y1].reshape(-1, 3)
            )
            status = library.nf_ao6_display_f32_apply_v1(
                ctypes.byref(base_profile),
                ctypes.byref(context),
                ctypes.byref(residual_profile),
                _pointer(input_rows),
                input_rows.shape[0],
                _pointer(scratch),
                _pointer(output_rows),
            )
            if status != 0:
                raise RuntimeError(
                    f"P8AL display tile {tile_rows} failed: {status}"
                )
            output[y0:y1] = output_rows.reshape(
                y1 - y0, width, 3
            )
        output_sha = hashlib.sha256(output.tobytes()).hexdigest()
        partition_outputs.append(output)
        partition_rows.append(
            {
                "tile_rows": int(tile_rows),
                "output_sha256": output_sha,
            }
        )
    if len({row["output_sha256"] for row in partition_rows}) != 1:
        raise RuntimeError("P8AL partition output drift")
    native_output = partition_outputs[0]
    difference = np.abs(
        native_output.astype(np.float64) -
        python_output.astype(np.float64)
    )
    maximum_error = float(np.max(difference))
    if maximum_error > float(config["gates"]["maximum_output_error"]):
        raise RuntimeError(
            f"P8AL output error exceeds gate: {maximum_error}"
        )

    invalid = encoded[:1].copy().reshape(-1, 3)
    invalid[0, 0] = np.float32(np.nan)
    scratch = np.empty_like(invalid)
    sentinel = np.full_like(invalid, np.float32(-9.0))
    before = sentinel.tobytes()
    status = library.nf_ao6_display_f32_apply_v1(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        ctypes.byref(residual_profile),
        _pointer(np.ascontiguousarray(invalid)),
        invalid.shape[0],
        _pointer(scratch),
        _pointer(sentinel),
    )
    invalid_unchanged = status != 0 and sentinel.tobytes() == before
    if not invalid_unchanged:
        raise RuntimeError("P8AL invalid input changed final output")

    stable = {
        "schema": "neuro_film.u6_p8al_native_ao6_display.v1",
        "display_payload_sha256": display_sha,
        "display_dll_sha256": display_builds[0]["dll_sha256"],
        "context_dll_sha256": context_build["dll_sha256"],
        "context_sha256": context_sha,
        "independent_display_build_dll_sha_exact": True,
        "partition_rows": partition_rows,
        "partition_output_byte_exact": True,
        "invalid_input_final_output_unchanged": True,
        "oracle": {
            "shape": [height, width, 3],
            "maximum_output_error": maximum_error,
            "p999_output_error": float(np.quantile(difference, 0.999)),
            "native_output_sha256": hashlib.sha256(
                native_output.tobytes()
            ).hexdigest(),
            "python_output_sha256": hashlib.sha256(
                python_output.tobytes()
            ).hexdigest(),
            "minimum_output": float(np.min(native_output)),
            "maximum_output": float(np.max(native_output)),
        },
        "ordered_stages": [
            "native density-plus-source-context base",
            "native encoded-sRGB-to-linear-sRGB transfer",
            "native fixed t15/c35 residual and analytical guard",
            "native linear-sRGB-to-encoded-sRGB transfer",
        ],
        "decision": (
            "pass one complete row-streamed native AO6 display-look "
            "execution using the native source context; output is partition "
            "exact and remains inside the frozen Python-reference gate"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AM compose the native Standard physical chain, neutral "
            "gauge and complete native AO6 display look end to end; measure "
            "partition parity, no-double-count receipts and resources"
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
