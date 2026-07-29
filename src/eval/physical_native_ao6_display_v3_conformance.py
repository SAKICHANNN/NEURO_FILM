"""Exact-output conformance for the single-compute AO6 display v3."""

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
from src.eval.physical_native_ao6_fastpath_conformance import (
    _apply_display,
    _load_context_v2,
    _load_display_v2,
    _native_context_v2,
    build_msvc_native_ao6_context_v2_dll,
    build_msvc_native_ao6_display_v2_dll,
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


def build_msvc_native_ao6_display_v3_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    source_relatives = [
        "native/film_physics/nf_ao6_base_f32_v3.c",
        "native/film_physics/nf_ao6_residual_f32_v1.c",
        "native/film_physics/nf_ao6_display_f32_v3.c",
    ]
    header_relatives = [
        "native/film_physics/nf_ao6_base_f32_v3.h",
        "native/film_physics/nf_ao6_display_f32_v3.h",
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
    dll = (output_dir / "nf_ao6_display_f32_v3.dll").resolve()
    import_library = (
        output_dir / "nf_ao6_display_f32_v3.lib"
    ).resolve()
    batch = output_dir / "build_nf_ao6_display_f32_v3.bat"
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
            "MSVC native AO6 display v3 build failed:\n" + compiler_output
        )
    if "warning" in compiler_output.lower():
        raise RuntimeError(
            "MSVC native AO6 display v3 build warned:\n" + compiler_output
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


def _load_display_v3(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_base_f32_apply_scratch_v3.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_base_f32_apply_scratch_v3.restype = ctypes.c_int
    library.nf_ao6_display_f32_apply_v3.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(NativeAo6ResidualProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_display_f32_apply_v3.restype = ctypes.c_int
    return library


def _apply_display_v3(
    *,
    library: ctypes.CDLL,
    base_profile: NativeAo6BaseProfileF32V1,
    context: NativeAo6BaseContextF32V1,
    residual_profile: NativeAo6ResidualProfileF32V1,
    encoded: np.ndarray,
    tile_rows: int,
) -> np.ndarray:
    height, width, _ = encoded.shape
    output = np.empty_like(encoded)
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        input_rows = np.ascontiguousarray(encoded[y0:y1].reshape(-1, 3))
        scratch = np.empty_like(input_rows)
        output_rows = np.empty_like(input_rows)
        status = library.nf_ao6_display_f32_apply_v3(
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
                f"P8AS display v3 tile {tile_rows} failed: {status}"
            )
        output[y0:y1] = output_rows.reshape(y1 - y0, width, 3)
    return output


def run_native_ao6_display_v3_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AS"):
        raise ValueError("P8AS parent decision drift")
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
        raise ValueError("P8AS display payload drift")
    base_profile = native_ao6_base_profile_struct(display)
    residual_profile = native_ao6_residual_profile_struct(display)

    context_build = build_msvc_native_ao6_context_v2_dll(
        root=root, output_dir=output_dir / "context_v2"
    )
    display_v2_build = build_msvc_native_ao6_display_v2_dll(
        root=root, output_dir=output_dir / "display_v2"
    )
    display_v3_builds = [
        build_msvc_native_ao6_display_v3_dll(
            root=root, output_dir=output_dir / f"display_v3_{index}"
        )
        for index in range(2)
    ]
    if (
        display_v3_builds[0]["dll_sha256"]
        != display_v3_builds[1]["dll_sha256"]
    ):
        raise RuntimeError("P8AS independent display v3 build drift")
    expected = config["native_sources"]
    for build in display_v3_builds:
        if (
            build["sources"] != expected["sources"]
            or build["headers"] != expected["headers"]
        ):
            raise ValueError("P8AS native source identity drift")

    height = int(config["oracle"]["height"])
    width = int(config["oracle"]["width"])
    random = np.random.default_rng(int(config["oracle"]["seed"]))
    source = random.random((height, width, 3), dtype=np.float32)
    encoded = random.random((height, width, 3), dtype=np.float32)
    source[0, :5] = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    encoded[0, :5] = source[0, :5]
    context_library = _load_context_v2(
        Path(context_build["dll_path"])
    )
    context, context_sha = _native_context_v2(
        library=context_library,
        profile=base_profile,
        source=source,
        tile_rows=7,
    )
    library_v2 = _load_display_v2(Path(display_v2_build["dll_path"]))
    library_v3 = _load_display_v3(
        Path(display_v3_builds[0]["dll_path"])
    )

    rows = []
    for tile_rows in config["tile_rows"]:
        output_v2 = _apply_display(
            library=library_v2,
            version=2,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=encoded,
            tile_rows=int(tile_rows),
        )
        output_v3 = _apply_display_v3(
            library=library_v3,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=encoded,
            tile_rows=int(tile_rows),
        )
        if output_v2.tobytes() != output_v3.tobytes():
            raise RuntimeError("P8AS display v2/v3 output drift")
        rows.append(
            {
                "tile_rows": int(tile_rows),
                "output_sha256": hashlib.sha256(
                    output_v3.tobytes()
                ).hexdigest(),
            }
        )
    if len({row["output_sha256"] for row in rows}) != 1:
        raise RuntimeError("P8AS display v3 partition drift")

    flat = np.ascontiguousarray(encoded.reshape(-1, 3))
    base_v2 = np.empty_like(flat)
    base_v3 = np.empty_like(flat)
    if library_v2.nf_ao6_base_f32_apply_v2(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        _pointer(flat),
        flat.shape[0],
        _pointer(base_v2),
    ) != 0:
        raise RuntimeError("P8AS base v2 failed")
    if library_v3.nf_ao6_base_f32_apply_scratch_v3(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        _pointer(flat),
        flat.shape[0],
        _pointer(base_v3),
    ) != 0 or base_v2.tobytes() != base_v3.tobytes():
        raise RuntimeError("P8AS base v2/v3 output drift")

    invalid = np.ascontiguousarray(encoded[:1].reshape(-1, 3))
    invalid[0, 0] = np.float32(np.nan)
    scratch = np.full_like(invalid, np.float32(-7.0))
    final = np.full_like(invalid, np.float32(-9.0))
    scratch_before = scratch.tobytes()
    final_before = final.tobytes()
    status = library_v3.nf_ao6_display_f32_apply_v3(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        ctypes.byref(residual_profile),
        _pointer(invalid),
        invalid.shape[0],
        _pointer(scratch),
        _pointer(final),
    )
    if (
        status == 0
        or scratch.tobytes() != scratch_before
        or final.tobytes() != final_before
    ):
        raise RuntimeError("P8AS invalid input changed scratch or final")

    stable = {
        "schema": "neuro_film.u6_p8as_native_ao6_display_v3.v1",
        "display_payload_sha256": display_sha,
        "context_v2_dll_sha256": context_build["dll_sha256"],
        "display_v2_dll_sha256": display_v2_build["dll_sha256"],
        "display_v3_dll_sha256": display_v3_builds[0]["dll_sha256"],
        "independent_display_v3_build_byte_exact": True,
        "context_sha256": context_sha,
        "base_v2_v3_byte_exact": True,
        "display_rows": rows,
        "display_v2_v3_byte_exact": True,
        "display_v3_partition_byte_exact": True,
        "invalid_input_scratch_unchanged": True,
        "invalid_input_final_output_unchanged": True,
        "optimization": [
            "single base pixel computation into caller-owned scratch",
            "single final OETF computation followed by exact scratch copy",
        ],
        "decision": (
            "pass AO6 display v3 under exact v2 output, partition, "
            "independent-build and invalid-input failure-atomic gates"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AT rerun the unchanged 12MP resource gate with context "
            "v2 and display v3; if latency still fails, re-profile"
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
