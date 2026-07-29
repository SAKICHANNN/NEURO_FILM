"""Exact-output conformance for versioned native AO6 fast paths."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from src.eval.native_msvc import (
    build_msvc_c11_dll,
    find_msvc_installation,
    sha256_file,
)
from src.eval.physical_native_ao6_base_conformance import _pointer
from src.eval.physical_native_ao6_context_conformance import (
    _load_context,
    _native_context,
    build_msvc_native_ao6_context_dll,
)
from src.eval.physical_native_ao6_display_conformance import (
    _load_display,
    build_msvc_native_ao6_display_dll,
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


def build_msvc_native_ao6_context_v2_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_ao6_context_f32_v2.c",
        header_relative="native/film_physics/nf_ao6_context_f32_v2.h",
        basename="nf_ao6_context_f32_v2",
    )


def build_msvc_native_ao6_display_v2_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    source_relatives = [
        "native/film_physics/nf_ao6_base_f32_v2.c",
        "native/film_physics/nf_ao6_residual_f32_v1.c",
        "native/film_physics/nf_ao6_display_f32_v2.c",
    ]
    header_relatives = [
        "native/film_physics/nf_ao6_base_f32_v2.h",
        "native/film_physics/nf_ao6_context_f32_v2.h",
        "native/film_physics/nf_ao6_display_f32_v2.h",
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
    dll = (output_dir / "nf_ao6_display_f32_v2.dll").resolve()
    import_library = (
        output_dir / "nf_ao6_display_f32_v2.lib"
    ).resolve()
    batch = output_dir / "build_nf_ao6_display_f32_v2.bat"
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
            "MSVC native AO6 v2 display build failed:\n" + compiler_output
        )
    if "warning" in compiler_output.lower():
        raise RuntimeError(
            "MSVC native AO6 v2 display build warned:\n" + compiler_output
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


def _load_context_v2(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_context_f32_init_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1)
    ]
    library.nf_ao6_context_f32_init_v1.restype = ctypes.c_int
    library.nf_ao6_context_f32_update_v2.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_context_f32_update_v2.restype = ctypes.c_int
    library.nf_ao6_context_f32_finalize_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
    ]
    library.nf_ao6_context_f32_finalize_v1.restype = ctypes.c_int
    return library


def _load_display_v2(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_base_f32_apply_v2.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_base_f32_apply_v2.restype = ctypes.c_int
    library.nf_ao6_display_f32_apply_v2.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(NativeAo6ResidualProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_display_f32_apply_v2.restype = ctypes.c_int
    return library


def _native_context_v2(
    *,
    library: ctypes.CDLL,
    profile: NativeAo6BaseProfileF32V1,
    source: np.ndarray,
    tile_rows: int,
) -> tuple[NativeAo6BaseContextF32V1, str]:
    state = NativeAo6ContextStateF32V1()
    if library.nf_ao6_context_f32_init_v1(ctypes.byref(state)) != 0:
        raise RuntimeError("P8AP context init failed")
    height, width, _ = source.shape
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        rows = np.ascontiguousarray(source[y0:y1].reshape(-1, 3))
        scratch = np.empty_like(rows)
        status = library.nf_ao6_context_f32_update_v2(
            ctypes.byref(profile),
            ctypes.byref(state),
            _pointer(rows),
            rows.shape[0],
            _pointer(scratch),
        )
        if status != 0:
            raise RuntimeError(
                f"P8AP context v2 update {tile_rows} failed: {status}"
            )
    context = NativeAo6BaseContextF32V1()
    if library.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise RuntimeError("P8AP context v2 finalize failed")
    return context, hashlib.sha256(bytes(context)).hexdigest()


def _apply_display(
    *,
    library: ctypes.CDLL,
    version: int,
    base_profile: NativeAo6BaseProfileF32V1,
    context: NativeAo6BaseContextF32V1,
    residual_profile: NativeAo6ResidualProfileF32V1,
    encoded: np.ndarray,
    tile_rows: int,
) -> np.ndarray:
    height, width, _ = encoded.shape
    output = np.empty_like(encoded)
    function = getattr(library, f"nf_ao6_display_f32_apply_v{version}")
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        input_rows = np.ascontiguousarray(encoded[y0:y1].reshape(-1, 3))
        scratch = np.empty_like(input_rows)
        output_rows = np.empty_like(input_rows)
        status = function(
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
                f"P8AP display v{version} tile {tile_rows} failed: {status}"
            )
        output[y0:y1] = output_rows.reshape(y1 - y0, width, 3)
    return output


def run_native_ao6_fastpath_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AP"):
        raise ValueError("P8AP parent decision drift")
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
        raise ValueError("P8AP display payload drift")
    base_profile = native_ao6_base_profile_struct(display)
    residual_profile = native_ao6_residual_profile_struct(display)

    context_v1_build = build_msvc_native_ao6_context_dll(
        root=root, output_dir=output_dir / "context_v1"
    )
    context_v2_builds = [
        build_msvc_native_ao6_context_v2_dll(
            root=root, output_dir=output_dir / f"context_v2_{index}"
        )
        for index in range(2)
    ]
    display_v1_build = build_msvc_native_ao6_display_dll(
        root=root, output_dir=output_dir / "display_v1"
    )
    display_v2_builds = [
        build_msvc_native_ao6_display_v2_dll(
            root=root, output_dir=output_dir / f"display_v2_{index}"
        )
        for index in range(2)
    ]
    if (
        context_v2_builds[0]["dll_sha256"]
        != context_v2_builds[1]["dll_sha256"]
        or display_v2_builds[0]["dll_sha256"]
        != display_v2_builds[1]["dll_sha256"]
    ):
        raise RuntimeError("P8AP independent build drift")
    expected_sources = config["native_sources"]
    for build in context_v2_builds:
        if (
            build["source_sha256"]
            != expected_sources["context"]["source_sha256"]
            or build["header_sha256"]
            != expected_sources["context"]["header_sha256"]
        ):
            raise ValueError("P8AP context source identity drift")
    for build in display_v2_builds:
        if (
            build["sources"] != expected_sources["display"]["sources"]
            or build["headers"] != expected_sources["display"]["headers"]
        ):
            raise ValueError("P8AP display source identity drift")

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

    context_v1_library = _load_context(
        Path(context_v1_build["dll_path"])
    )
    context_v2_library = _load_context_v2(
        Path(context_v2_builds[0]["dll_path"])
    )
    context_rows = []
    contexts_v2 = []
    for tile_rows in config["tile_rows"]:
        context_v1, sha_v1 = _native_context(
            library=context_v1_library,
            profile=base_profile,
            source=source,
            tile_rows=int(tile_rows),
        )
        context_v2, sha_v2 = _native_context_v2(
            library=context_v2_library,
            profile=base_profile,
            source=source,
            tile_rows=int(tile_rows),
        )
        if bytes(context_v1) != bytes(context_v2):
            raise RuntimeError("P8AP context v1/v2 drift")
        contexts_v2.append(context_v2)
        context_rows.append(
            {
                "tile_rows": int(tile_rows),
                "v1_sha256": sha_v1,
                "v2_sha256": sha_v2,
            }
        )
    if len({row["v2_sha256"] for row in context_rows}) != 1:
        raise RuntimeError("P8AP context partition drift")
    context = contexts_v2[0]

    display_v1_library = _load_display(
        Path(display_v1_build["dll_path"])
    )
    display_v2_library = _load_display_v2(
        Path(display_v2_builds[0]["dll_path"])
    )
    base_input = np.ascontiguousarray(encoded.reshape(-1, 3))
    base_v1 = np.empty_like(base_input)
    base_v2 = np.empty_like(base_input)
    if display_v1_library.nf_ao6_base_f32_apply_v1(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        _pointer(base_input),
        base_input.shape[0],
        _pointer(base_v1),
    ) != 0:
        raise RuntimeError("P8AP base v1 failed")
    if display_v2_library.nf_ao6_base_f32_apply_v2(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        _pointer(base_input),
        base_input.shape[0],
        _pointer(base_v2),
    ) != 0:
        raise RuntimeError("P8AP base v2 failed")
    if base_v1.tobytes() != base_v2.tobytes():
        raise RuntimeError("P8AP base v1/v2 output drift")
    inplace = base_input.copy()
    if display_v2_library.nf_ao6_base_f32_apply_v2(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        _pointer(inplace),
        inplace.shape[0],
        _pointer(inplace),
    ) != 0 or inplace.tobytes() != base_v2.tobytes():
        raise RuntimeError("P8AP base v2 in-place drift")

    display_rows = []
    for tile_rows in config["tile_rows"]:
        output_v1 = _apply_display(
            library=display_v1_library,
            version=1,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=encoded,
            tile_rows=int(tile_rows),
        )
        output_v2 = _apply_display(
            library=display_v2_library,
            version=2,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=encoded,
            tile_rows=int(tile_rows),
        )
        if output_v1.tobytes() != output_v2.tobytes():
            raise RuntimeError("P8AP display v1/v2 output drift")
        display_rows.append(
            {
                "tile_rows": int(tile_rows),
                "output_sha256": hashlib.sha256(
                    output_v2.tobytes()
                ).hexdigest(),
            }
        )
    if len({row["output_sha256"] for row in display_rows}) != 1:
        raise RuntimeError("P8AP display partition drift")

    invalid = np.ascontiguousarray(encoded[:1].reshape(-1, 3))
    invalid[0, 0] = np.float32(np.nan)
    scratch = np.empty_like(invalid)
    sentinel = np.full_like(invalid, np.float32(-9.0))
    before = sentinel.tobytes()
    status = display_v2_library.nf_ao6_display_f32_apply_v2(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        ctypes.byref(residual_profile),
        _pointer(invalid),
        invalid.shape[0],
        _pointer(scratch),
        _pointer(sentinel),
    )
    if status == 0 or sentinel.tobytes() != before:
        raise RuntimeError("P8AP invalid display input changed final output")

    state = NativeAo6ContextStateF32V1()
    context_v2_library.nf_ao6_context_f32_init_v1(ctypes.byref(state))
    state_before = bytes(state)
    context_scratch = np.full_like(invalid, np.float32(-7.0))
    status = context_v2_library.nf_ao6_context_f32_update_v2(
        ctypes.byref(base_profile),
        ctypes.byref(state),
        _pointer(invalid),
        invalid.shape[0],
        _pointer(context_scratch),
    )
    if status == 0 or bytes(state) != state_before:
        raise RuntimeError("P8AP invalid context update changed state")

    stable = {
        "schema": "neuro_film.u6_p8ap_native_ao6_fastpath.v1",
        "display_payload_sha256": display_sha,
        "context_v2_dll_sha256": context_v2_builds[0]["dll_sha256"],
        "display_v2_dll_sha256": display_v2_builds[0]["dll_sha256"],
        "independent_context_build_byte_exact": True,
        "independent_display_build_byte_exact": True,
        "context_rows": context_rows,
        "context_v1_v2_byte_exact": True,
        "context_partition_byte_exact": True,
        "base_v1_v2_byte_exact": True,
        "base_v2_in_place_byte_exact": True,
        "display_rows": display_rows,
        "display_v1_v2_byte_exact": True,
        "display_partition_byte_exact": True,
        "invalid_context_state_unchanged": True,
        "invalid_display_final_output_unchanged": True,
        "oracle": {
            "shape": [height, width, 3],
            "base_output_sha256": hashlib.sha256(
                base_v2.tobytes()
            ).hexdigest(),
            "display_output_sha256": display_rows[0]["output_sha256"],
        },
        "optimization": [
            (
                "analytically resolve the exact legacy 14-step low endpoint "
                "when the full-chroma target is already in gamut"
            ),
            (
                "compute density plus Lab once into caller-owned scratch "
                "before mutating the streaming context state"
            ),
        ],
        "decision": (
            "pass versioned AO6 v2 fast paths under byte-exact v1 output, "
            "partition, in-place, independent-build and failure-atomic gates"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AQ rerun the frozen 12MP native Standard physical plus "
            "AO6 resource gates using only the exact-output v2 fast paths"
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
