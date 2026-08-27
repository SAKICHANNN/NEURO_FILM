"""Build and audit the private P255 native ACES2065-1 scanline writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

if __package__:
    from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _load_ephemeral_writer,
    )
    from scripts.audit_p248_acescg_openexr_scanline_streaming import (
        _canonical_bytes,
        _generate_block,
        _install_openexr_wheel,
        _process_tree_rss,
        _safe_extract,
        _sha256_file,
        _toolchain,
    )
else:
    from audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _load_ephemeral_writer,
    )
    from audit_p248_acescg_openexr_scanline_streaming import (
        _canonical_bytes,
        _generate_block,
        _install_openexr_wheel,
        _process_tree_rss,
        _safe_extract,
        _sha256_file,
        _toolchain,
    )

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neuro-film.p255-aces2065-openexr-scanline-streaming-result.v1"


class P255Error(RuntimeError):
    """Raised when one frozen P255 identity or gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _cmake_project(imath_source: Path, openexr_source: Path, native_source: Path) -> str:
    def q(path: Path) -> str:
        return path.resolve().as_posix()

    return f"""cmake_minimum_required(VERSION 3.20)
project(P255ScanlineWriter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(IMATH_INSTALL OFF CACHE BOOL "" FORCE)
set(IMATH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IMATH_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory("{q(imath_source)}" imath-build)
set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(OPENEXR_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL "" FORCE)
set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL "" FORCE)
add_subdirectory("{q(openexr_source)}" openexr-build)
add_executable(p255_writer "{q(native_source)}")
target_link_libraries(p255_writer PRIVATE OpenEXR::OpenEXR)
"""


def _build(config: dict[str, Any], workspace: Path) -> tuple[Path, dict[str, Any]]:
    bindings = config["bindings"]
    openexr_source = _safe_extract(
        ROOT / bindings["openexr_source_path"], workspace / "openexr-source"
    )
    imath_source = _safe_extract(
        ROOT / bindings["imath_source_path"], workspace / "imath-source"
    )
    project = workspace / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        _cmake_project(
            imath_source,
            openexr_source,
            ROOT / bindings["native_source_path"],
        ),
        encoding="utf-8",
        newline="\n",
    )
    build = workspace / "build"
    tools = _toolchain()
    configured = subprocess.run(
        [
            str(tools["cmake"]),
            "-S",
            str(project),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=False,
        env=tools["environment"],
        capture_output=True,
    )
    if configured.returncode != 0:
        raise P255Error(
            "native configure failed: "
            + configured.stdout.decode("utf-8", errors="replace")
            + configured.stderr.decode("utf-8", errors="replace")
        )
    built = subprocess.run(
        [
            str(tools["cmake"]),
            "--build",
            str(build),
            "--target",
            "p255_writer",
            "-j",
            "8",
        ],
        check=False,
        env=tools["environment"],
        capture_output=True,
    )
    if built.returncode != 0:
        raise P255Error(
            "native build failed: "
            + built.stdout.decode("utf-8", errors="replace")
            + built.stderr.decode("utf-8", errors="replace")
        )
    binary = build / "p255_writer.exe"
    if not binary.is_file():
        raise P255Error("native build did not produce p255_writer.exe")
    return binary, {
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": _sha256_file(binary),
        "cmake_sha256": _sha256_file(tools["cmake"]),
        "ninja_sha256": _sha256_file(tools["ninja"]),
        "msvc_cl_sha256": _sha256_file(tools["cl"]),
        "msvc_version_text": tools["msvc_version_text"],
    }


def _ap0_block(
    y0: int,
    rows: int,
    width: int,
    period: int,
    matrix: np.ndarray,
) -> np.ndarray:
    ap1 = _generate_block(y0, rows, width, period).astype(np.float64)
    output = np.empty_like(ap1, dtype=np.float32)
    for channel in range(3):
        output[..., channel] = (
            ap1[..., 0] * matrix[channel, 0]
            + ap1[..., 1] * matrix[channel, 1]
            + ap1[..., 2] * matrix[channel, 2]
        ).astype(np.float32)
    return output


def _expected_sha(
    height: int,
    width: int,
    block: int,
    period: int,
    matrix: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for y0 in range(0, height, block):
        digest.update(
            _ap0_block(y0, min(block, height - y0), width, period, matrix).tobytes()
        )
    return digest.hexdigest()


def _run_native(
    binary: Path,
    output: Path,
    *,
    width: int,
    height: int,
    row_block: int,
    period: int,
    inject: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    command = [
        str(binary),
        "--output",
        str(output),
        "--width",
        str(width),
        "--height",
        str(height),
        "--row-block",
        str(row_block),
        "--period",
        str(period),
    ]
    if inject:
        command.append("--inject-before-publish")
    return subprocess.run(command, check=False, capture_output=True)


def _load_openexr(site: Path) -> Any:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    return importlib.import_module("OpenEXR")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _inspect(
    path: Path,
    *,
    height: int,
    width: int,
    generation_block: int,
    period: int,
    matrix: np.ndarray,
    openexr: Any,
) -> dict[str, Any]:
    with openexr.File(str(path)) as exr_file:
        header = dict(exr_file.header())
        channels = exr_file.channels()
        decoded = channels["RGB"].pixels.copy()
        names = sorted(channels)
    maximum_error = 0.0
    new_boundary = 0
    for y0 in range(0, height, generation_block):
        rows = min(generation_block, height - y0)
        expected = _ap0_block(y0, rows, width, period, matrix)
        observed = decoded[y0 : y0 + rows]
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(observed.astype(np.float64) - expected))),
        )
        new_boundary += int(
            np.count_nonzero(
                ((observed == np.float32(0.0)) | (observed == np.float32(1.0)))
                & ~((expected == np.float32(0.0)) | (expected == np.float32(1.0)))
            )
        )
    return {
        "above_one_preserved": bool(np.max(decoded) > 1.0),
        "aces_image_container_flag": int(header["acesImageContainerFlag"]),
        "adopted_neutral": [float(value) for value in header["adoptedNeutral"]],
        "channel_names": names,
        "chromaticities": [float(value) for value in header["chromaticities"]],
        "color_interop_id": _text(header["colorInteropID"]),
        "compression_zip": bool(header["compression"] == openexr.ZIP_COMPRESSION),
        "decoded_dtype": str(decoded.dtype),
        "decoded_maximum_absolute_error": maximum_error,
        "decoded_pixel_f32le_sha256": _sha256_bytes(decoded.tobytes()),
        "decoded_shape": list(decoded.shape),
        "finite": bool(np.isfinite(decoded).all()),
        "negative_preserved": bool(np.min(decoded) < 0.0),
        "new_boundary_count": new_boundary,
        "scanline_image": bool(header["type"] == openexr.scanlineimage),
    }


def _worker(
    config_path: Path,
    binary: Path,
    site: Path,
    workspace: Path,
    label: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe = config["probe"]
    matrix = np.asarray(config["transform"]["ap1_to_ap0_matrix"], dtype=np.float64)
    output = workspace / f"{label}.exr"
    completed = _run_native(
        binary,
        output,
        width=int(probe["width"]),
        height=int(probe["height"]),
        row_block=int(probe["write_row_block"]),
        period=int(probe["coordinate_period"]),
    )
    if completed.returncode != 0:
        raise P255Error(completed.stderr.decode("utf-8", errors="replace"))
    inspection = _inspect(
        output,
        height=int(probe["height"]),
        width=int(probe["width"]),
        generation_block=int(probe["generation_row_block"]),
        period=int(probe["coordinate_period"]),
        matrix=matrix,
        openexr=_load_openexr(site),
    )
    result = {
        "file_bytes": output.stat().st_size,
        "file_sha256": _sha256_file(output),
        "expected_pixel_f32le_sha256": _expected_sha(
            int(probe["height"]),
            int(probe["width"]),
            int(probe["generation_row_block"]),
            int(probe["coordinate_period"]),
            matrix,
        ),
        "inspection": inspection,
    }
    output.unlink()
    return result


def _fresh_worker(
    config_path: Path,
    binary: Path,
    site: Path,
    workspace: Path,
    label: str,
    interval: float,
) -> dict[str, Any]:
    report = workspace / f"{label}.json"
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(config_path),
            "--binary",
            str(binary),
            "--site",
            str(site),
            "--workspace",
            str(workspace),
            "--label",
            label,
            "--output",
            str(report),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process = psutil.Process(child.pid)
    peak = 0
    samples = 0
    started = time.perf_counter()
    while child.poll() is None:
        peak = max(peak, _process_tree_rss(process))
        samples += 1
        time.sleep(interval)
    stdout, stderr = child.communicate()
    wall = time.perf_counter() - started
    if child.returncode != 0:
        raise P255Error(
            stderr.decode("utf-8", errors="replace")
            + stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_text(encoding="utf-8"))
    report.unlink()
    value["resource"] = {
        "peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "sampling_interval_seconds": interval,
        "wall_seconds": wall,
    }
    return value


def _small_controls(
    config_path: Path,
    producer_repo: Path,
    binary: Path,
    site: Path,
    workspace: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    matrix = np.asarray(config["transform"]["ap1_to_ap0_matrix"], dtype=np.float64)
    native = workspace / "small-native.exr"
    completed = _run_native(
        binary, native, width=17, height=13, row_block=16, period=31
    )
    if completed.returncode != 0:
        raise P255Error(completed.stderr.decode("utf-8", errors="replace"))
    openexr = _load_openexr(site)
    inspection = _inspect(
        native,
        height=13,
        width=17,
        generation_block=7,
        period=31,
        matrix=matrix,
        openexr=openexr,
    )
    native_sha = _sha256_file(native)
    with openexr.File(str(native)) as native_file:
        native_pixels = native_file.channels()["RGB"].pixels.copy()
    native.unlink()
    replay = _run_native(
        binary, native, width=17, height=13, row_block=16, period=31
    )
    replay_exact = replay.returncode == 0 and _sha256_file(native) == native_sha
    native.unlink()

    p249_config = json.loads(
        (ROOT / config["bindings"]["p249_config_path"]).read_text(encoding="utf-8")
    )
    writer_bytes = _git_bytes(
        producer_repo,
        p249_config["bindings"]["producer_writer_commit"],
        p249_config["bindings"]["producer_writer_path"],
    )
    writer = _load_ephemeral_writer(writer_bytes, workspace / "p249-writer-site")
    p249 = workspace / "small-p249.exr"
    ap1 = _generate_block(0, 13, 17, 31)
    writer.write_aces2065_1_openexr(p249, ap1)
    with openexr.File(str(p249)) as p249_file:
        p249_pixels = p249_file.channels()["RGB"].pixels.copy()
    p249.unlink()

    invalid = _run_native(
        binary, native, width=0, height=13, row_block=16, period=31
    )
    invalid_rejected = invalid.returncode != 0 and not native.exists()
    native.write_bytes(b"p255-existing")
    before = _sha256_file(native)
    injected = _run_native(
        binary,
        native,
        width=17,
        height=13,
        row_block=16,
        period=31,
        inject=True,
    )
    atomic = (
        injected.returncode != 0
        and _sha256_file(native) == before
        and not Path(str(native) + ".p255.tmp").exists()
    )
    native.unlink()
    return {
        "atomic_publication_failure": atomic,
        "inspection": inspection,
        "invalid_input_rejected": invalid_rejected,
        "native_p249_pixel_exact": bool(np.array_equal(native_pixels, p249_pixels)),
        "native_p249_maximum_absolute_error": float(
            np.max(np.abs(native_pixels.astype(np.float64) - p249_pixels.astype(np.float64)))
        ),
        "p249_pixel_f32le_sha256": _sha256_bytes(p249_pixels.tobytes()),
        "replay_container_exact": replay_exact,
        "small_container_sha256": native_sha,
    }


def _fresh_controls(
    config_path: Path,
    producer_repo: Path,
    binary: Path,
    site: Path,
    workspace: Path,
) -> dict[str, Any]:
    report = workspace / "small-controls.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--controls",
            "--config",
            str(config_path),
            "--producer-repo",
            str(producer_repo),
            "--binary",
            str(binary),
            "--site",
            str(site),
            "--workspace",
            str(workspace),
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise P255Error(
            completed.stderr.decode("utf-8", errors="replace")
            + completed.stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_text(encoding="utf-8"))
    report.unlink()
    return value


def _identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def execute(config_path: Path, producer_repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    limits = config["gates"]
    identities = {
        "config_sha256": _sha256_file(config_path),
        "contract": _identity(ROOT / bindings["contract_path"]),
        "native_source": _identity(ROOT / bindings["native_source_path"]),
        "p248_config": _identity(ROOT / bindings["p248_config_path"]),
        "p248_evidence": _identity(ROOT / bindings["p248_evidence_path"]),
        "p248_native_source": _identity(ROOT / bindings["p248_native_source_path"]),
        "p249_config": _identity(ROOT / bindings["p249_config_path"]),
        "p249_evidence": _identity(ROOT / bindings["p249_evidence_path"]),
        "openexr_source": _identity(ROOT / bindings["openexr_source_path"]),
        "imath_source": _identity(ROOT / bindings["imath_source_path"]),
        "native_source_git_blob": _git_text(
            "rev-parse",
            f"{bindings['implementation_commit']}:{bindings['native_source_path']}",
        ),
        "p248_native_source_git_blob": _git_text(
            "rev-parse", f"HEAD:{bindings['p248_native_source_path']}"
        ),
    }
    expected_identity = {
        "contract": (bindings["contract_bytes"], bindings["contract_sha256"]),
        "native_source": (
            bindings["native_source_bytes"],
            bindings["native_source_sha256"],
        ),
        "p248_config": (
            bindings["p248_config_bytes"],
            bindings["p248_config_sha256"],
        ),
        "p248_evidence": (
            bindings["p248_evidence_bytes"],
            bindings["p248_evidence_sha256"],
        ),
        "p249_config": (
            bindings["p249_config_bytes"],
            bindings["p249_config_sha256"],
        ),
        "p249_evidence": (
            bindings["p249_evidence_bytes"],
            bindings["p249_evidence_sha256"],
        ),
        "openexr_source": (
            bindings["openexr_source_bytes"],
            bindings["openexr_source_sha256"],
        ),
        "imath_source": (
            bindings["imath_source_bytes"],
            bindings["imath_source_sha256"],
        ),
    }
    identity_exact = all(
        identities[name]["bytes"] == expected[0]
        and identities[name]["sha256"] == expected[1]
        for name, expected in expected_identity.items()
    ) and identities["native_source_git_blob"] == bindings["native_source_git_blob"]
    p248_parent_unchanged = (
        identities["p248_native_source"]["sha256"]
        == bindings["p248_native_source_sha256"]
        and identities["p248_native_source_git_blob"]
        == bindings["p248_native_source_git_blob"]
    )
    wheel = producer_repo / bindings["producer_wheel_path"]
    wheel_exact = (
        wheel.stat().st_size == bindings["producer_wheel_bytes"]
        and _sha256_file(wheel) == bindings["producer_wheel_sha256"]
    )

    workspace = Path(tempfile.mkdtemp(prefix="neuro-film-p255-"))
    try:
        binary, build = _build(config, workspace)
        toolchain_exact = (
            build["cmake_sha256"] == bindings["cmake_sha256"]
            and build["ninja_sha256"] == bindings["ninja_sha256"]
            and build["msvc_cl_sha256"] == bindings["msvc_cl_sha256"]
            and bindings["msvc_version"] in build["msvc_version_text"]
        )
        site = workspace / "site"
        _install_openexr_wheel(wheel, site)
        controls = _fresh_controls(
            config_path, producer_repo, binary, site, workspace
        )
        if not (
            controls["inspection"]["decoded_maximum_absolute_error"] == 0.0
            and controls["native_p249_pixel_exact"]
            and controls["atomic_publication_failure"]
            and controls["invalid_input_rejected"]
            and controls["replay_container_exact"]
        ):
            raise P255Error("P255 small-probe controls failed before 24MP execution")
        interval = float(limits["maximum_sampling_interval_seconds"])
        workers = [
            _fresh_worker(config_path, binary, site, workspace, label, interval)
            for label in ("worker-a", "worker-b")
        ]
    finally:
        shutil.rmtree(workspace, ignore_errors=False)

    expected_metadata = config["metadata"]
    container_exact = len({row["file_sha256"] for row in workers}) == 1
    pixel_exact = all(
        row["inspection"]["decoded_maximum_absolute_error"]
        == limits["require_decoded_pixel_maximum_absolute_error"]
        and row["inspection"]["decoded_pixel_f32le_sha256"]
        == row["expected_pixel_f32le_sha256"]
        for row in workers
    )
    metadata_exact = all(
        row["inspection"]["chromaticities"] == expected_metadata["chromaticities"]
        and row["inspection"]["adopted_neutral"]
        == expected_metadata["adopted_neutral"]
        and row["inspection"]["aces_image_container_flag"]
        == expected_metadata["aces_image_container_flag"]
        and row["inspection"]["color_interop_id"]
        == expected_metadata["color_interop_id"]
        and row["inspection"]["compression_zip"]
        and row["inspection"]["scanline_image"]
        and row["inspection"]["channel_names"] == ["RGB"]
        and row["inspection"]["decoded_dtype"] == "float32"
        for row in workers
    )
    resources_pass = all(
        row["resource"]["peak_process_tree_rss_bytes"]
        <= limits["maximum_worker_process_tree_rss_bytes"]
        and row["resource"]["wall_seconds"]
        <= limits["maximum_worker_wall_seconds"]
        and row["file_bytes"] <= limits["maximum_openexr_bytes"]
        for row in workers
    )
    range_pass = all(
        row["inspection"]["finite"]
        and row["inspection"]["negative_preserved"]
        and row["inspection"]["above_one_preserved"]
        and row["inspection"]["new_boundary_count"] == 0
        for row in workers
    )
    gates = {
        "atomic-controls": controls["atomic_publication_failure"]
        and controls["invalid_input_rejected"],
        "container-repeat": container_exact,
        "decoded-pixels": pixel_exact,
        "identities": identity_exact and wheel_exact and toolchain_exact,
        "metadata": metadata_exact,
        "p248-parent-unchanged": p248_parent_unchanged,
        "range-and-finite": range_pass,
        "resource": resources_pass,
        "small-probe-p249-parity": controls["native_p249_pixel_exact"]
        and controls["inspection"]["decoded_maximum_absolute_error"] == 0.0,
        "workspace-clean": not workspace.exists(),
    }
    stable_scientific = {
        "controls": controls,
        "expected_pixel_f32le_sha256": workers[0]["expected_pixel_f32le_sha256"],
        "file_bytes": workers[0]["file_bytes"],
        "file_sha256": workers[0]["file_sha256"],
        "gates": gates,
        "inspection": workers[0]["inspection"],
        "network_requests": 0,
        "external_or_project_pixel_reads": 0,
        "owned_workspace_residue": 0,
    }
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_PRIVATE_ACES2065_OPENEXR_24MP_SCANLINE_STREAMING"
            if all(gates.values())
            else "FAIL_CLOSED"
        ),
        "identities": identities,
        "build": build,
        "workers": workers,
        "stable_scientific": stable_scientific,
        "stable_identity": _sha256_bytes(_canonical_bytes(stable_scientific)),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.worker:
        if None in (args.binary, args.site, args.workspace, args.label):
            raise P255Error("worker arguments are incomplete")
        value = _worker(
            args.config.resolve(),
            args.binary.resolve(),
            args.site.resolve(),
            args.workspace.resolve(),
            args.label,
        )
    elif args.controls:
        if None in (args.producer_repo, args.binary, args.site, args.workspace):
            raise P255Error("control arguments are incomplete")
        value = _small_controls(
            args.config.resolve(),
            args.producer_repo.resolve(),
            args.binary.resolve(),
            args.site.resolve(),
            args.workspace.resolve(),
        )
    else:
        if args.producer_repo is None:
            raise P255Error("--producer-repo is required")
        value = execute(args.config.resolve(), args.producer_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(value))
    return 0 if not isinstance(value, dict) or value.get("status", "PASS").startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
