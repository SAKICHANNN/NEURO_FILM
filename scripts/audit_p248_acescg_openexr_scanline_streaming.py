"""Build and audit the private P248 native scanline OpenEXR writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class P248Error(RuntimeError):
    """Raised when a frozen P248 identity or gate fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        roots: set[str] = set()
        extractable: list[tarfile.TarInfo] = []
        for member in members:
            parsed = PurePosixPath(member.name)
            if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
                raise P248Error(f"unsafe archive member: {member.name}")
            if member.islnk() or member.isdev():
                raise P248Error(f"unsupported archive member type: {member.name}")
            roots.add(parsed.parts[0])
            if member.issym():
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
                target_path = PurePosixPath(target)
                if (
                    target_path.is_absolute()
                    or ".." in target_path.parts
                    or not target_path.parts
                    or target_path.parts[0] != parsed.parts[0]
                ):
                    raise P248Error(f"unsafe archive symlink: {member.name}")
                continue
            extractable.append(member)
        if len(roots) != 1:
            raise P248Error("source archive must have exactly one root")
        source.extractall(destination, members=extractable, filter="data")
    return destination / roots.pop()


def _program_files_x86() -> Path:
    value = os.environ.get("ProgramFiles(x86)")
    if not value:
        raise P248Error("ProgramFiles(x86) is unavailable")
    return Path(value)


def _toolchain() -> dict[str, Any]:
    vs_root = _program_files_x86() / "Microsoft Visual Studio" / "18" / "BuildTools"
    vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    cmake = (
        vs_root
        / "Common7"
        / "IDE"
        / "CommonExtensions"
        / "Microsoft"
        / "CMake"
        / "CMake"
        / "bin"
        / "cmake.exe"
    )
    ninja = (
        vs_root
        / "Common7"
        / "IDE"
        / "CommonExtensions"
        / "Microsoft"
        / "CMake"
        / "Ninja"
        / "ninja.exe"
    )
    for path in (vcvars, cmake, ninja):
        if not path.is_file():
            raise P248Error(f"required toolchain file is absent: {path}")
    command = f'call "{vcvars}" >nul && set'
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        shell=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    environment = dict(os.environ)
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            environment[key] = value
            if key.casefold() == "path":
                environment["PATH"] = value
    cl = shutil.which("cl.exe", path=environment.get("PATH"))
    if cl is None:
        raise P248Error("cl.exe was not exposed by vcvars64")
    version = subprocess.run(
        [cl],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    version_text = version.stderr + version.stdout
    return {
        "cmake": cmake,
        "ninja": ninja,
        "cl": Path(cl),
        "environment": environment,
        "msvc_version_text": version_text.splitlines()[0].strip(),
    }


def _cmake_project(imath_source: Path, openexr_source: Path, native_source: Path) -> str:
    def q(path: Path) -> str:
        return path.resolve().as_posix()

    return f"""cmake_minimum_required(VERSION 3.20)
project(P248ScanlineWriter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL \"\" FORCE)
set(BUILD_TESTING OFF CACHE BOOL \"\" FORCE)
set(IMATH_INSTALL OFF CACHE BOOL \"\" FORCE)
set(IMATH_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)
set(IMATH_BUILD_PYTHON OFF CACHE BOOL \"\" FORCE)
add_subdirectory(\"{q(imath_source)}\" imath-build)
set(OPENEXR_INSTALL OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_ENABLE_THREADING OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL \"\" FORCE)
set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL \"\" FORCE)
add_subdirectory(\"{q(openexr_source)}\" openexr-build)
add_executable(p248_writer \"{q(native_source)}\")
target_link_libraries(p248_writer PRIVATE OpenEXR::OpenEXR)
"""


def _build(config: dict[str, Any], workspace: Path) -> tuple[Path, dict[str, Any]]:
    bindings = config["bindings"]
    openexr_archive = ROOT / bindings["openexr_source_path"]
    imath_archive = ROOT / bindings["imath_source_path"]
    native_source = ROOT / bindings["native_source_path"]
    openexr_source = _safe_extract(openexr_archive, workspace / "openexr-source")
    imath_source = _safe_extract(imath_archive, workspace / "imath-source")
    project = workspace / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        _cmake_project(imath_source, openexr_source, native_source), encoding="utf-8", newline="\n"
    )
    build = workspace / "build"
    tools = _toolchain()
    configure = [
        str(tools["cmake"]),
        "-S",
        str(project),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    configured = subprocess.run(
        configure, check=False, env=tools["environment"], capture_output=True
    )
    if configured.returncode != 0:
        raise P248Error(
            "native configure failed: "
            + configured.stdout.decode("utf-8", errors="replace")
            + configured.stderr.decode("utf-8", errors="replace")
        )
    built = subprocess.run(
        [str(tools["cmake"]), "--build", str(build), "--target", "p248_writer", "-j", "8"],
        check=False,
        env=tools["environment"],
        capture_output=True,
    )
    if built.returncode != 0:
        raise P248Error(
            "native build failed: "
            + built.stdout.decode("utf-8", errors="replace")
            + built.stderr.decode("utf-8", errors="replace")
        )
    binary = build / "p248_writer.exe"
    if not binary.is_file():
        raise P248Error("native build did not produce p248_writer.exe")
    identity = {
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": _sha256_file(binary),
        "cmake_sha256": _sha256_file(tools["cmake"]),
        "ninja_sha256": _sha256_file(tools["ninja"]),
        "msvc_cl_sha256": _sha256_file(tools["cl"]),
        "msvc_version_text": tools["msvc_version_text"],
    }
    return binary, identity


def _generate_block(y0: int, rows: int, width: int, period: int) -> np.ndarray:
    x = np.arange(width, dtype=np.uint32)[None, :]
    y = np.arange(y0, y0 + rows, dtype=np.uint32)[:, None]
    modulus = np.uint32(period)
    output = np.empty((rows, width, 3), dtype=np.float32)
    output[:, :, 0] = ((x + np.uint32(3) * y) % modulus).astype(np.float32) * np.float32(
        4.0 / (period - 1)
    ) - np.float32(0.25)
    output[:, :, 1] = ((np.uint32(5) * x + np.uint32(7) * y) % modulus).astype(
        np.float32
    ) * np.float32(2.0 / (period - 1))
    output[:, :, 2] = (np.bitwise_xor(x, np.uint32(13) * y) % modulus).astype(
        np.float32
    ) * np.float32(16.0 / (period - 1))
    return output


def _expected_sha(height: int, width: int, block: int, period: int) -> str:
    digest = hashlib.sha256()
    for y0 in range(0, height, block):
        rows = min(block, height - y0)
        digest.update(_generate_block(y0, rows, width, period).tobytes(order="C"))
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


def _inspect(
    path: Path,
    *,
    height: int,
    width: int,
    generation_block: int,
    period: int,
    openexr: Any,
) -> dict[str, Any]:
    with openexr.File(str(path)) as exr_file:
        header = dict(exr_file.header())
        channels = exr_file.channels()
        decoded = channels["RGB"].pixels.copy()
        names = sorted(channels)
    decoded_sha = hashlib.sha256(decoded.tobytes(order="C")).hexdigest()
    maximum_error = 0.0
    new_boundary = 0
    for y0 in range(0, height, generation_block):
        rows = min(generation_block, height - y0)
        expected = _generate_block(y0, rows, width, period)
        observed = decoded[y0 : y0 + rows]
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)))),
        )
        new_boundary += int(
            np.count_nonzero(
                ((observed == np.float32(0.0)) | (observed == np.float32(1.0)))
                & ~((expected == np.float32(0.0)) | (expected == np.float32(1.0)))
            )
        )
    return {
        "above_one_preserved": bool(np.max(decoded) > 1.0),
        "adopted_neutral": [float(value) for value in header["adoptedNeutral"]],
        "channel_names": names,
        "chromaticities": [float(value) for value in header["chromaticities"]],
        "compression_zip": bool(header["compression"] == openexr.ZIP_COMPRESSION),
        "decoded_dtype": str(decoded.dtype),
        "decoded_maximum_absolute_error": maximum_error,
        "decoded_pixel_f32le_sha256": decoded_sha,
        "decoded_shape": list(decoded.shape),
        "finite": bool(np.isfinite(decoded).all()),
        "negative_preserved": bool(np.min(decoded) < 0.0),
        "new_boundary_count": new_boundary,
        "scanline_image": bool(header["type"] == openexr.scanlineimage),
        "white_luminance_absent": "whiteLuminance" not in header,
    }


def _install_openexr_wheel(wheel: Path, site: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--target",
            str(site),
            str(wheel),
        ],
        check=True,
        capture_output=True,
    )


def _load_openexr(site: Path) -> Any:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    return importlib.import_module("OpenEXR")


def _worker(
    config_path: Path, binary: Path, site: Path, workspace: Path, label: str
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe = config["probe"]
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
        raise P248Error(completed.stderr.decode("utf-8", errors="replace"))
    openexr = _load_openexr(site)
    inspection = _inspect(
        output,
        height=int(probe["height"]),
        width=int(probe["width"]),
        generation_block=int(probe["generation_row_block"]),
        period=int(probe["coordinate_period"]),
        openexr=openexr,
    )
    result = {
        "file_bytes": output.stat().st_size,
        "file_sha256": _sha256_file(output),
        "expected_pixel_f32le_sha256": _expected_sha(
            int(probe["height"]),
            int(probe["width"]),
            int(probe["generation_row_block"]),
            int(probe["coordinate_period"]),
        ),
        "inspection": inspection,
    }
    output.unlink()
    return result


def _process_tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        total += process.memory_info().rss
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return total
    for child in children:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _fresh_worker(
    config_path: Path,
    binary: Path,
    site: Path,
    workspace: Path,
    label: str,
    interval: float,
) -> dict[str, Any]:
    report = workspace / f"{label}.json"
    command = [
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
    ]
    started = time.perf_counter()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = psutil.Process(child.pid)
    peak = 0
    samples = 0
    while child.poll() is None:
        peak = max(peak, _process_tree_rss(process))
        samples += 1
        time.sleep(interval)
    stdout, stderr = child.communicate()
    wall = time.perf_counter() - started
    if child.returncode != 0:
        raise P248Error(
            stderr.decode("utf-8", errors="replace") + stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_bytes())
    report.unlink()
    value["resource"] = {
        "peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "sampling_interval_seconds": interval,
        "wall_seconds": wall,
    }
    return value


def _fresh_controls(
    config_path: Path, binary: Path, site: Path, workspace: Path
) -> dict[str, Any]:
    report = workspace / "small-controls.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--controls",
            "--config",
            str(config_path),
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
        raise P248Error(
            completed.stderr.decode("utf-8", errors="replace")
            + completed.stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_bytes())
    report.unlink()
    return value


def _small_controls(binary: Path, site: Path, workspace: Path, config: dict[str, Any]) -> dict[str, Any]:
    openexr = _load_openexr(site)
    probe = workspace / "small.exr"
    result = _run_native(binary, probe, width=17, height=13, row_block=16, period=31)
    if result.returncode != 0:
        raise P248Error(result.stderr.decode("utf-8", errors="replace"))
    first_sha = _sha256_file(probe)
    inspection = _inspect(
        probe,
        height=13,
        width=17,
        generation_block=7,
        period=31,
        openexr=openexr,
    )
    probe.unlink()
    replay = _run_native(binary, probe, width=17, height=13, row_block=16, period=31)
    replay_exact = replay.returncode == 0 and _sha256_file(probe) == first_sha
    probe.unlink()
    invalid = _run_native(binary, probe, width=0, height=13, row_block=16, period=31)
    invalid_rejected = invalid.returncode != 0 and not probe.exists()
    probe.write_bytes(b"p248-existing")
    before = _sha256_file(probe)
    injected = _run_native(
        binary, probe, width=17, height=13, row_block=16, period=31, inject=True
    )
    atomic = (
        injected.returncode != 0
        and _sha256_file(probe) == before
        and not Path(str(probe) + ".p248.tmp").exists()
    )
    probe.unlink()
    return {
        "atomic_publication_failure": atomic,
        "inspection": inspection,
        "invalid_input_rejected": invalid_rejected,
        "replay_container_exact": replay_exact,
        "small_container_sha256": first_sha,
    }


def execute(config_path: Path, producer_repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    limits = config["gates"]
    p246_config = json.loads((ROOT / "configs/p246_acescg_openexr_exact_consumer_intake_v1.json").read_text())
    wheel_binding = p246_config["bindings"]
    wheel = producer_repo / wheel_binding["producer_wheel_path"]
    identities = {
        "contract_sha256": _sha256_file(ROOT / bindings["contract_path"]),
        "p246_evidence_sha256": _sha256_file(ROOT / bindings["p246_evidence_path"]),
        "p247_evidence_sha256": _sha256_file(ROOT / bindings["p247_evidence_path"]),
        "p247_config_sha256": _sha256_file(ROOT / bindings["p247_config_path"]),
        "openexr_source_bytes": (ROOT / bindings["openexr_source_path"]).stat().st_size,
        "openexr_source_sha256": _sha256_file(ROOT / bindings["openexr_source_path"]),
        "imath_source_bytes": (ROOT / bindings["imath_source_path"]).stat().st_size,
        "imath_source_sha256": _sha256_file(ROOT / bindings["imath_source_path"]),
    }
    if not all(identities[key] == bindings[key] for key in identities):
        raise P248Error("frozen source or parent identity differs")
    if wheel.stat().st_size != wheel_binding["producer_wheel_bytes"] or _sha256_file(wheel) != wheel_binding[
        "producer_wheel_sha256"
    ]:
        raise P248Error("frozen OpenEXR Python wheel differs")

    scratch = ROOT / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="p248-scanline-", dir=scratch))
    try:
        binary, build_identity = _build(config, workspace)
        for key in ("cmake_sha256", "ninja_sha256", "msvc_cl_sha256"):
            if build_identity[key] != bindings[key]:
                raise P248Error(f"frozen toolchain identity differs: {key}")
        site = workspace / "site"
        _install_openexr_wheel(wheel, site)
        controls = _fresh_controls(config_path, binary, site, workspace)
        interval = float(limits["maximum_sampling_interval_seconds"]) / 2.0
        workers = [
            _fresh_worker(config_path, binary, site, workspace, label, interval)
            for label in ("outer-a", "outer-b")
        ]
        first, second = workers
        p246 = json.loads((ROOT / bindings["p246_evidence_path"]).read_text(encoding="utf-8"))
        expected_metadata = p246["result"]
        scientific_exact = {
            "container": first["file_sha256"] == second["file_sha256"],
            "decoded": first["inspection"] == second["inspection"],
            "expected": first["expected_pixel_f32le_sha256"]
            == second["expected_pixel_f32le_sha256"],
        }
        metadata_exact = all(
            row["inspection"]["chromaticities"] == expected_metadata["chromaticities"]
            and row["inspection"]["adopted_neutral"] == expected_metadata["adopted_neutral"]
            for row in workers
        )
        gates = {
            "atomic_controls": controls["atomic_publication_failure"]
            and controls["invalid_input_rejected"],
            "container_repeat": all(scientific_exact.values()),
            "decoded_pixels": all(
                row["inspection"]["decoded_maximum_absolute_error"]
                <= float(limits["require_decoded_pixel_maximum_absolute_error"])
                and row["inspection"]["decoded_pixel_f32le_sha256"]
                == row["expected_pixel_f32le_sha256"]
                for row in workers
            ),
            "metadata": metadata_exact,
            "range_and_finite": all(
                row["inspection"]["finite"]
                and row["inspection"]["negative_preserved"]
                and row["inspection"]["above_one_preserved"]
                and row["inspection"]["new_boundary_count"] == 0
                for row in workers
            ),
            "resource": all(
                row["resource"]["peak_process_tree_rss_bytes"]
                <= int(limits["maximum_worker_process_tree_rss_bytes"])
                and row["resource"]["wall_seconds"] <= float(limits["maximum_worker_wall_seconds"])
                and row["file_bytes"] <= int(limits["maximum_openexr_bytes"])
                for row in workers
            ),
            "small_probe": controls["replay_container_exact"]
            and controls["inspection"]["decoded_maximum_absolute_error"] == 0.0,
        }
        status = (
            "PASS_PRIVATE_ACESCG_OPENEXR_24MP_SCANLINE_STREAMING"
            if all(gates.values())
            else "FAIL_CLOSED_ACESCG_OPENEXR_24MP_SCANLINE_STREAMING"
        )
        scientific = {
            "build": build_identity,
            "controls": controls,
            "gates": gates,
            "identities": identities,
            "status": status,
            "workers": workers,
        }
        return {
            "schema": "neuro-film.p248-acescg-openexr-scanline-streaming-result.v1",
            "experiment_id": "P248",
            "status": status,
            "scientific": scientific,
            "scientific_identity": hashlib.sha256(_canonical_bytes(scientific)).hexdigest(),
            "network_requests": 0,
            "external_or_project_pixel_reads": 0,
            "claim_ceiling": config["claim_ceiling"],
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=False)


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
        if args.binary is None or args.site is None or args.workspace is None or args.label is None:
            raise P248Error("worker arguments are incomplete")
        value = _worker(args.config, args.binary, args.site, args.workspace, args.label)
    elif args.controls:
        if args.binary is None or args.site is None or args.workspace is None:
            raise P248Error("control arguments are incomplete")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        value = _small_controls(args.binary, args.site, args.workspace, config)
    else:
        if args.producer_repo is None:
            raise P248Error("--producer-repo is required")
        value = execute(args.config, args.producer_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
