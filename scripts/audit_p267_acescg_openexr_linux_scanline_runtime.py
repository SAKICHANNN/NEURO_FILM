"""Build and audit the private P267 Linux scanline OpenEXR writer."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p248_acescg_openexr_scanline_streaming import (
    _canonical_bytes,
    _install_openexr_wheel,
    _sha256_file,
)


class P267Error(RuntimeError):
    """Raised when a frozen P267 identity or gate fails."""


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    )
    return completed.stdout.decode("ascii").strip()


def _wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    if len(resolved) < 3 or resolved[1:3] != ":\\":
        raise P267Error(f"path is not an absolute Windows drive path: {resolved}")
    drive = resolved[0].lower()
    tail = resolved[3:].replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def _wsl(
    distribution: str,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment["WSL_UTF8"] = "1"
    completed = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", *arguments],
        check=False,
        capture_output=True,
        env=environment,
    )
    if check and completed.returncode != 0:
        raise P267Error(
            completed.stdout.decode("utf-8", errors="replace")
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed


def _linux_write_text(distribution: str, path: str, value: str) -> None:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    _wsl(
        distribution,
        "python3",
        "-c",
        (
            "import base64,pathlib,sys; "
            "pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode(sys.argv[2]))"
        ),
        path,
        encoded,
    )


def _linux_safe_extract(
    distribution: str,
    archive: str,
    destination: str,
) -> str:
    code = r"""
import pathlib
import posixpath
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive, "r:gz") as source:
    members = source.getmembers()
    roots = set()
    for member in members:
        parsed = pathlib.PurePosixPath(member.name)
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if member.islnk() or member.isdev():
            raise RuntimeError(f"unsupported archive member: {member.name}")
        roots.add(parsed.parts[0])
        if member.issym():
            target = pathlib.PurePosixPath(posixpath.normpath(
                posixpath.join(posixpath.dirname(member.name), member.linkname)
            ))
            if (target.is_absolute() or ".." in target.parts or not target.parts
                    or target.parts[0] != parsed.parts[0]):
                raise RuntimeError(f"unsafe archive symlink: {member.name}")
    if len(roots) != 1:
        raise RuntimeError("source archive must have exactly one root")
    source.extractall(destination)
print(str(destination / roots.pop()))
"""
    completed = _wsl(distribution, "python3", "-c", code, archive, destination)
    lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
    if not lines:
        raise P267Error("Linux source extraction returned no root")
    return lines[-1].strip()


def _cmake_project(imath_source: str, openexr_source: str, native_source: str) -> str:
    return f"""cmake_minimum_required(VERSION 3.20)
project(P267ScanlineWriter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL \"\" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL \"\" FORCE)
set(BUILD_TESTING OFF CACHE BOOL \"\" FORCE)
set(IMATH_INSTALL OFF CACHE BOOL \"\" FORCE)
set(IMATH_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)
set(IMATH_BUILD_PYTHON OFF CACHE BOOL \"\" FORCE)
add_subdirectory(\"{imath_source}\" imath-build)
set(OPENEXR_INSTALL OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_ENABLE_THREADING OFF CACHE BOOL \"\" FORCE)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL \"\" FORCE)
set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL \"\" FORCE)
add_subdirectory(\"{openexr_source}\" openexr-build)
add_executable(p267_writer \"{native_source}\")
target_link_libraries(p267_writer PRIVATE OpenEXR::OpenEXR)
"""


def _linux_sha(distribution: str, path: str) -> str:
    completed = _wsl(distribution, "/usr/bin/sha256sum", path)
    return completed.stdout.decode("ascii").split()[0]


def _linux_size(distribution: str, path: str) -> int:
    completed = _wsl(distribution, "/usr/bin/stat", "-c", "%s", path)
    return int(completed.stdout.decode("ascii").strip())


def _linux_version(distribution: str, path: str) -> str:
    completed = _wsl(distribution, path, "--version")
    return completed.stdout.decode("utf-8", errors="replace").splitlines()[0].strip()


def _build_linux(
    config: dict[str, Any],
    linux_root: str,
) -> tuple[str, dict[str, Any]]:
    distribution = config["platform"]["wsl_distribution"]
    bindings = config["bindings"]
    _wsl(distribution, "/usr/bin/mkdir", "-p", linux_root)
    cmake_site = f"{linux_root}/cmake-site"
    cmake_wheel = _wsl_path(ROOT / bindings["cmake_wheel_path"])
    _wsl(
        distribution,
        "python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--no-deps",
        "--target",
        cmake_site,
        cmake_wheel,
    )
    cmake = f"{cmake_site}/cmake/data/bin/cmake"
    openexr_source = _linux_safe_extract(
        distribution,
        _wsl_path(ROOT / bindings["openexr_source_path"]),
        f"{linux_root}/openexr-source",
    )
    imath_source = _linux_safe_extract(
        distribution,
        _wsl_path(ROOT / bindings["imath_source_path"]),
        f"{linux_root}/imath-source",
    )
    project = f"{linux_root}/project"
    _wsl(distribution, "/usr/bin/mkdir", "-p", project)
    _linux_write_text(
        distribution,
        f"{project}/CMakeLists.txt",
        _cmake_project(
            imath_source,
            openexr_source,
            _wsl_path(ROOT / bindings["linux_native_source_path"]),
        ),
    )
    build = f"{linux_root}/build"
    _wsl(
        distribution,
        cmake,
        "-S",
        project,
        "-B",
        build,
        "-G",
        "Ninja",
        "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=/usr/bin/gcc",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
    )
    _wsl(
        distribution,
        cmake,
        "--build",
        build,
        "--target",
        "p267_writer",
        "-j",
        str(config["build"]["build_jobs"]),
    )
    binary = f"{build}/p267_writer"
    identity = {
        "binary_bytes": _linux_size(distribution, binary),
        "binary_sha256": _linux_sha(distribution, binary),
        "cmake_sha256": _linux_sha(distribution, cmake),
        "cmake_version": _linux_version(distribution, cmake),
        "gcc_sha256": _linux_sha(distribution, "/usr/bin/gcc"),
        "gcc_version": _linux_version(distribution, "/usr/bin/gcc"),
        "gxx_sha256": _linux_sha(distribution, "/usr/bin/g++"),
        "gxx_version": _linux_version(distribution, "/usr/bin/g++"),
        "ninja_sha256": _linux_sha(distribution, "/usr/bin/ninja"),
        "ninja_version": _linux_version(distribution, "/usr/bin/ninja"),
    }
    return binary, identity


def _run_linux_native(
    distribution: str,
    binary: str,
    output: Path,
    *,
    width: int,
    height: int,
    row_block: int,
    period: int,
    inject: bool = False,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    linux_output = _wsl_path(output)
    time_file = f"/tmp/p267-time-{uuid.uuid4().hex}.txt"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        time_file,
        binary,
        "--output",
        linux_output,
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
    started = time.perf_counter()
    completed = _wsl(distribution, *command, check=False)
    wall = time.perf_counter() - started
    time_result = _wsl(distribution, "/usr/bin/cat", time_file, check=False)
    _wsl(distribution, "/usr/bin/rm", "-f", time_file, check=False)
    maximum_rss_kib = None
    if time_result.returncode == 0:
        for line in time_result.stdout.decode("utf-8", errors="replace").splitlines():
            if "Maximum resident set size (kbytes):" in line:
                maximum_rss_kib = int(line.rsplit(":", 1)[1].strip())
                break
    return completed, {
        "maximum_resident_set_size_bytes": (
            None if maximum_rss_kib is None else maximum_rss_kib * 1024
        ),
        "wall_seconds": wall,
    }


def _inspect_worker(
    config: dict[str, Any],
    binary: str,
    workspace: Path,
    label: str,
    site: Path,
) -> dict[str, Any]:
    probe = config["probe"]
    output = workspace / f"{label}.exr"
    completed, resource = _run_linux_native(
        config["platform"]["wsl_distribution"],
        binary,
        output,
        width=int(probe["width"]),
        height=int(probe["height"]),
        row_block=int(probe["write_row_block"]),
        period=int(probe["coordinate_period"]),
    )
    if completed.returncode != 0:
        raise P267Error(completed.stderr.decode("utf-8", errors="replace"))
    inspection = _fresh_inspect(
        output,
        site=site,
        height=int(probe["height"]),
        width=int(probe["width"]),
        generation_block=int(probe["generation_row_block"]),
        period=int(probe["coordinate_period"]),
    )
    result = {
        "file_bytes": output.stat().st_size,
        "file_sha256": _sha256_file(output),
        "inspection": inspection,
        "resource": resource,
    }
    output.unlink()
    return result


def _small_controls(
    config: dict[str, Any],
    binary: str,
    workspace: Path,
    site: Path,
) -> dict[str, Any]:
    distribution = config["platform"]["wsl_distribution"]
    probe = workspace / "small.exr"
    first, _ = _run_linux_native(
        distribution,
        binary,
        probe,
        width=17,
        height=13,
        row_block=16,
        period=31,
    )
    if first.returncode != 0:
        raise P267Error(first.stderr.decode("utf-8", errors="replace"))
    first_sha = _sha256_file(probe)
    inspection = _fresh_inspect(
        probe,
        site=site,
        height=13,
        width=17,
        generation_block=7,
        period=31,
    )
    probe.unlink()
    second, _ = _run_linux_native(
        distribution,
        binary,
        probe,
        width=17,
        height=13,
        row_block=16,
        period=31,
    )
    replay_exact = second.returncode == 0 and _sha256_file(probe) == first_sha
    probe.unlink()
    invalid, _ = _run_linux_native(
        distribution,
        binary,
        probe,
        width=0,
        height=13,
        row_block=16,
        period=31,
    )
    invalid_rejected = invalid.returncode != 0 and not probe.exists()
    probe.write_bytes(b"p267-existing")
    before = _sha256_file(probe)
    injected, _ = _run_linux_native(
        distribution,
        binary,
        probe,
        width=17,
        height=13,
        row_block=16,
        period=31,
        inject=True,
    )
    atomic = (
        injected.returncode != 0
        and _sha256_file(probe) == before
        and not Path(str(probe) + ".p267.tmp").exists()
    )
    probe.unlink()
    return {
        "atomic_publication_failure": atomic,
        "inspection": inspection,
        "invalid_input_rejected": invalid_rejected,
        "replay_container_exact": replay_exact,
        "small_container_sha256": first_sha,
    }


def _fresh_inspect(
    path: Path,
    *,
    site: Path,
    height: int,
    width: int,
    generation_block: int,
    period: int,
) -> dict[str, Any]:
    code = r"""
import importlib
import json
import pathlib
import sys

site, root, path = sys.argv[1:4]
sys.path.insert(0, site)
sys.path.insert(0, root)
from scripts.audit_p248_acescg_openexr_scanline_streaming import _inspect
openexr = importlib.import_module("OpenEXR")
value = _inspect(
    pathlib.Path(path),
    height=int(sys.argv[4]),
    width=int(sys.argv[5]),
    generation_block=int(sys.argv[6]),
    period=int(sys.argv[7]),
    openexr=openexr,
)
sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(site),
            str(ROOT),
            str(path),
            str(height),
            str(width),
            str(generation_block),
            str(period),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise P267Error(completed.stderr.decode("utf-8", errors="replace"))
    return json.loads(completed.stdout)


def _stable_controller_payload(controller: dict[str, Any]) -> bytes:
    value = copy.deepcopy(controller)
    value.pop("controller_order")
    build = value["build"]
    build.pop("binary_bytes")
    build.pop("binary_sha256")
    for worker in value["workers"]:
        worker.pop("resource")
        worker.pop("file_sha256")
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cleanup_linux(distribution: str, linux_root: str) -> None:
    if not linux_root.startswith("/tmp/p267-") or "/../" in linux_root:
        raise P267Error(f"refusing unsafe Linux cleanup target: {linux_root}")
    _wsl(distribution, "/usr/bin/rm", "-rf", linux_root)


def _validate_bindings(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    bindings = config["bindings"]
    identities = {
        "config_sha256": _sha256_file(config_path),
        "contract_bytes": (ROOT / bindings["contract_path"]).stat().st_size,
        "contract_sha256": _sha256_file(ROOT / bindings["contract_path"]),
        "p248_evidence_bytes": (ROOT / bindings["p248_evidence_path"]).stat().st_size,
        "p248_evidence_sha256": _sha256_file(ROOT / bindings["p248_evidence_path"]),
        "p248_config_bytes": (ROOT / bindings["p248_config_path"]).stat().st_size,
        "p248_config_sha256": _sha256_file(ROOT / bindings["p248_config_path"]),
        "p248_native_source_bytes": (ROOT / bindings["p248_native_source_path"])
        .stat()
        .st_size,
        "p248_native_source_sha256": _sha256_file(
            ROOT / bindings["p248_native_source_path"]
        ),
        "openexr_source_bytes": (ROOT / bindings["openexr_source_path"]).stat().st_size,
        "openexr_source_sha256": _sha256_file(ROOT / bindings["openexr_source_path"]),
        "imath_source_bytes": (ROOT / bindings["imath_source_path"]).stat().st_size,
        "imath_source_sha256": _sha256_file(ROOT / bindings["imath_source_path"]),
        "cmake_wheel_bytes": (ROOT / bindings["cmake_wheel_path"]).stat().st_size,
        "cmake_wheel_sha256": _sha256_file(ROOT / bindings["cmake_wheel_path"]),
    }
    for key, observed in identities.items():
        if key == "config_sha256":
            continue
        if bindings[key] != observed:
            raise P267Error(f"frozen binding differs: {key}")
    if config["status"] == "FORMAL_EXECUTION_LOCKED":
        for prefix in ("linux_native_source", "runner"):
            path = ROOT / bindings[f"{prefix}_path"]
            if path.stat().st_size != bindings[f"{prefix}_bytes"]:
                raise P267Error(f"frozen binding differs: {prefix}_bytes")
            if _sha256_file(path) != bindings[f"{prefix}_sha256"]:
                raise P267Error(f"frozen binding differs: {prefix}_sha256")
            if _git_text(
                "rev-parse",
                f"{bindings[f'{prefix}_commit']}:{bindings[f'{prefix}_path']}",
            ) != _git_text("hash-object", str(path)):
                raise P267Error(f"frozen Git object differs: {prefix}")
    return identities


def execute(
    config_path: Path, producer_repo: Path, *, preflight: bool = False
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not preflight and config["status"] != "FORMAL_EXECUTION_LOCKED":
        raise P267Error("formal execution requires a locked config")
    identities = _validate_bindings(config_path, config)
    p246 = json.loads(
        (ROOT / "configs/p246_acescg_openexr_exact_consumer_intake_v1.json").read_text()
    )
    wheel_binding = p246["bindings"]
    wheel = producer_repo / wheel_binding["producer_wheel_path"]
    if (
        not wheel.is_file()
        or wheel.stat().st_size != wheel_binding["producer_wheel_bytes"]
        or _sha256_file(wheel) != wheel_binding["producer_wheel_sha256"]
    ):
        raise P267Error("frozen Windows OpenEXR inspection wheel differs")

    workspace = Path(tempfile.mkdtemp(prefix="p267-linux-scanline-", dir=ROOT / "tmp"))
    if ROOT / "tmp" not in workspace.parents:
        raise P267Error("owned workspace escaped repository tmp")
    site = workspace / "site"
    _install_openexr_wheel(wheel, site)
    controllers = []
    try:
        orders = config["execution"]["controller_orders"][: 1 if preflight else None]
        for index, order in enumerate(orders):
            linux_root = f"/tmp/p267-{uuid.uuid4().hex}"
            try:
                binary, build = _build_linux(config, linux_root)
                controls = _small_controls(config, binary, workspace, site)
                workers = [
                    _inspect_worker(config, binary, workspace, label, site)
                    for label in order
                ]
                workers.sort(
                    key=lambda row: row["inspection"]["decoded_pixel_f32le_sha256"]
                )
                controllers.append(
                    {
                        "build": build,
                        "controller_order": order,
                        "controls": controls,
                        "workers": workers,
                    }
                )
            finally:
                _cleanup_linux(config["platform"]["wsl_distribution"], linux_root)

        expected_sha = config["bindings"]["p248_expected_pixel_f32le_sha256"]
        limits = config["gates"]
        p248_config = json.loads(
            (ROOT / config["bindings"]["p248_config_path"]).read_text(encoding="utf-8")
        )
        p246_evidence_path = ROOT / p248_config["bindings"]["p246_evidence_path"]
        if (
            _sha256_file(p246_evidence_path)
            != p248_config["bindings"]["p246_evidence_sha256"]
        ):
            raise P267Error("P248-bound P246 metadata evidence differs")
        expected_metadata = json.loads(p246_evidence_path.read_text(encoding="utf-8"))[
            "result"
        ]
        all_workers = [
            worker for controller in controllers for worker in controller["workers"]
        ]
        gates = {
            "atomic_controls": all(
                controller["controls"]["atomic_publication_failure"]
                and controller["controls"]["invalid_input_rejected"]
                for controller in controllers
            ),
            "decoded_pixels": all(
                worker["inspection"]["decoded_pixel_f32le_sha256"] == expected_sha
                and worker["inspection"]["decoded_maximum_absolute_error"] == 0.0
                for worker in all_workers
            ),
            "metadata": all(
                worker["inspection"]["chromaticities"]
                == expected_metadata["chromaticities"]
                and worker["inspection"]["adopted_neutral"]
                == expected_metadata["adopted_neutral"]
                for worker in all_workers
            ),
            "range_and_finite": all(
                worker["inspection"]["finite"]
                and worker["inspection"]["negative_preserved"]
                and worker["inspection"]["above_one_preserved"]
                and worker["inspection"]["new_boundary_count"] == 0
                for worker in all_workers
            ),
            "resource": all(
                worker["resource"]["maximum_resident_set_size_bytes"] is not None
                and worker["resource"]["maximum_resident_set_size_bytes"]
                <= limits["maximum_worker_process_tree_rss_bytes"]
                and worker["resource"]["wall_seconds"]
                <= limits["maximum_worker_wall_seconds"]
                and worker["file_bytes"] <= limits["maximum_openexr_bytes"]
                for worker in all_workers
            ),
            "small_probe": all(
                controller["controls"]["replay_container_exact"]
                and controller["controls"]["inspection"][
                    "decoded_maximum_absolute_error"
                ]
                == 0.0
                for controller in controllers
            ),
        }
        stable_payloads = [
            _stable_controller_payload(controller) for controller in controllers
        ]
        controller_scientific_exact = (
            len(stable_payloads) == 1 or len(set(stable_payloads)) == 1
        )
        gates["controller_scientific_exact"] = controller_scientific_exact
        status = (
            "PASS_PRIVATE_ACESCG_OPENEXR_24MP_LINUX_SCANLINE_RUNTIME"
            if all(gates.values())
            else "FAIL_CLOSED_ACESCG_OPENEXR_24MP_LINUX_SCANLINE_RUNTIME"
        )
        scientific = {
            "controllers": controllers,
            "gates": gates,
            "identities": identities,
            "status": status,
        }
        return {
            "schema": "neuro-film.p267-acescg-openexr-linux-scanline-runtime-result.v1",
            "experiment_id": "P267",
            "status": status,
            "preflight": preflight,
            "scientific": scientific,
            "stable_controller_identity_sha256": (
                hashlib.sha256(stable_payloads[0]).hexdigest()
                if stable_payloads
                else None
            ),
            "network_requests": 0,
            "external_or_project_pixel_reads": 0,
            "claim_ceiling": config["claim_ceiling"],
        }
    finally:
        shutil.rmtree(workspace)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    value = execute(args.config, args.producer_repo, preflight=args.preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
