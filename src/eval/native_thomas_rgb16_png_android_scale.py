"""12MP host/Android runtime audit for the freestanding Thomas PNG core."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import (
    _decode_rgb16,
    _icc_payload,
)
from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.native_thomas_rgb16_png_android_runtime import (
    NativeThomasRgb16PngAndroidRuntimeError,
    _adb,
    _android_env,
    _finish_owned_emulator_processes,
    _owned_emulator_processes,
    _wait_for_boot,
)
from src.eval.native_thomas_rgb16_png_conformance import SOURCE_PATHS
from src.preprocess.output_encode import srgb_icc_profile

PROBE_SOURCE = "native/film_physics/nf_thomas_rgb16_png_scale_probe_v1.c"
_FACT_PATTERN = re.compile(
    r"^status=(?P<status>\d+) height=(?P<height>\d+) width=(?P<width>\d+) "
    r"rows=(?P<rows>\d+) bytes=(?P<bytes>\d+) calls=(?P<calls>\d+) "
    r"workspace=(?P<workspace>\d+) means=(?P<means>.+)$"
)


class NativeThomasRgb16PngScaleError(RuntimeError):
    """Raised when the scale-runtime contract or execution fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sources(root: Path) -> list[Path]:
    return [*(root / path for path in SOURCE_PATHS), root / PROBE_SOURCE]


def _run(command: list[str], *, cwd: Path, timeout: float = 240.0) -> str:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, check=False, timeout=timeout
    )
    output = (completed.stdout + completed.stderr).decode(errors="replace")
    if completed.returncode != 0:
        raise NativeThomasRgb16PngScaleError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output.strip()


def build_msvc_probe(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = (output_dir / "nf_p8cn_scale_probe.exe").resolve()
    sources = " ".join(f'"{path.resolve()}"' for path in _sources(root))
    batch = output_dir / "build_p8cn.bat"
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        "cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX "
        "/D_CRT_SECURE_NO_WARNINGS "
        f'/I"{(root / "native").resolve()}" '
        f'/I"{(root / "native/film_physics").resolve()}" {sources} '
        f'/Fe:"{executable}" /link /Brepro\r\n',
        encoding="ascii",
        newline="",
    )
    _run(["cmd.exe", "/d", "/c", str(batch.resolve())], cwd=output_dir)
    return {
        "toolchain": "msvc-x64-c11-strict",
        "executable": str(executable),
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": sha256_file(executable),
    }


def build_android_probe(root: Path, ndk: Path, output: Path) -> dict[str, Any]:
    clang = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    if not clang.is_file():
        raise NativeThomasRgb16PngScaleError("Android NDK clang missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(clang),
            "--target=x86_64-linux-android21",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-model=strict",
            "-I",
            str(root / "native"),
            "-I",
            str(root / "native/film_physics"),
            *(str(path) for path in _sources(root)),
            "-fPIE",
            "-pie",
            "-Wl,--build-id=none",
            "-Wl,--no-undefined",
            "-lm",
            "-o",
            str(output),
        ],
        cwd=root,
    )
    return {
        "toolchain": "android-ndk-r27d-x86_64-c11-strict",
        "executable": str(output),
        "executable_bytes": output.stat().st_size,
        "executable_sha256": sha256_file(output),
        "clang_sha256": sha256_file(clang),
    }


def parse_facts(stdout: str) -> dict[str, Any]:
    match = _FACT_PATTERN.fullmatch(stdout.strip())
    if match is None:
        raise NativeThomasRgb16PngScaleError("scale probe stdout drift")
    row = match.groupdict()
    return {
        name: int(row[name])
        for name in ("status", "height", "width", "rows", "bytes", "calls", "workspace")
    }


def inspect_png(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    decoded = _decode_rgb16(payload)
    icc = _icc_payload(payload)
    return {
        "png_bytes": len(payload),
        "png_sha256": hashlib.sha256(payload).hexdigest(),
        "shape_hwc": list(decoded.shape),
        "dtype": str(decoded.dtype),
        "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
        "icc_sha256": hashlib.sha256(icc).hexdigest(),
        "icc_exact": icc == srgb_icc_profile(),
    }


def run_host_probe(
    executable: Path,
    png: Path,
    *,
    height: int,
    width: int,
    row_partition: int,
) -> dict[str, Any]:
    png.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [str(executable), str(png), str(height), str(width), str(row_partition)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak = 0
    observed = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            peak = max(peak, observed.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    wall = time.monotonic() - started
    if process.returncode != 0:
        raise NativeThomasRgb16PngScaleError(stderr.decode(errors="replace"))
    text = stdout.decode(errors="replace").strip()
    return {
        "wall_seconds": wall,
        "peak_rss_bytes": peak,
        "facts": parse_facts(text),
        **inspect_png(png),
    }


def _one_android_boot(
    *,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    port: int,
    probe: Path,
    output_dir: Path,
    boot_index: int,
    height: int,
    width: int,
    row_partition: int,
) -> dict[str, Any]:
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    env = _android_env(sdk, avd_home)
    serial = f"emulator-{port}"
    stdout_path = output_dir / f"boot{boot_index}_emulator.stdout.log"
    stderr_path = output_dir / f"boot{boot_index}_emulator.stderr.log"
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] | None = None
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        started = time.monotonic()
        process = subprocess.Popen(
            [
                str(emulator_exe),
                "-avd",
                avd_name,
                "-port",
                str(port),
                "-no-window",
                "-no-audio",
                "-no-boot-anim",
                "-no-snapshot",
                "-wipe-data",
                "-gpu",
                "swiftshader_indirect",
                "-accel",
                "auto",
            ],
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_for_boot(adb, serial, process, env=env, timeout=180.0)
            boot_seconds = time.monotonic() - started
            remote_probe = "/data/local/tmp/nf_p8cn_probe"
            remote_png = "/data/local/tmp/nf_p8cn_12mp.png"
            _adb(adb, serial, "push", str(probe), remote_probe, env=env)
            _adb(adb, serial, "shell", "chmod", "755", remote_probe, env=env)
            run_started = time.monotonic()
            text = _adb(
                adb,
                serial,
                "shell",
                remote_probe,
                remote_png,
                str(height),
                str(width),
                str(row_partition),
                env=env,
                timeout=180.0,
            )
            device_seconds = time.monotonic() - run_started
            local_png = output_dir / f"android_boot{boot_index}.png"
            _adb(
                adb, serial, "pull", remote_png, str(local_png), env=env, timeout=120.0
            )
            listing = _adb(adb, serial, "shell", "ps", "-A", env=env)
            result = {
                "boot_seconds": boot_seconds,
                "device_command_seconds": device_seconds,
                "device": {
                    "abi": _adb(
                        adb, serial, "shell", "getprop", "ro.product.cpu.abi", env=env
                    ),
                    "api": _adb(
                        adb, serial, "shell", "getprop", "ro.build.version.sdk", env=env
                    ),
                    "model": _adb(
                        adb, serial, "shell", "getprop", "ro.product.model", env=env
                    ),
                    "build_fingerprint": _adb(
                        adb, serial, "shell", "getprop", "ro.build.fingerprint", env=env
                    ),
                },
                "facts": parse_facts(text),
                "probe_process_survived": "nf_p8cn_probe" in listing,
                **inspect_png(local_png),
            }
        finally:
            try:
                _adb(
                    adb,
                    serial,
                    "shell",
                    "rm",
                    "-f",
                    "/data/local/tmp/nf_p8cn_probe",
                    "/data/local/tmp/nf_p8cn_12mp.png",
                    env=env,
                    timeout=15.0,
                )
            except (
                NativeThomasRgb16PngAndroidRuntimeError,
                subprocess.TimeoutExpired,
            ):
                pass
            try:
                _adb(adb, serial, "emu", "kill", env=env, timeout=15.0)
            except (
                NativeThomasRgb16PngAndroidRuntimeError,
                subprocess.TimeoutExpired,
            ):
                pass
            if process is not None:
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15.0)
    _finish_owned_emulator_processes(emulator_exe, avd_name, port)
    if result is None:
        raise NativeThomasRgb16PngScaleError("Android scale run produced no result")
    result["emulator_process_survived"] = bool(
        _owned_emulator_processes(emulator_exe, avd_name, port)
    )
    return result


def evaluate(
    *,
    root: Path,
    contract_path: Path,
    ndk: Path,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    output_dir: Path,
    port: int = 5582,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cn_android_thomas_rgb16_png_scale_contract.v1"
    ):
        raise NativeThomasRgb16PngScaleError("P8CN contract drift")
    parent = contract["parent"]
    parent_path = root / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise NativeThomasRgb16PngScaleError("P8CM parent evidence drift")
    fixture = contract["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    row_partition = int(fixture["row_partition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    host_build = build_msvc_probe(root, output_dir / "host_build")
    android_build = build_android_probe(
        root, ndk, output_dir / "android_build/nf_p8cn_probe"
    )
    host_runs = [
        run_host_probe(
            Path(host_build["executable"]),
            output_dir / f"host_run{index}.png",
            height=height,
            width=width,
            row_partition=row_partition,
        )
        for index in (1, 2)
    ]
    android_runs = [
        _one_android_boot(
            sdk=sdk,
            avd_home=avd_home,
            avd_name=avd_name,
            port=port,
            probe=Path(android_build["executable"]),
            output_dir=output_dir,
            boot_index=index,
            height=height,
            width=width,
            row_partition=row_partition,
        )
        for index in (1, 2)
    ]
    stable_keys = (
        "facts",
        "png_bytes",
        "png_sha256",
        "shape_hwc",
        "dtype",
        "decoded_sha256",
        "icc_sha256",
        "icc_exact",
    )
    host_stable = [{key: row[key] for key in stable_keys} for row in host_runs]
    android_stable = [{key: row[key] for key in stable_keys} for row in android_runs]
    gates = {
        "host_repeat_exact": host_stable[0] == host_stable[1],
        "android_cold_boot_exact": android_stable[0] == android_stable[1],
        "host_android_exact": host_stable[0] == android_stable[0],
        "decoded_shape_exact": host_stable[0]["shape_hwc"] == [height, width, 3],
        "icc_exact": all(row["icc_exact"] for row in [*host_runs, *android_runs]),
        "host_wall_bounded": max(row["wall_seconds"] for row in host_runs)
        <= float(contract["gates"]["max_host_wall_seconds"]),
        "host_rss_bounded": max(row["peak_rss_bytes"] for row in host_runs)
        <= int(contract["gates"]["max_host_peak_rss_bytes"]),
        "android_wall_bounded": max(
            row["device_command_seconds"] for row in android_runs
        )
        <= float(contract["gates"]["max_android_command_seconds"]),
        "device_identity_exact": all(
            row["device"]["abi"] == "x86_64" and row["device"]["api"] == "34"
            for row in android_runs
        ),
        "cleanup_exact": all(
            not row["probe_process_survived"] and not row["emulator_process_survived"]
            for row in android_runs
        ),
    }
    automatic_pass = all(gates.values())
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_sha256": {
            path: sha256_file(root / path) for path in (*SOURCE_PATHS, PROBE_SOURCE)
        },
        "host_executable_sha256": host_build["executable_sha256"],
        "android_executable_sha256": android_build["executable_sha256"],
        "host_run": host_stable[0],
        "android_run": android_stable[0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cn_android_thomas_rgb16_png_scale_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": automatic_pass,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "host_build": host_build,
        "android_build": android_build,
        "host_runs": host_runs,
        "android_runs": android_runs,
    }


__all__ = [
    "NativeThomasRgb16PngScaleError",
    "build_android_probe",
    "build_msvc_probe",
    "evaluate",
    "inspect_png",
    "parse_facts",
    "run_host_probe",
]
