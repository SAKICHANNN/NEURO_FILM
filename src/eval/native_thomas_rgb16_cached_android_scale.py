"""Exact 12MP host/Android audit for the cached parallel Thomas RGB16 core."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.native_thomas_rgb16_png_android_runtime import (
    NativeThomasRgb16PngAndroidRuntimeError,
    _adb,
    _android_env,
    _finish_owned_emulator_processes,
    _owned_emulator_processes,
    _wait_for_boot,
)

SOURCE_PATHS = (
    "native/film_physics/nf_granularity_amplitude_f32_v1.c",
    "native/film_physics/nf_thomas_field_f32_v1.c",
    "native/film_physics/nf_thomas_rows_f32_v1.c",
    "native/film_physics/nf_neumaier_f32_v1.c",
    "native/film_physics/nf_neutral_gauge_f32_v1.c",
    "native/reference_srgb_oetf_quantize_v1.c",
    "native/film_physics/nf_thomas_rgb16_f32_v1.c",
    "native/film_physics/nf_thomas_rgb16_cached_f32_v1.c",
    "native/film_physics/nf_thomas_rgb16_cached_scale_probe_v1.c",
)

_FACT_PATTERN = re.compile(
    r"^mode=(?P<mode>legacy|cached|cached_output3) status=(?P<status>\d+) "
    r"height=(?P<height>\d+) width=(?P<width>\d+) rows=(?P<rows>\d+) "
    r"parallel=(?P<parallel>\d+) values=(?P<values>\d+) "
    r"calls=(?P<calls>\d+) workspace=(?P<workspace>\d+) "
    r"means=(?P<means>[^ ]+) invalid_status=(?P<invalid_status>\d+) "
    r"invalid_calls=(?P<invalid_calls>\d+) invalid_means=(?P<invalid_means>.+)$"
)


class NativeThomasRgb16CachedScaleError(RuntimeError):
    """Raised when the cached-scale contract or execution fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _run(command: list[str], *, cwd: Path, timeout: float = 240.0) -> str:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, check=False, timeout=timeout
    )
    output = (completed.stdout + completed.stderr).decode(errors="replace")
    if completed.returncode != 0:
        raise NativeThomasRgb16CachedScaleError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output.strip()


def _sources(root: Path) -> list[Path]:
    return [root / path for path in SOURCE_PATHS]


def build_msvc_probe(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = (output_dir / "nf_p8co_probe.exe").resolve()
    sources = " ".join(f'"{path.resolve()}"' for path in _sources(root))
    batch = output_dir / "build_p8co.bat"
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
        raise NativeThomasRgb16CachedScaleError("Android NDK clang missing")
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
            "-pthread",
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
        raise NativeThomasRgb16CachedScaleError("cached probe stdout drift")
    row = match.groupdict()
    facts: dict[str, Any] = {"mode": row["mode"]}
    facts.update(
        {
            name: int(row[name])
            for name in (
                "status",
                "height",
                "width",
                "rows",
                "parallel",
                "values",
                "calls",
                "workspace",
                "invalid_status",
                "invalid_calls",
            )
        }
    )
    return facts


def run_host_probe(
    executable: Path,
    output: Path,
    *,
    mode: str,
    height: int,
    width: int,
    row_partition: int,
    parallel_layers: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [
            str(executable),
            mode,
            str(output),
            str(height),
            str(width),
            str(row_partition),
            str(parallel_layers),
        ],
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
        raise NativeThomasRgb16CachedScaleError(stderr.decode(errors="replace"))
    return {
        "wall_seconds": wall,
        "peak_rss_bytes": peak,
        "raw_bytes": output.stat().st_size,
        "raw_sha256": sha256_file(output),
        "facts": parse_facts(stdout.decode(errors="replace")),
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
    modes: tuple[tuple[str, str, int], ...] | None = None,
) -> dict[str, Any]:
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    env = _android_env(sdk, avd_home)
    serial = f"emulator-{port}"
    stdout_path = output_dir / f"boot{boot_index}_emulator.stdout.log"
    stderr_path = output_dir / f"boot{boot_index}_emulator.stderr.log"
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] | None = None
    remote_probe = "/data/local/tmp/nf_p8co_probe"
    if modes is None:
        modes = (
            ("legacy", "legacy", 1),
            ("cached1", "cached", 1),
            ("cached3a", "cached", 3),
            ("cached3b", "cached", 3),
        )
    remote_outputs = [
        f"/data/local/tmp/nf_p8co_b{boot_index}_{tag}.raw" for tag, _, _ in modes
    ]
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
            _adb(adb, serial, "push", str(probe), remote_probe, env=env)
            _adb(adb, serial, "shell", "chmod", "755", remote_probe, env=env)
            runs = []
            for remote_output, (tag, mode, parallel) in zip(remote_outputs, modes):
                run_started = time.monotonic()
                stdout = _adb(
                    adb,
                    serial,
                    "shell",
                    remote_probe,
                    mode,
                    remote_output,
                    str(height),
                    str(width),
                    str(row_partition),
                    str(parallel),
                    env=env,
                    timeout=120.0,
                )
                device_seconds = time.monotonic() - run_started
                raw_sha256 = _adb(
                    adb, serial, "shell", "sha256sum", remote_output, env=env
                ).split()[0]
                raw_bytes = int(
                    _adb(
                        adb,
                        serial,
                        "shell",
                        "stat",
                        "-c",
                        "%s",
                        remote_output,
                        env=env,
                    )
                )
                runs.append(
                    {
                        "tag": tag,
                        "device_command_seconds": device_seconds,
                        "raw_bytes": raw_bytes,
                        "raw_sha256": raw_sha256,
                        "facts": parse_facts(stdout),
                    }
                )
            listing = _adb(adb, serial, "shell", "ps", "-A", env=env)
            result = {
                "boot_seconds": boot_seconds,
                "device": {
                    "abi": _adb(
                        adb, serial, "shell", "getprop", "ro.product.cpu.abi", env=env
                    ),
                    "api": _adb(
                        adb,
                        serial,
                        "shell",
                        "getprop",
                        "ro.build.version.sdk",
                        env=env,
                    ),
                    "model": _adb(
                        adb, serial, "shell", "getprop", "ro.product.model", env=env
                    ),
                    "build_fingerprint": _adb(
                        adb,
                        serial,
                        "shell",
                        "getprop",
                        "ro.build.fingerprint",
                        env=env,
                    ),
                },
                "runs": runs,
                "probe_process_survived": "nf_p8co_probe" in listing,
            }
        finally:
            try:
                _adb(
                    adb,
                    serial,
                    "shell",
                    "rm",
                    "-f",
                    remote_probe,
                    *remote_outputs,
                    env=env,
                    timeout=15.0,
                )
            except (NativeThomasRgb16PngAndroidRuntimeError, subprocess.TimeoutExpired):
                pass
            try:
                _adb(adb, serial, "emu", "kill", env=env, timeout=15.0)
            except (NativeThomasRgb16PngAndroidRuntimeError, subprocess.TimeoutExpired):
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
        raise NativeThomasRgb16CachedScaleError("Android scale run produced no result")
    result["emulator_process_survived"] = bool(
        _owned_emulator_processes(emulator_exe, avd_name, port)
    )
    return result


def _stable_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_bytes": row["raw_bytes"],
        "raw_sha256": row["raw_sha256"],
        "facts": row["facts"],
    }


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
        != "neuro_film.u6_p8co_cached_parallel_thomas_rgb16_contract.v1"
    ):
        raise NativeThomasRgb16CachedScaleError("P8CO contract drift")
    parent = contract["parent"]
    if sha256_file(root / parent["path"]) != parent["sha256"]:
        raise NativeThomasRgb16CachedScaleError("P8CN parent evidence drift")
    fixture = contract["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    row_partition = int(fixture["row_partition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    host_build = build_msvc_probe(root, output_dir / "host_build")
    android_build = build_android_probe(
        root, ndk, output_dir / "android_build/nf_p8co_probe"
    )
    modes = (
        ("legacy", "legacy", 1),
        ("cached1", "cached", 1),
        ("cached3a", "cached", 3),
        ("cached3b", "cached", 3),
    )
    host_runs = [
        {
            "tag": tag,
            **run_host_probe(
                Path(host_build["executable"]),
                output_dir / f"host_{tag}.raw",
                mode=mode,
                height=height,
                width=width,
                row_partition=row_partition,
                parallel_layers=parallel,
            ),
        }
        for tag, mode, parallel in modes
    ]
    android_boots = [
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
    host_stable = [_stable_run(row) for row in host_runs]
    android_stable = [
        [_stable_run(row) for row in boot["runs"]] for boot in android_boots
    ]
    raw_identities = {row["raw_sha256"] for row in host_runs for _ in (0,)} | {
        row["raw_sha256"] for boot in android_boots for row in boot["runs"]
    }
    cached_parallel_android = [
        row
        for boot in android_boots
        for row in boot["runs"]
        if row["facts"]["parallel"] == 3
    ]
    cached_workspaces = [
        row["facts"]["workspace"]
        for row in host_runs
        if row["facts"]["mode"] == "cached"
    ]
    gates = {
        "old_cached_rgb16_byte_exact": len(raw_identities) == 1,
        "cached_single_parallel_byte_exact": all(
            row["raw_sha256"] == host_runs[0]["raw_sha256"] for row in host_runs[1:]
        ),
        "host_repeat_exact": host_stable[2] == host_stable[3],
        "android_cold_boot_repeat_exact": android_stable[0] == android_stable[1],
        "host_android_exact": host_stable == android_stable[0],
        "android_parallel_wall_bounded": max(
            row["device_command_seconds"] for row in cached_parallel_android
        )
        <= float(contract["gates"]["max_android_parallel_command_seconds"]),
        "cached_workspace_bounded": max(cached_workspaces)
        <= int(contract["gates"]["max_cached_workspace_bytes"]),
        "invalid_input_zero_sink_calls": all(
            row["facts"]["invalid_status"] == 3 and row["facts"]["invalid_calls"] == 0
            for row in host_runs
        )
        and all(
            row["facts"]["invalid_status"] == 3 and row["facts"]["invalid_calls"] == 0
            for boot in android_boots
            for row in boot["runs"]
        ),
        "device_identity_exact": all(
            boot["device"]["abi"] == "x86_64" and boot["device"]["api"] == "34"
            for boot in android_boots
        ),
        "cleanup_exact": all(
            not boot["probe_process_survived"] and not boot["emulator_process_survived"]
            for boot in android_boots
        ),
    }
    automatic_pass = all(gates.values())
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "host_executable_sha256": host_build["executable_sha256"],
        "android_executable_sha256": android_build["executable_sha256"],
        "host_runs": host_stable,
        "android_runs": android_stable[0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8co_cached_parallel_thomas_rgb16_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": automatic_pass,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "host_build": host_build,
        "android_build": android_build,
        "host_observations": host_runs,
        "android_observations": android_boots,
    }


__all__ = [
    "NativeThomasRgb16CachedScaleError",
    "build_android_probe",
    "build_msvc_probe",
    "evaluate",
    "parse_facts",
    "run_host_probe",
]
