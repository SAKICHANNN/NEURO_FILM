"""12MP host/Android audit for the parallel Thomas RGB16 PNG stream."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_png_android_runtime import (
    NativeThomasRgb16PngAndroidRuntimeError,
    _adb,
    _android_env,
    _finish_owned_emulator_processes,
    _owned_emulator_processes,
    _wait_for_boot,
)
from src.eval.native_thomas_rgb16_png_android_scale import (
    build_android_probe,
    build_msvc_probe,
    inspect_png,
)
from src.eval.native_thomas_rgb16_png_conformance import SOURCE_PATHS

_FACT_PATTERN = re.compile(
    r"^mode=parallel status=(?P<status>\d+) height=(?P<height>\d+) "
    r"width=(?P<width>\d+) rows=(?P<rows>\d+) bytes=(?P<bytes>\d+) "
    r"calls=(?P<calls>\d+) workspace=(?P<workspace>\d+) means=(?P<means>[^ ]+) "
    r"invalid_status=(?P<invalid_status>\d+) invalid_bytes=(?P<invalid_bytes>\d+) "
    r"invalid_calls=(?P<invalid_calls>\d+) invalid_means=(?P<invalid_means>.+)$"
)


class NativeThomasRgb16PngParallelScaleError(RuntimeError):
    """Raised when the parallel PNG contract or execution fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def parse_facts(stdout: str) -> dict[str, int]:
    match = _FACT_PATTERN.fullmatch(stdout.strip())
    if match is None:
        raise NativeThomasRgb16PngParallelScaleError("parallel PNG stdout drift")
    row = match.groupdict()
    return {
        name: int(row[name])
        for name in (
            "status",
            "height",
            "width",
            "rows",
            "bytes",
            "calls",
            "workspace",
            "invalid_status",
            "invalid_bytes",
            "invalid_calls",
        )
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
        [
            str(executable),
            "parallel",
            str(png),
            str(height),
            str(width),
            str(row_partition),
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
        raise NativeThomasRgb16PngParallelScaleError(stderr.decode(errors="replace"))
    return {
        "wall_seconds": wall,
        "peak_rss_bytes": peak,
        "facts": parse_facts(stdout.decode(errors="replace")),
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
    remote_probe = "/data/local/tmp/nf_p8cq_probe"
    remote_pngs = [
        f"/data/local/tmp/nf_p8cq_b{boot_index}_p{index}.png" for index in (1, 2)
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
            for process_index, remote_png in enumerate(remote_pngs, start=1):
                run_started = time.monotonic()
                stdout = _adb(
                    adb,
                    serial,
                    "shell",
                    remote_probe,
                    "parallel",
                    remote_png,
                    str(height),
                    str(width),
                    str(row_partition),
                    env=env,
                    timeout=120.0,
                )
                device_seconds = time.monotonic() - run_started
                local_png = (
                    output_dir / f"android_boot{boot_index}_p{process_index}.png"
                )
                _adb(
                    adb,
                    serial,
                    "pull",
                    remote_png,
                    str(local_png),
                    env=env,
                    timeout=120.0,
                )
                runs.append(
                    {
                        "device_command_seconds": device_seconds,
                        "facts": parse_facts(stdout),
                        **inspect_png(local_png),
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
                "probe_process_survived": "nf_p8cq_probe" in listing,
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
                    *remote_pngs,
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
        raise NativeThomasRgb16PngParallelScaleError(
            "Android parallel PNG run produced no result"
        )
    result["emulator_process_survived"] = bool(
        _owned_emulator_processes(emulator_exe, avd_name, port)
    )
    return result


def _stable_run(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "facts",
        "png_bytes",
        "png_sha256",
        "shape_hwc",
        "dtype",
        "decoded_sha256",
        "icc_sha256",
        "icc_exact",
    )
    return {key: row[key] for key in keys}


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
        != "neuro_film.u6_p8cq_parallel_thomas_rgb16_png_contract.v1"
    ):
        raise NativeThomasRgb16PngParallelScaleError("P8CQ contract drift")
    parent = contract["parent"]
    if sha256_file(root / parent["path"]) != parent["sha256"]:
        raise NativeThomasRgb16PngParallelScaleError("P8CP parent evidence drift")
    fixture = contract["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    row_partition = int(fixture["row_partition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    host_build = build_msvc_probe(root, output_dir / "host_build")
    android_build = build_android_probe(
        root, ndk, output_dir / "android_build/nf_p8cq_probe"
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
    all_runs = [*host_runs, *(row for boot in android_boots for row in boot["runs"])]
    gates = {
        "p8cn_png_byte_exact": all(
            row["png_bytes"] == int(fixture["expected_png_bytes"])
            and row["png_sha256"] == fixture["expected_png_sha256"]
            for row in all_runs
        ),
        "decoded_rgb16_exact": all(
            row["decoded_sha256"] == fixture["expected_decoded_rgb16_sha256"]
            for row in all_runs
        ),
        "icc_exact": all(row["icc_exact"] for row in all_runs),
        "host_repeat_exact": host_stable[0] == host_stable[1],
        "android_cold_boot_repeat_exact": android_stable[0] == android_stable[1],
        "host_android_exact": all(
            host_stable[0] == row for boot in android_stable for row in boot
        ),
        "android_wall_bounded": max(
            row["device_command_seconds"]
            for boot in android_boots
            for row in boot["runs"]
        )
        <= float(contract["gates"]["max_android_command_seconds"]),
        "host_rss_bounded": max(row["peak_rss_bytes"] for row in host_runs)
        <= int(contract["gates"]["max_host_peak_rss_bytes"]),
        "workspace_bounded": all(
            row["facts"]["workspace"] <= int(contract["gates"]["max_workspace_bytes"])
            for row in all_runs
        ),
        "preoutput_failure_zero_bytes": all(
            row["facts"]["invalid_status"] == 3
            and row["facts"]["invalid_bytes"] == 0
            and row["facts"]["invalid_calls"] == 0
            for row in all_runs
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
        "host_run": host_stable[0],
        "android_run": android_stable[0][0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cq_parallel_thomas_rgb16_png_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": automatic_pass,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "host_build": host_build,
        "android_build": android_build,
        "host_observations": host_runs,
        "android_observations": android_boots,
    }


__all__ = ["evaluate", "parse_facts", "run_host_probe"]
