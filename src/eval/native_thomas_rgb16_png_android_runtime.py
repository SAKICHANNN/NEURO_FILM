"""Android 14 virtual-device runtime evidence for the P8CK PNG program."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_png_conformance import HEADER_PATHS, SOURCE_PATHS

PROBE_SOURCE = "native/film_physics/nf_thomas_rgb16_png_runtime_probe_v1.c"


class NativeThomasRgb16PngAndroidRuntimeError(RuntimeError):
    """Raised when build or Android runtime evidence is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    output = (completed.stdout + completed.stderr).decode(errors="replace")
    if completed.returncode != 0:
        raise NativeThomasRgb16PngAndroidRuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output.strip()


def _sources(root: Path) -> list[str]:
    return [str(root / path) for path in (*SOURCE_PATHS, PROBE_SOURCE)]


def build_host_probe(root: Path, clang: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(clang),
            "--target=x86_64-w64-windows-gnu",
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
            *_sources(root),
            "-lm",
            "-Wl,--no-insert-timestamp",
            "-o",
            str(output),
        ],
        cwd=root,
    )
    return {"bytes": output.stat().st_size, "sha256": sha256_file(output)}


def build_android_probe(root: Path, ndk: Path, output: Path) -> dict[str, Any]:
    clang = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    if not clang.is_file():
        raise NativeThomasRgb16PngAndroidRuntimeError("Android NDK clang missing")
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
            *_sources(root),
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
    return {"bytes": output.stat().st_size, "sha256": sha256_file(output)}


def run_host_probe(executable: Path, png: Path) -> dict[str, Any]:
    png.parent.mkdir(parents=True, exist_ok=True)
    stdout = _run([str(executable), str(png)], cwd=executable.parent, timeout=60.0)
    return {
        "stdout": stdout,
        "png_bytes": png.stat().st_size,
        "png_sha256": sha256_file(png),
    }


def _android_env(sdk: Path, avd_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "ANDROID_SDK_ROOT": str(sdk),
            "ANDROID_HOME": str(sdk),
            "ANDROID_AVD_HOME": str(avd_home),
        }
    )
    return env


def _adb(
    adb: Path,
    serial: str,
    *arguments: str,
    env: dict[str, str],
    timeout: float = 60.0,
) -> str:
    return _run(
        [str(adb), "-s", serial, *arguments],
        cwd=adb.parent,
        env=env,
        timeout=timeout,
    )


def _wait_for_boot(
    adb: Path,
    serial: str,
    emulator: subprocess.Popen[bytes],
    *,
    env: dict[str, str],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if emulator.poll() is not None:
            raise NativeThomasRgb16PngAndroidRuntimeError(
                f"emulator exited before boot: {emulator.returncode}"
            )
        try:
            value = _adb(
                adb,
                serial,
                "shell",
                "getprop",
                "sys.boot_completed",
                env=env,
                timeout=10.0,
            )
        except (NativeThomasRgb16PngAndroidRuntimeError, subprocess.TimeoutExpired):
            value = ""
        if value.strip() == "1":
            return
        time.sleep(1.0)
    raise NativeThomasRgb16PngAndroidRuntimeError("Android emulator boot timed out")


def _owned_emulator_processes(
    emulator_exe: Path, avd_name: str, port: int
) -> list[psutil.Process]:
    expected_executable = os.path.normcase(str(emulator_exe.resolve()))
    matches: list[psutil.Process] = []
    for process in psutil.process_iter(["exe", "cmdline"]):
        try:
            executable = process.info["exe"]
            command = process.info["cmdline"] or []
            if (
                executable
                and os.path.normcase(str(Path(executable).resolve()))
                == expected_executable
                and avd_name in command
                and str(port) in command
            ):
                matches.append(process)
        except (OSError, psutil.Error):
            continue
    return matches


def _finish_owned_emulator_processes(
    emulator_exe: Path, avd_name: str, port: int
) -> bool:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        matches = _owned_emulator_processes(emulator_exe, avd_name, port)
        if not matches:
            return False
        time.sleep(0.25)
    matches = _owned_emulator_processes(emulator_exe, avd_name, port)
    for process in matches:
        process.terminate()
    _, alive = psutil.wait_procs(matches, timeout=5.0)
    for process in alive:
        process.kill()
    psutil.wait_procs(alive, timeout=5.0)
    if _owned_emulator_processes(emulator_exe, avd_name, port):
        raise NativeThomasRgb16PngAndroidRuntimeError(
            "owned emulator process survived exact cleanup"
        )
    return bool(matches)


def _one_boot(
    *,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    serial: str,
    port: int,
    android_probe: Path,
    output_dir: Path,
    boot_index: int,
    processes: int,
) -> dict[str, Any]:
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    if not emulator_exe.is_file() or not adb.is_file():
        raise NativeThomasRgb16PngAndroidRuntimeError("Android runtime tools missing")
    env = _android_env(sdk, avd_home)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"boot{boot_index}_emulator.stdout.log"
    stderr_path = output_dir / f"boot{boot_index}_emulator.stderr.log"
    started = time.monotonic()
    boot_result: dict[str, Any]
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        emulator = subprocess.Popen(
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
            _wait_for_boot(adb, serial, emulator, env=env, timeout=180.0)
            boot_seconds = time.monotonic() - started
            device = {
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
            }
            remote_probe = "/data/local/tmp/nf_p8cm_probe"
            _adb(adb, serial, "push", str(android_probe), remote_probe, env=env)
            _adb(adb, serial, "shell", "chmod", "755", remote_probe, env=env)
            runs: list[dict[str, Any]] = []
            for process_index in range(1, processes + 1):
                remote_png = f"/data/local/tmp/p8cm_b{boot_index}_p{process_index}.png"
                stdout = _adb(
                    adb,
                    serial,
                    "shell",
                    remote_probe,
                    remote_png,
                    env=env,
                    timeout=90.0,
                )
                local_png = output_dir / f"boot{boot_index}_process{process_index}.png"
                _adb(adb, serial, "pull", remote_png, str(local_png), env=env)
                runs.append(
                    {
                        "stdout": stdout,
                        "png_bytes": local_png.stat().st_size,
                        "png_sha256": sha256_file(local_png),
                    }
                )
            process_listing = _adb(adb, serial, "shell", "ps", "-A", env=env)
            boot_result = {
                "boot_seconds": boot_seconds,
                "device": device,
                "runs": runs,
                "probe_process_survived": "nf_p8cm_probe" in process_listing,
            }
        finally:
            try:
                _adb(
                    adb,
                    serial,
                    "shell",
                    "rm",
                    "-f",
                    "/data/local/tmp/nf_p8cm_probe",
                    "/data/local/tmp/p8cm_b1_p1.png",
                    "/data/local/tmp/p8cm_b1_p2.png",
                    "/data/local/tmp/p8cm_b2_p1.png",
                    "/data/local/tmp/p8cm_b2_p2.png",
                    env=env,
                    timeout=15.0,
                )
            except (NativeThomasRgb16PngAndroidRuntimeError, subprocess.TimeoutExpired):
                pass
            try:
                _adb(adb, serial, "emu", "kill", env=env, timeout=15.0)
            except (NativeThomasRgb16PngAndroidRuntimeError, subprocess.TimeoutExpired):
                pass
            try:
                emulator.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                emulator.terminate()
                try:
                    emulator.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    emulator.kill()
                    emulator.wait(timeout=15.0)
    forced_cleanup = _finish_owned_emulator_processes(
        emulator_exe, avd_name, port
    )
    boot_result["emulator_force_cleanup_required"] = forced_cleanup
    boot_result["emulator_process_survived"] = bool(
        _owned_emulator_processes(emulator_exe, avd_name, port)
    )
    return boot_result


def evaluate(
    *,
    root: Path,
    contract_path: Path,
    ndk: Path,
    host_clang: Path,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    output_dir: Path,
    port: int = 5580,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("schema") != (
        "neuro_film.u6_p8cm_android_thomas_rgb16_png_runtime_contract.v1"
    ):
        raise NativeThomasRgb16PngAndroidRuntimeError("P8CM contract drift")
    parent = contract["parent"]
    parent_path = root / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise NativeThomasRgb16PngAndroidRuntimeError("P8CL evidence hash drift")
    if json.loads(parent_path.read_bytes()).get("decision") != parent[
        "required_decision"
    ]:
        raise NativeThomasRgb16PngAndroidRuntimeError("P8CL decision drift")
    image_properties = sdk / "system-images/android-34/google_apis/x86_64/source.properties"
    if sha256_file(image_properties) != contract["runtime"][
        "system_image_source_properties_sha256"
    ]:
        raise NativeThomasRgb16PngAndroidRuntimeError("system image identity drift")

    output_dir.mkdir(parents=True, exist_ok=True)
    host1 = output_dir / "build/host1/nf_p8cm_probe.exe"
    host2 = output_dir / "build/host2/nf_p8cm_probe.exe"
    android1 = output_dir / "build/android1/nf_p8cm_probe"
    android2 = output_dir / "build/android2/nf_p8cm_probe"
    host_build1 = build_host_probe(root, host_clang, host1)
    host_build2 = build_host_probe(root, host_clang, host2)
    android_build1 = build_android_probe(root, ndk, android1)
    android_build2 = build_android_probe(root, ndk, android2)
    host_run1 = run_host_probe(host1, output_dir / "host_process1.png")
    host_run2 = run_host_probe(host1, output_dir / "host_process2.png")
    runtime = contract["runtime"]
    boot_rows = [
        _one_boot(
            sdk=sdk,
            avd_home=avd_home,
            avd_name=avd_name,
            serial=f"emulator-{port}",
            port=port,
            android_probe=android1,
            output_dir=output_dir,
            boot_index=index,
            processes=int(runtime["fresh_processes_per_boot"]),
        )
        for index in range(1, int(runtime["require_wipe_data_cold_boots"]) + 1)
    ]
    all_device_runs = [row for boot in boot_rows for row in boot["runs"]]
    host_exact = host_run1 == host_run2
    target_exact = all(row == all_device_runs[0] for row in all_device_runs)
    cross_exact = all(row == host_run1 for row in all_device_runs)
    device_exact = all(
        boot["device"]["abi"] == runtime["abi"]
        and boot["device"]["api"] == str(runtime["api_level"])
        and not boot["probe_process_survived"]
        and not boot["emulator_process_survived"]
        for boot in boot_rows
    )
    build_exact = (
        host_build1 == host_build2 and android_build1 == android_build2
    )
    automatic_pass = host_exact and target_exact and cross_exact and device_exact and build_exact
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_sha256": {
            path: sha256_file(root / path)
            for path in (*SOURCE_PATHS, PROBE_SOURCE)
        },
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "host_build": host_build1,
        "android_build": android_build1,
        "host_run": host_run1,
        "device": boot_rows[0]["device"],
        "cold_boot_count": len(boot_rows),
        "processes_per_boot": int(runtime["fresh_processes_per_boot"]),
        "host_build_exact": host_build1 == host_build2,
        "android_build_exact": android_build1 == android_build2,
        "device_run_exact": target_exact,
        "host_android_exact": cross_exact,
        "failure_atomicity_exact": "invalid_status=3 invalid_calls=0" in host_run1[
            "stdout"
        ]
        and "failed_status=4 failed_calls=2" in host_run1["stdout"],
        "zero_surviving_probe_processes": all(
            not boot["probe_process_survived"] for boot in boot_rows
        ),
        "zero_surviving_emulator_processes": all(
            not boot["emulator_process_survived"] for boot in boot_rows
        ),
        "decision": (
            contract["decision_if_pass"]
            if automatic_pass
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cm_android_thomas_rgb16_png_runtime_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": automatic_pass,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "boot_rows": boot_rows,
    }


__all__ = [
    "NativeThomasRgb16PngAndroidRuntimeError",
    "build_android_probe",
    "build_host_probe",
    "evaluate",
    "run_host_probe",
]
