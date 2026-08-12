"""Android virtual runtime for the P4DZ attenuation kernel."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import psutil

from src.eval.native_thomas_rgb16_png_android_runtime import (
    _android_env,
    _finish_owned_emulator_processes,
    _run,
    _wait_for_boot,
)


def _cleanup_owned_launchers(emulator_exe: Path, avd_name: str, port: int) -> bool:
    expected = os.path.normcase(str(emulator_exe.resolve()))
    matches = []
    for process in psutil.process_iter(["exe", "cmdline"]):
        try:
            executable = process.info["exe"]
            command = " ".join(process.info["cmdline"] or [])
            if (
                executable
                and os.path.normcase(str(Path(executable).resolve())) == expected
                and avd_name in command
                and str(port) in command
            ):
                matches.append(process)
        except (OSError, psutil.Error):
            continue
    for process in matches:
        process.terminate()
    _, alive = psutil.wait_procs(matches, timeout=5.0)
    for process in alive:
        process.kill()
    psutil.wait_procs(alive, timeout=5.0)
    return not any(process.is_running() for process in matches)


def build(
    root: Path, compiler: Path, target: str, output: Path, *, android: bool
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler),
        f"--target={target}",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-model=strict",
        "-I",
        str(root / "native/film_physics"),
        str(root / "native/film_physics/nf_cloud_attenuation_f32_v1.c"),
        str(root / "native/film_physics/nf_cloud_attenuation_runtime_probe_v1.c"),
        "-lm",
        "-o",
        str(output),
    ]
    if android:
        command[8:8] = ["-fPIE", "-pie", "-Wl,--build-id=none"]
    else:
        command.extend(["-Wl,--no-insert-timestamp"])
    _run(command, cwd=root)


def evaluate(
    root: Path,
    contract_path: Path,
    ndk: Path,
    host_clang: Path,
    sdk: Path,
    avd_home: Path,
    output: Path,
    port: int = 5582,
) -> dict:
    contract = json.loads(contract_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    host = output / "host.exe"
    android = output / "android-probe"
    build(root, host_clang, "x86_64-w64-windows-gnu", host, android=False)
    build(
        root,
        ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe",
        "x86_64-linux-android21",
        android,
        android=True,
    )
    host_density = output / "host-density.f32"
    host_transmittance = output / "host-transmittance.f32"
    host_stdout = _run(
        [str(host), str(host_density), str(host_transmittance)], cwd=output
    )
    env = _android_env(sdk, avd_home)
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    serial = f"emulator-{port}"
    boots = []
    for boot in range(2):
        process = subprocess.Popen(
            [
                str(emulator_exe),
                "-avd",
                contract["runtime"]["avd_name"],
                "-port",
                str(port),
                "-no-window",
                "-no-audio",
                "-no-boot-anim",
                "-no-snapshot",
                "-wipe-data",
                "-gpu",
                "swiftshader_indirect",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_for_boot(adb, serial, process, env=env, timeout=180.0)
            _run(
                [
                    str(adb),
                    "-s",
                    serial,
                    "push",
                    str(android),
                    "/data/local/tmp/nf_p4eb_probe",
                ],
                cwd=output,
                env=env,
            )
            _run(
                [
                    str(adb),
                    "-s",
                    serial,
                    "shell",
                    "chmod",
                    "755",
                    "/data/local/tmp/nf_p4eb_probe",
                ],
                cwd=output,
                env=env,
            )
            rows = []
            for process_index in range(2):
                remote_density = f"/data/local/tmp/p4ec-{boot}-{process_index}-d.f32"
                remote_t = f"/data/local/tmp/p4ec-{boot}-{process_index}-t.f32"
                stdout = _run(
                    [
                        str(adb),
                        "-s",
                        serial,
                        "shell",
                        "/data/local/tmp/nf_p4eb_probe",
                        remote_density,
                        remote_t,
                    ],
                    cwd=output,
                    env=env,
                )
                local_density = (
                    output / f"boot{boot}-process{process_index}-density.f32"
                )
                local_t = (
                    output / f"boot{boot}-process{process_index}-transmittance.f32"
                )
                _run(
                    [
                        str(adb),
                        "-s",
                        serial,
                        "pull",
                        remote_density,
                        str(local_density),
                    ],
                    cwd=output,
                    env=env,
                )
                _run(
                    [str(adb), "-s", serial, "pull", remote_t, str(local_t)],
                    cwd=output,
                    env=env,
                )
                rows.append(
                    {
                        "stdout": stdout,
                        "density_path": str(local_density),
                        "transmittance_path": str(local_t),
                    }
                )
            abi = _run(
                [str(adb), "-s", serial, "shell", "getprop", "ro.product.cpu.abi"],
                cwd=output,
                env=env,
            )
            boots.append({"abi": abi, "runs": rows})
        finally:
            try:
                _run(
                    [str(adb), "-s", serial, "emu", "kill"],
                    cwd=output,
                    env=env,
                    timeout=15,
                )
            except (RuntimeError, subprocess.SubprocessError):
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _finish_owned_emulator_processes(
                emulator_exe, contract["runtime"]["avd_name"], port
            )
            if not _cleanup_owned_launchers(
                emulator_exe, contract["runtime"]["avd_name"], port
            ):
                raise RuntimeError("owned emulator launcher survived cleanup")
    all_rows = [row for boot in boots for row in boot["runs"]]
    host_d = np.fromfile(host_density, dtype="<f4")
    host_t = np.fromfile(host_transmittance, dtype="<f4")
    densities = [np.fromfile(row["density_path"], dtype="<f4") for row in all_rows]
    transmittances = [
        np.fromfile(row["transmittance_path"], dtype="<f4") for row in all_rows
    ]
    density_error = max(float(np.max(np.abs(value - host_d))) for value in densities)
    gates = {
        "density_tolerance": density_error
        <= contract["gates"]["maximum_host_android_density_absolute_error"],
        "transmittance": all(np.array_equal(value, host_t) for value in transmittances),
        "repeat": all(np.array_equal(value, densities[0]) for value in densities)
        and all(np.array_equal(value, transmittances[0]) for value in transmittances),
        "atomic": all(
            "invalid_status=2 unchanged=1" in row["stdout"] for row in all_rows
        ),
        "abi": all(b["abi"] == "x86_64" for b in boots),
        "cleanup": _cleanup_owned_launchers(
            emulator_exe, contract["runtime"]["avd_name"], port
        ),
    }
    return {
        "schema": "neuro_film.u6_p4ec_native_cloud_attenuation_android_tolerance.v2",
        "automatic_pass": all(gates.values()),
        "host_stdout": host_stdout,
        "boots": boots,
        "maximum_host_android_density_absolute_error": density_error,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = ["evaluate"]
