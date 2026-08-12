"""P4HV host/Android runtime and mobile target-build evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_png_android_runtime import (
    _adb,
    _android_env,
    _finish_owned_emulator_processes,
    _run,
    _wait_for_boot,
)

SCHEMA = "neuro-film.u6-p4hv-portable-native-spatial-density-runtime-contract.v1"
SOURCES = (
    "native/film_physics/nf_thomas_field_f32_v1.c",
    "native/film_physics/nf_histogram_copula_f32_v1.c",
    "native/film_physics/nf_gamma_density_fast_f64_v1.c",
    "native/film_physics/nf_spatial_density_runtime_probe_v1.c",
)


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _validate(root: Path, contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HV contract")
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HV parent drift: {path}")
        if (
            json.loads(path.read_text("utf-8")).get("decision")
            != binding["required_decision"]
        ):
            raise ValueError(f"P4HV parent decision drift: {path}")


def _compile(
    root: Path, clang: Path, output: Path, target: str, *, executable: bool
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(clang),
        f"--target={target}",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-model=strict",
        "-I",
        str(root / "native/film_physics"),
        *[str(root / source) for source in SOURCES],
    ]
    if executable:
        command += ["-lm", "-Wl,--build-id=none", "-o", str(output)]
    else:
        raise ValueError("P4HV multi-source object compile requires per-source mode")
    _run(command, cwd=root, timeout=180.0)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
    }


def _compile_apple_objects(
    root: Path, clang: Path, output_dir: Path, target: str
) -> list[dict[str, Any]]:
    rows = []
    for source in SOURCES:
        source_path = root / source
        output = output_dir / (source_path.stem + ".o")
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(clang),
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffp-model=strict",
                "-I",
                str(root / "native/film_physics"),
                "-c",
                str(source_path),
                "-o",
                str(output),
            ],
            cwd=root,
            timeout=120.0,
        )
        rows.append(
            {
                "source": source,
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
            }
        )
    return rows


def _boot(
    *, sdk: Path, avd_home: Path, avd_name: str, port: int, probe: Path, index: int
) -> dict[str, Any]:
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    env = _android_env(sdk, avd_home)
    serial = f"emulator-{port}"
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    started = time.monotonic()
    remote = "/data/local/tmp/nf_p4hv_probe"
    try:
        _wait_for_boot(adb, serial, emulator, env=env, timeout=180.0)
        boot_seconds = time.monotonic() - started
        _adb(adb, serial, "push", str(probe), remote, env=env)
        _adb(adb, serial, "shell", "chmod", "755", remote, env=env)
        outputs = [
            _adb(adb, serial, "shell", remote, env=env, timeout=90.0).strip()
            for _ in range(2)
        ]
        device = {
            "abi": _adb(adb, serial, "shell", "getprop", "ro.product.cpu.abi", env=env),
            "api": _adb(
                adb, serial, "shell", "getprop", "ro.build.version.sdk", env=env
            ),
        }
        _adb(adb, serial, "shell", "rm", "-f", remote, env=env)
        _adb(adb, serial, "emu", "kill", env=env)
        emulator.wait(timeout=30.0)
        return {
            "boot_index": index,
            "boot_seconds": boot_seconds,
            "outputs": outputs,
            "device": device,
        }
    finally:
        if emulator.poll() is None:
            emulator.terminate()
            try:
                emulator.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                emulator.kill()
                emulator.wait(timeout=10.0)
        _finish_owned_emulator_processes(emulator_exe, avd_name, port)


def evaluate(
    *,
    root: Path,
    contract: dict[str, Any],
    output_dir: Path,
    host_clang: Path,
    ndk: Path,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    port: int,
) -> dict[str, Any]:
    _validate(root, contract)
    host = _compile(
        root,
        host_clang,
        output_dir / "host/probe.exe",
        "x86_64-w64-windows-gnu",
        executable=True,
    )
    host_stdout = _run([host["path"]], cwd=output_dir, timeout=60.0).strip()
    ndk_clang = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    android_x86 = _compile(
        root,
        ndk_clang,
        output_dir / "android/x86_64/probe",
        "x86_64-linux-android21",
        executable=True,
    )
    android_arm64 = _compile(
        root,
        ndk_clang,
        output_dir / "android/arm64-v8a/probe",
        "aarch64-linux-android21",
        executable=True,
    )
    apple = {
        "macos_arm64": _compile_apple_objects(
            root, host_clang, output_dir / "apple/macos", "arm64-apple-macos13"
        ),
        "ios_arm64": _compile_apple_objects(
            root, host_clang, output_dir / "apple/ios", "arm64-apple-ios15"
        ),
    }
    boots = [
        _boot(
            sdk=sdk,
            avd_home=avd_home,
            avd_name=avd_name,
            port=port,
            probe=Path(android_x86["path"]),
            index=index,
        )
        for index in range(1, 3)
    ]
    outputs = [value for boot in boots for value in boot["outputs"]]
    checks = {
        "host_android_stdout_exact": all(value == host_stdout for value in outputs),
        "android_fresh_process_exact": len(set(outputs)) == 1,
        "android_device": all(
            boot["device"] == {"abi": "x86_64", "api": "34"} for boot in boots
        ),
        "android_arm64_link": Path(android_arm64["path"]).is_file(),
        "macos_objects": len(apple["macos_arm64"]) == len(SOURCES),
        "ios_objects": len(apple["ios_arm64"]) == len(SOURCES),
        "failure_atomic": "atomic=1" in host_stdout
        and all("atomic=1" in value for value in outputs),
    }
    stable = {
        "contract_sha256": hashlib.sha256(_bytes(contract)).hexdigest(),
        "source_sha256": {source: sha256_file(root / source) for source in SOURCES},
        "host_stdout": host_stdout,
        "android_outputs": outputs,
        "android_device": boots[0]["device"],
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_bytes(stable)).hexdigest(),
        "builds": {
            "host": host,
            "android_x86_64": android_x86,
            "android_arm64": android_arm64,
            "apple": apple,
        },
        "boot_seconds": [boot["boot_seconds"] for boot in boots],
    }


__all__ = ["evaluate"]
