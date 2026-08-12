from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_cloud_attenuation_android_runtime import _cleanup_owned_launchers
from src.eval.native_thomas_rgb16_png_android_runtime import (
    _android_env,
    _finish_owned_emulator_processes,
    _run,
    _wait_for_boot,
)

SOURCES = [
    ROOT / "native/film_physics/nf_gaussian_rgb_f32_v1.c",
    ROOT / "native/film_physics/nf_gaussian_row_window_f32_v1.c",
    ROOT / "native/film_physics/nf_gaussian_row_window_runtime_probe_v1.c",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(compiler: Path, target: str, output: Path, android: bool) -> None:
    command = [
        str(compiler), f"--target={target}", "-std=c11", "-O2", "-Wall",
        "-Wextra", "-Werror", "-ffp-model=strict", *map(str, SOURCES),
    ]
    command += ["-fPIE", "-pie", "-Wl,--build-id=none", "-lm"] if android else ["-Wl,--no-insert-timestamp"]
    _run([*command, "-o", str(output)], cwd=ROOT)


def evaluate(
    contract_path: Path,
    ndk: Path,
    host_clang: Path,
    sdk: Path,
    avd_home: Path,
    output: Path,
) -> dict:
    contract = json.loads(contract_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    host = output / "host.exe"
    android = output / "android-probe"
    _build(host_clang, "x86_64-w64-windows-gnu", host, False)
    _build(
        ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe",
        "x86_64-linux-android34",
        android,
        True,
    )
    host_output = output / "host.f32"
    host_stdout = _run([str(host), str(host_output)], cwd=output)
    env = _android_env(sdk, avd_home)
    emulator = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    port = int(contract["runtime"]["port"])
    serial = f"emulator-{port}"
    rows = []
    for boot in range(int(contract["runtime"]["cold_boots"])):
        process = subprocess.Popen(
            [
                str(emulator), "-avd", contract["runtime"]["avd_name"],
                "-port", str(port), "-no-window", "-no-audio", "-no-boot-anim",
                "-no-snapshot", "-wipe-data", "-gpu", "swiftshader_indirect",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_for_boot(adb, serial, process, env=env, timeout=180.0)
            remote = "/data/local/tmp/nf_p4fp_probe"
            _run([str(adb), "-s", serial, "push", str(android), remote], cwd=output, env=env)
            _run([str(adb), "-s", serial, "shell", "chmod", "755", remote], cwd=output, env=env)
            for run in range(int(contract["runtime"]["fresh_processes_per_boot"])):
                remote_output = f"/data/local/tmp/p4fp-{boot}-{run}.f32"
                stdout = _run([str(adb), "-s", serial, "shell", remote, remote_output], cwd=output, env=env)
                local = output / f"android-{boot}-{run}.f32"
                _run([str(adb), "-s", serial, "pull", remote_output, str(local)], cwd=output, env=env)
                rows.append({"boot": boot, "run": run, "stdout": stdout, "sha256": _sha(local)})
        finally:
            try:
                _run([str(adb), "-s", serial, "emu", "kill"], cwd=output, env=env, timeout=15)
            except (RuntimeError, subprocess.SubprocessError):
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _finish_owned_emulator_processes(emulator, contract["runtime"]["avd_name"], port)
            if not _cleanup_owned_launchers(emulator, contract["runtime"]["avd_name"], port):
                raise RuntimeError("owned emulator survived cleanup")
    host_sha = _sha(host_output)
    gates = {
        "host_android": all(row["sha256"] == host_sha for row in rows),
        "repeat": len({row["sha256"] for row in rows}) == 1,
        "atomic": all("exact=1 invalid=1" in row["stdout"] for row in rows) and "exact=1 invalid=1" in host_stdout,
        "cleanup": _cleanup_owned_launchers(emulator, contract["runtime"]["avd_name"], port),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "source_sha256": [_sha(path) for path in SOURCES],
        "host_output_sha256": host_sha,
        "android_runs": rows,
        "gates": gates,
        "decision": contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4fp_gaussian_row_window_android_runtime.v1",
        "automatic_pass": all(gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        ROOT / "configs/u6_p4fp_gaussian_row_window_android_runtime_v1.json",
        Path(r"D:\nf-019f4b76-android\android-ndk-r27d"),
        ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
        Path(r"D:\nf-019f4b76-android\runtime-sdk"),
        Path(r"D:\nf-019f4b76-android\avd"),
        args.work,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
