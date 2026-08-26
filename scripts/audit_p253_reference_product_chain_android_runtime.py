"""Execute the frozen reference product-chain verifier on Android 14."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reference_chain_android import build
from src.eval.native_thomas_rgb16_png_android_runtime import (
    _finish_owned_emulator_processes,
    _wait_for_boot,
)

DEFAULT_CONTRACT = ROOT / "configs/p253_reference_product_chain_android_runtime_v1.json"


class P253RuntimeError(RuntimeError):
    """Raised when the frozen P253 runtime contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _environment(sdk: Path, avd_home: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "ANDROID_HOME": str(sdk),
            "ANDROID_SDK_ROOT": str(sdk),
            "ANDROID_AVD_HOME": str(avd_home),
            "ANDROID_EMULATOR_HOME": str(avd_home.parent),
            "ANDROID_USER_HOME": str(avd_home.parent),
        }
    )
    return environment


def _command(
    arguments: Iterable[object],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 180.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(value) for value in arguments]
    if command[0].casefold().endswith((".bat", ".cmd")):
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", *command]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise P253RuntimeError(
            f"command failed with exit {completed.returncode}: {command[0]}"
        )
    return completed


def _validate_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise P253RuntimeError(f"{label} identity mismatch")


def _validate_inputs(contract: dict[str, Any]) -> None:
    for label, item in contract["inputs"].items():
        path_value = item.get("path")
        expected = item.get("sha256")
        if path_value is not None and expected is not None:
            _validate_file(ROOT / path_value, expected, label)


def _fixture_rows(
    fixture: dict[str, Any], order: str
) -> list[tuple[str, str, str, str]]:
    rows = [
        (
            case["case_id"],
            identity["label"],
            identity["canonical_hex"],
            identity["sha256"],
        )
        for case in fixture["cases"]
        for identity in case["identities"]
    ]
    if order == "reverse":
        rows.reverse()
    elif order != "normal":
        raise P253RuntimeError("unsupported enumeration order")
    return rows


def _truth_table_rows(contract: dict[str, Any], order: str) -> list[dict[str, Any]]:
    rows = list(contract["truth_table"])
    if order == "reverse":
        rows.reverse()
    elif order != "normal":
        raise P253RuntimeError("unsupported enumeration order")
    return rows


def _adb_command(
    adb: Path,
    serial: str,
    remote: str,
    arguments: list[str],
    *,
    environment: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _command(
        [adb, "-s", serial, "shell", remote, *arguments],
        cwd=adb.parent,
        environment=environment,
        timeout=90.0,
        check=check,
    )


def _build_twice(ndk: Path, work: Path) -> dict[str, Any]:
    reports = [build(ndk, work / f"build-{index}") for index in range(2)]
    artifacts: dict[str, Any] = {}
    for abi in ("arm64-v8a", "x86_64"):
        first = reports[0]["artifacts"][abi]
        second = reports[1]["artifacts"][abi]
        artifacts[abi] = {
            "target": first["target"],
            "elf_machine": first["elf_machine"],
            "bytes": first["bytes"],
            "sha256": first["sha256"],
            "two_builds_exact": first == second,
        }
    return {
        "compiler": reports[0]["compiler"],
        "artifacts": artifacts,
        "all_exact": all(value["two_builds_exact"] for value in artifacts.values()),
        "x86_executable": work / "build-0/reference_product_chain_x86_64",
    }


def _create_avd(
    avdmanager: Path,
    avd_home: Path,
    sdk: Path,
    contract: dict[str, Any],
) -> None:
    runtime = contract["runtime"]
    avd_home.mkdir(parents=True, exist_ok=True)
    environment = _environment(sdk, avd_home)
    result = _command(
        [
            avdmanager,
            "create",
            "avd",
            "--name",
            runtime["avd_name"],
            "--package",
            runtime["system_image"],
            "--device",
            runtime["device"],
            "--force",
        ],
        cwd=avd_home,
        environment=environment,
        input_text="no\n",
        timeout=120.0,
    )
    if "Error:" in result.stdout or "Error:" in result.stderr:
        raise P253RuntimeError("AVD creation reported an error")
    config = avd_home / f"{runtime['avd_name']}.avd/config.ini"
    if not config.is_file():
        raise P253RuntimeError("owned AVD config was not created")


def _run_runtime(
    *,
    contract: dict[str, Any],
    fixture: dict[str, Any],
    sdk: Path,
    avd_home: Path,
    executable: Path,
    order: str,
    work: Path,
) -> dict[str, Any]:
    runtime = contract["runtime"]
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    environment = _environment(sdk, avd_home)
    port = int(runtime["port"])
    serial = f"emulator-{port}"
    remote = "/data/local/tmp/nf_reference_product_chain_p253"
    stdout_path = work / "emulator.stdout.log"
    stderr_path = work / "emulator.stderr.log"
    emulator: subprocess.Popen[bytes] | None = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        emulator = subprocess.Popen(
            [
                str(emulator_exe),
                "-avd",
                runtime["avd_name"],
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
                "on",
            ],
            stdout=stdout,
            stderr=stderr,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_for_boot(
                adb,
                serial,
                emulator,
                env=environment,
                timeout=240.0,
            )
            _command(
                [adb, "-s", serial, "push", executable, remote],
                cwd=work,
                environment=environment,
            )
            _command(
                [adb, "-s", serial, "shell", "chmod", "755", remote],
                cwd=work,
                environment=environment,
            )
            identities = []
            for case_id, label, canonical_hex, expected in _fixture_rows(
                fixture, order
            ):
                completed = _adb_command(
                    adb,
                    serial,
                    remote,
                    ["hash", canonical_hex],
                    environment=environment,
                )
                identities.append(
                    {
                        "case_id": case_id,
                        "label": label,
                        "expected_sha256": expected,
                        "observed_sha256": completed.stdout.strip(),
                    }
                )
            truth_table = []
            for row in _truth_table_rows(contract, order):
                completed = _adb_command(
                    adb,
                    serial,
                    remote,
                    [
                        "state",
                        str(row["numeric"]),
                        str(row["promoted"]),
                        str(row["research"]),
                    ],
                    environment=environment,
                )
                truth_table.append({**row, "observed": completed.stdout.strip()})
            first_hex = fixture["cases"][0]["identities"][0]["canonical_hex"]
            mutated_hex = first_hex[:-1] + ("1" if first_hex[-1] != "1" else "0")
            mutation = _adb_command(
                adb,
                serial,
                remote,
                ["hash", mutated_hex],
                environment=environment,
            ).stdout.strip()
            negative_commands = {
                "odd_hex": ["hash", "0"],
                "uppercase_hex": ["hash", "AA"],
                "invalid_flag": ["state", "2", "1", "0"],
                "missing_arguments": [],
            }
            negative_controls = []
            for label, arguments in negative_commands.items():
                completed = _adb_command(
                    adb,
                    serial,
                    remote,
                    arguments,
                    environment=environment,
                    check=False,
                )
                negative_controls.append(
                    {
                        "label": label,
                        "returncode_nonzero": completed.returncode != 0,
                        "stdout_empty": completed.stdout == "",
                    }
                )
            device = {
                "abi": _command(
                    [adb, "-s", serial, "shell", "getprop", "ro.product.cpu.abi"],
                    cwd=work,
                    environment=environment,
                ).stdout.strip(),
                "api": _command(
                    [adb, "-s", serial, "shell", "getprop", "ro.build.version.sdk"],
                    cwd=work,
                    environment=environment,
                ).stdout.strip(),
                "model": _command(
                    [adb, "-s", serial, "shell", "getprop", "ro.product.model"],
                    cwd=work,
                    environment=environment,
                ).stdout.strip(),
                "build_fingerprint": _command(
                    [adb, "-s", serial, "shell", "getprop", "ro.build.fingerprint"],
                    cwd=work,
                    environment=environment,
                ).stdout.strip(),
            }
            return {
                "device": device,
                "identities": sorted(
                    identities, key=lambda value: (value["case_id"], value["label"])
                ),
                "truth_table": sorted(
                    truth_table,
                    key=lambda value: (
                        value["numeric"],
                        value["promoted"],
                        value["research"],
                    ),
                ),
                "negative_controls": sorted(
                    negative_controls, key=lambda value: value["label"]
                ),
                "mutation_detected": mutation
                != fixture["cases"][0]["identities"][0]["sha256"],
            }
        finally:
            if emulator is not None and emulator.poll() is None:
                _command(
                    [adb, "-s", serial, "emu", "kill"],
                    cwd=work,
                    environment=environment,
                    timeout=30.0,
                    check=False,
                )
                try:
                    emulator.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    emulator.terminate()
                    emulator.wait(timeout=15.0)
            _finish_owned_emulator_processes(emulator_exe, runtime["avd_name"], port)


def evaluate(
    contract_path: Path,
    sdk: Path,
    ndk: Path,
    avdmanager: Path,
    avd_home: Path,
    work_root: Path,
    order: str,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    sdk = sdk.resolve()
    ndk = ndk.resolve()
    avdmanager = avdmanager.resolve()
    avd_home = avd_home.resolve()
    work_root = work_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_inputs(contract)
    runtime = contract["runtime"]
    emulator = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    system_package = sdk / runtime["system_package_relative_path"]
    _validate_file(emulator, runtime["emulator_sha256"], "emulator")
    _validate_file(adb, runtime["adb_sha256"], "adb")
    _validate_file(avdmanager, runtime["avdmanager_sha256"], "avdmanager")
    _validate_file(system_package, runtime["system_package_sha256"], "system package")
    fixture_path = ROOT / contract["inputs"]["fixture"]["path"]
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if avd_home.exists():
        raise P253RuntimeError("owned AVD home must be absent before execution")
    if work_root.exists():
        raise P253RuntimeError("owned work root must be absent before execution")
    runtime_result: dict[str, Any] | None = None
    build_result: dict[str, Any] | None = None
    work_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        _create_avd(avdmanager, avd_home, sdk, contract)
        with tempfile.TemporaryDirectory(
            prefix=f"{work_root.name}-", dir=work_root.parent
        ) as temporary:
            work = Path(temporary)
            build_result = _build_twice(ndk, work)
            runtime_result = _run_runtime(
                contract=contract,
                fixture=fixture,
                sdk=sdk,
                avd_home=avd_home,
                executable=build_result["x86_executable"],
                order=order,
                work=work,
            )
    finally:
        if avd_home.exists():
            shutil.rmtree(avd_home)
    assert build_result is not None and runtime_result is not None
    identities_exact = all(
        row["observed_sha256"] == row["expected_sha256"]
        for row in runtime_result["identities"]
    )
    truth_table_exact = all(
        row["observed"] == row["expected"] for row in runtime_result["truth_table"]
    )
    negative_controls_exact = all(
        row["returncode_nonzero"] and row["stdout_empty"]
        for row in runtime_result["negative_controls"]
    )
    device_exact = runtime_result["device"]["abi"] == runtime["abi"] and runtime_result[
        "device"
    ]["api"] == str(runtime["api_level"])
    cleanup_exact = not avd_home.exists() and not work_root.exists()
    gates = {
        "two_disjoint_builds_exact": build_result["all_exact"],
        "all_fixture_hashes_exact": identities_exact,
        "truth_table_exact": truth_table_exact,
        "negative_controls_exact": negative_controls_exact,
        "mutation_detected": runtime_result["mutation_detected"],
        "device_exact": device_exact,
        "owned_processes_zero": not _finish_owned_emulator_processes(
            emulator, runtime["avd_name"], int(runtime["port"])
        ),
        "owned_scratch_residue_zero": cleanup_exact,
    }
    passed = all(gates.values())
    stable = {
        "schema": "neuro-film.p253-reference-product-chain-android-runtime-result.v1",
        "status": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
        "contract_sha256": _sha256(contract_path),
        "fixture_sha256": _sha256(fixture_path),
        "toolchain": {
            "ndk_revision": contract["inputs"]["ndk_lock"]["revision"],
            "emulator_sha256": _sha256(emulator),
            "adb_sha256": _sha256(adb),
            "avdmanager_sha256": _sha256(avdmanager),
            "system_package_sha256": _sha256(system_package),
            "compiler": build_result["compiler"],
        },
        "builds": build_result["artifacts"],
        "runtime": runtime_result,
        "gates": gates,
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--avdmanager", type=Path, required=True)
    parser.add_argument("--avd-home", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--order", choices=("normal", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate(
        arguments.contract,
        arguments.sdk,
        arguments.ndk,
        arguments.avdmanager,
        arguments.avd_home,
        arguments.work_root,
        arguments.order,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
