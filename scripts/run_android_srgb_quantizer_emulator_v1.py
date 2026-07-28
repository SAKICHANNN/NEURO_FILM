"""Run the exact consumer sRGB quantizer on an owned Android 14 emulator."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OWNED_SDK = ROOT / "outputs/tmp/android-emulator-sdk"
OWNED_AVD_HOME = ROOT / "outputs/tmp/android-emulator-avd"
OWNED_LOGS = ROOT / "outputs/tmp/android-emulator-runtime"
PACKAGE_REPORT = (
    ROOT / "outputs/eval/android_srgb_quantizer_testlab_package_v1.json"
)
RUNTIME_REPORT = (
    ROOT / "outputs/eval/android_srgb_quantizer_emulator_runtime_v1.json"
)
AVDMANAGER = OWNED_SDK / "cmdline-tools/latest/bin/avdmanager.bat"
AVD_NAME = "nf_019f9f37_p90_api34"
SYSTEM_IMAGE = "system-images;android-34;google_apis;x86_64"
EMULATOR_PORT = 5580
SERIAL = f"emulator-{EMULATOR_PORT}"
REQUIRED_RESULT = {
    "schema": "neuro-film.android-srgb-quantizer-runtime.v1",
    "status": "PASS",
    "sample_count": 4096,
    "threshold_identity": (
        "fae645ef1aad04fcd1233631a32f820c"
        "f7696e3ca65d31939acf60d7f123674c"
    ),
    "vector_sha256": (
        "3d4205e51de80392ea7a4e5eccaf05ab"
        "6d322a48475603764c28e47d61aa7628"
    ),
    "q8_exact": True,
    "q16_exact": True,
    "inner_replay_exact": True,
    "failure_atomic": True,
    "icc_exact": True,
    "eotf_q8_roundtrip_exact": True,
    "eotf_q16_roundtrip_exact": True,
    "eotf_failure_atomic": True,
    "product_chain_vector_count": 10,
    "product_chain_canonical_bytes": 14254,
    "product_chain_hashes_exact": True,
    "product_chain_sha_failure_atomic": True,
    "staging_truth_table_exact": True,
}


class EmulatorRuntimeError(RuntimeError):
    """Raised when the owned emulator run cannot satisfy its contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "ANDROID_HOME": str(OWNED_SDK),
        "ANDROID_SDK_ROOT": str(OWNED_SDK),
        "ANDROID_AVD_HOME": str(OWNED_AVD_HOME),
        "ANDROID_EMULATOR_HOME": str(OWNED_AVD_HOME.parent),
    }


def _run(
    arguments: list[object],
    *,
    timeout: float = 120.0,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(value) for value in arguments]
    if command[0].casefold().endswith((".bat", ".cmd")):
        command = [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            *command,
        ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
        timeout=timeout,
        env=_environment(),
    )
    if check and completed.returncode != 0:
        raise EmulatorRuntimeError(
            f"owned emulator command failed with exit {completed.returncode}"
        )
    return completed


def _package() -> dict[str, Any]:
    report = json.loads(PACKAGE_REPORT.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("schema")
        != "neuro-film.android-srgb-quantizer-testlab-package.v1"
    ):
        raise EmulatorRuntimeError("package report contract mismatch")
    for role in ("app", "test"):
        artifact = report["artifacts"][role]
        path = Path(artifact["path"])
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise EmulatorRuntimeError(f"{role} APK identity mismatch")
    return report


def _instrumentation_result(output: str) -> dict[str, Any]:
    marker = "nf_runtime_result="
    for line in output.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EmulatorRuntimeError(
                "instrumentation result JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            break
        return value
    raise EmulatorRuntimeError("instrumentation result is unavailable")


def _validate_instrumentation(output: str) -> dict[str, Any]:
    if "INSTRUMENTATION_CODE: -1" not in output:
        raise EmulatorRuntimeError("instrumentation did not finish successfully")
    if "outer_replays=2" not in output:
        raise EmulatorRuntimeError("instrumentation outer replay count missing")
    if "outer_replay_exact=true" not in output:
        raise EmulatorRuntimeError("instrumentation outer replay differs")
    result = _instrumentation_result(output)
    for key, expected in REQUIRED_RESULT.items():
        if result.get(key) != expected:
            raise EmulatorRuntimeError(
                f"instrumentation result mismatch for {key}"
            )
    return result


def _adb(adb: Path, *arguments: str, timeout: float = 120.0) -> str:
    return _run(
        [adb, "-s", SERIAL, *arguments],
        timeout=timeout,
    ).stdout


def _wait_for_boot(
    adb: Path,
    emulator: subprocess.Popen[str],
    *,
    timeout: float = 240.0,
) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if emulator.poll() is not None:
            raise EmulatorRuntimeError("emulator exited before boot")
        completed = _run(
            [adb, "-s", SERIAL, "shell", "getprop", "sys.boot_completed"],
            timeout=15.0,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip() == "1":
            return
        time.sleep(3.0)
    raise EmulatorRuntimeError("emulator boot timeout")


def execute() -> dict[str, Any]:
    package = _package()
    emulator_exe = OWNED_SDK / "emulator/emulator.exe"
    adb = OWNED_SDK / "platform-tools/adb.exe"
    system_package = (
        OWNED_SDK
        / "system-images/android-34/google_apis/x86_64/package.xml"
    )
    for required in (
        emulator_exe,
        adb,
        system_package,
        AVDMANAGER,
    ):
        if not required.is_file():
            raise EmulatorRuntimeError(
                "owned Android emulator dependency is unavailable"
            )
    OWNED_AVD_HOME.mkdir(parents=True, exist_ok=True)
    OWNED_LOGS.mkdir(parents=True, exist_ok=True)
    avd_config = OWNED_AVD_HOME / f"{AVD_NAME}.avd/config.ini"
    if not avd_config.is_file():
        _run(
            [
                AVDMANAGER,
                "create",
                "avd",
                "--name",
                AVD_NAME,
                "--package",
                SYSTEM_IMAGE,
                "--device",
                "pixel_8",
                "--force",
            ],
            input_text="no\n",
        )
    stdout_path = OWNED_LOGS / "emulator.stdout.log"
    stderr_path = OWNED_LOGS / "emulator.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8", newline="\n")
    stderr_handle = stderr_path.open("w", encoding="utf-8", newline="\n")
    emulator: subprocess.Popen[str] | None = None
    started_at = _now()
    started = time.monotonic()
    try:
        emulator = subprocess.Popen(
            [
                str(emulator_exe),
                "-avd",
                AVD_NAME,
                "-port",
                str(EMULATOR_PORT),
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
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _wait_for_boot(adb, emulator)
        app = Path(package["artifacts"]["app"]["path"])
        test = Path(package["artifacts"]["test"]["path"])
        if "Success" not in _adb(adb, "install", "-r", str(app)):
            raise EmulatorRuntimeError("target APK install failed")
        if "Success" not in _adb(adb, "install", "-r", str(test)):
            raise EmulatorRuntimeError("test APK install failed")
        component = (
            "com.neurofilm.srgbquantizer.test/"
            "com.neurofilm.srgbquantizer.QuantizerInstrumentation"
        )
        outputs = [
            _adb(
                adb,
                "shell",
                "am",
                "instrument",
                "-w",
                "-r",
                component,
                timeout=120.0,
            )
            for _ in range(2)
        ]
        results = [_validate_instrumentation(value) for value in outputs]
        if results[0] != results[1]:
            raise EmulatorRuntimeError(
                "two host instrumentation invocations differ"
            )
        device = {
            "abi": _adb(adb, "shell", "getprop", "ro.product.cpu.abi").strip(),
            "api": _adb(adb, "shell", "getprop", "ro.build.version.sdk").strip(),
            "model": _adb(adb, "shell", "getprop", "ro.product.model").strip(),
            "build_fingerprint": _adb(
                adb,
                "shell",
                "getprop",
                "ro.build.fingerprint",
            ).strip(),
        }
        if device["abi"] != "x86_64" or device["api"] != "34":
            raise EmulatorRuntimeError("emulator target identity mismatch")
        stable = {
            "schema": (
                "neuro-film.android-srgb-quantizer-emulator-runtime.v1"
            ),
            "status": "PASS",
            "claim_ceiling": (
                "Android-14-x86_64-emulator-consumer-quantizer-runtime-only"
            ),
            "package": {
                "package_identity": package["package_identity"],
                "app_sha256": package["artifacts"]["app"]["sha256"],
                "test_sha256": package["artifacts"]["test"]["sha256"],
                "certificate_sha256": package["artifacts"]["test"][
                    "certificate_sha256"
                ],
                "core_sha256": package["native"]["core_libraries"][
                    "x86_64"
                ],
                "jni_sha256": package["native"]["jni_libraries"]["x86_64"],
                "eotf_sha256": package["native"]["eotf_libraries"]["x86_64"],
                "icc_sha256": package["native"]["icc_libraries"]["x86_64"],
            },
            "runtime": {
                "emulator_sha256": _sha256(emulator_exe),
                "system_package_sha256": _sha256(system_package),
                "avd_name": AVD_NAME,
                "device": device,
                "instrumentation_invocations": 2,
                "java_outer_replays_per_invocation": 2,
                "native_inner_replays_per_depth": 2,
                "result": results[0],
            },
        }
        stable_identity = hashlib.sha256(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        report = {
            **stable,
            "stable_identity": f"sha256:{stable_identity}",
            "observed": {
                "started_at": started_at,
                "finished_at": _now(),
                "elapsed_seconds": time.monotonic() - started,
            },
        }
        _atomic_json(RUNTIME_REPORT, report)
        return report
    finally:
        if emulator is not None and emulator.poll() is None:
            _run(
                [adb, "-s", SERIAL, "emu", "kill"],
                timeout=30.0,
                check=False,
            )
            try:
                emulator.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                emulator.terminate()
                emulator.wait(timeout=15.0)
        stdout_handle.close()
        stderr_handle.close()


def main() -> int:
    report = execute()
    print(
        json.dumps(
            {
                "status": report["status"],
                "stable_identity": report["stable_identity"],
                "runtime_report_sha256": _sha256(RUNTIME_REPORT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
