"""Android x86_64 runtime conformance for the frozen P8BO component closure."""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from scripts.run_u6_p8bo_portable_native_standard import _render
from src.eval.native_standard_portable import (
    COMPONENT_NAMES,
    sha256_file,
    validate_ndk,
)
from src.eval.native_thomas_rgb16_png_android_runtime import (
    _android_env,
    _finish_owned_emulator_processes,
    _owned_emulator_processes,
    _wait_for_boot,
)
from src.film_physics.native_abi_layouts import (
    native_adjacency_profile_struct,
    native_gaussian_profile_struct,
    native_print_profile_struct,
)
from src.film_physics.native_adjacency_profile import (
    compile_native_adjacency_profile_payload,
)
from src.film_physics.native_ao6_base_profile import native_ao6_base_profile_struct
from src.film_physics.native_ao6_residual_profile import (
    native_ao6_residual_profile_struct,
)
from src.film_physics.native_gauge_profile import native_gauge_profile_struct
from src.film_physics.native_profile import compile_native_domains_profile_payload
from src.film_physics.native_spatial_profile import (
    compile_native_gaussian_profile_payload,
)
from src.film_physics.native_standard_factory import (
    create_opt_in_native_standard_runtime,
)
from src.film_physics.profile_consumer import compile_standalone_profile_artifact


class NativeStandardAndroidRuntimeError(RuntimeError):
    """Raised when the frozen Android runtime contract is violated."""


P8BO_SOURCE_COMMIT = "7af72347fca54a7a1324944f3c4c87e77e7079f2"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _write_struct(path: Path, value: ctypes.Structure) -> None:
    path.write_bytes(ctypes.string_at(ctypes.byref(value), ctypes.sizeof(value)))


def _git_bytes(root: Path, revision: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise NativeStandardAndroidRuntimeError(
            f"cannot read frozen Git source {revision}:{relative}"
        )
    return completed.stdout


def _frozen_native_headers(root: Path) -> dict[str, bytes]:
    completed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            P8BO_SOURCE_COMMIT,
            "native/film_physics",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise NativeStandardAndroidRuntimeError("cannot enumerate P8BO Git snapshot")
    return {
        Path(relative).name: _git_bytes(root, P8BO_SOURCE_COMMIT, relative)
        for relative in completed.stdout.splitlines()
        if relative.endswith(".h")
    }


def _validate_portable_snapshot(
    root: Path, portable: dict[str, Any]
) -> dict[str, bytes]:
    if (
        portable.get("schema")
        != ("neuro_film.u6_p8bo_portable_native_standard_contract.v1")
        or tuple(portable["components"]) != COMPONENT_NAMES
    ):
        raise NativeStandardAndroidRuntimeError("P8BO portable schema drift")
    parent = portable["parent_decision"]
    if sha256_file(root / parent["path"]) != parent["sha256"]:
        raise NativeStandardAndroidRuntimeError("P8BO portable parent drift")
    if portable["oracle"]["strength"] != 1.0:
        raise NativeStandardAndroidRuntimeError("P8BO portable strength drift")
    headers: dict[str, bytes] = {}
    for row in portable["components"].values():
        header_relative = f"native/film_physics/{row['basename']}.h"
        header = _git_bytes(root, P8BO_SOURCE_COMMIT, header_relative)
        if hashlib.sha256(header).hexdigest() != row["header_sha256"]:
            raise NativeStandardAndroidRuntimeError("P8BO frozen header drift")
        headers[Path(header_relative).name] = header
        sources = row.get("link_sources")
        if not isinstance(sources, dict) or not sources:
            raise NativeStandardAndroidRuntimeError("P8BO link closure missing")
        for relative, expected_sha256 in sources.items():
            source = _git_bytes(root, P8BO_SOURCE_COMMIT, relative)
            if hashlib.sha256(source).hexdigest() != expected_sha256:
                raise NativeStandardAndroidRuntimeError("P8BO frozen source drift")
    return headers


def _run(
    command: list[str | Path],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(item) for item in command],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise NativeStandardAndroidRuntimeError(
            "command failed: "
            + " ".join(map(str, command))
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def _validate_parent(
    root: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_spec = contract["parent"]
    parent_path = root / parent_spec["path"]
    if sha256_file(parent_path) != parent_spec["sha256"]:
        raise NativeStandardAndroidRuntimeError("P8BO parent hash drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent["result"]["status"] != parent_spec["required_result_status"]
        or parent["result"]["oracle_output_sha256"]
        != parent_spec["required_oracle_output_sha256"]
    ):
        raise NativeStandardAndroidRuntimeError("P8BO parent result drift")
    portable_spec = contract["portable_contract"]
    portable_path = root / portable_spec["path"]
    if sha256_file(portable_path) != portable_spec["sha256"]:
        raise NativeStandardAndroidRuntimeError("P8BO portable contract drift")
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    _validate_portable_snapshot(root, portable)
    return parent, portable


def _prepare_fixture(
    root: Path, output_dir: Path, portable: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    profile_config = json.loads(
        (root / portable["profile_compiler_config"]).read_text(encoding="utf-8")
    )
    artifact = compile_standalone_profile_artifact(root=root, config=profile_config)
    package = json.loads(
        (root / portable["windows_package"]).read_text(encoding="utf-8")
    )
    p8bo_report = json.loads(
        (root / "outputs/u6_p8bo_portable_native_standard_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    llvm_dir = root / "outputs/u6_p8bo_portable_native_standard_v1/build/llvm_windows_a"
    names = {name: row["basename"] for name, row in portable["components"].items()}
    llvm_package = deepcopy(package)
    library_paths: dict[str, Path] = {}
    for name, basename in names.items():
        path = llvm_dir / f"{basename}.dll"
        expected = p8bo_report["windows"]["llvm_component_sha256"][name]
        if sha256_file(path) != expected:
            raise NativeStandardAndroidRuntimeError("P8BO Windows oracle library drift")
        llvm_package["components"][name]["sha256"] = expected
        library_paths[name] = path
    runtime, _ = create_opt_in_native_standard_runtime(
        package=llvm_package, artifact=artifact, library_paths=library_paths
    )
    height = int(portable["oracle"]["height"])
    width = int(portable["oracle"]["width"])
    expected_bytes, _ = _render(runtime, height, width)
    expected = np.frombuffer(expected_bytes, dtype=np.float32).copy()
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    fixture = output_dir / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "source.f32").write_bytes(source.tobytes())
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    component_payloads = artifact["component_payloads"]
    display = component_payloads["ao6-source-context-display-look"]
    _write_struct(fixture / "domains.bin", native_print_profile_struct(domains_payload))
    _write_struct(
        fixture / "adjacency.bin", native_adjacency_profile_struct(adjacency_payload)
    )
    _write_struct(
        fixture / "gauge.bin",
        native_gauge_profile_struct(component_payloads["neutral-axis-gauge"]),
    )
    _write_struct(fixture / "base.bin", native_ao6_base_profile_struct(display))
    _write_struct(fixture / "residual.bin", native_ao6_residual_profile_struct(display))
    stages = {row["stage"]: row for row in spatial_payload["stages"]}
    for stage, filename in (
        ("forward_scatter", "gaussian_forward.bin"),
        ("development_adjacency", "gaussian_adjacency.bin"),
        ("dye_diffusion", "gaussian_diffusion.bin"),
        ("scanner_mtf", "gaussian_scanner.bin"),
    ):
        _write_struct(
            fixture / filename,
            native_gaussian_profile_struct(spatial_payload, stages[stage]),
        )
    identities = {
        path.name: sha256_file(path)
        for path in sorted(fixture.iterdir())
        if path.is_file()
    }
    return source, expected, identities


def _build_probe(
    root: Path,
    portable: dict[str, Any],
    ndk: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, str]]:
    clang, _ = validate_ndk(ndk, portable["toolchains"]["android_ndk"])
    build = output_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    include = output_dir / "include"
    include.mkdir(parents=True, exist_ok=True)
    _validate_portable_snapshot(root, portable)
    for name, payload in _frozen_native_headers(root).items():
        (include / name).write_bytes(payload)
    android_dir = (
        root / "outputs/u6_p8bo_portable_native_standard_v1/build/android_a/x86_64"
    )
    report = json.loads(
        (root / "outputs/u6_p8bo_portable_native_standard_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    hashes: dict[str, str] = {}
    for name, row in portable["components"].items():
        source = android_dir / f"lib{row['basename']}.so"
        expected = report["android"]["targets"]["x86_64"]["components"][name]["sha256"]
        if sha256_file(source) != expected:
            raise NativeStandardAndroidRuntimeError("P8BO Android library drift")
        target = build / source.name
        shutil.copyfile(source, target)
        hashes[target.name] = expected
    probe = build / "nf_native_standard_android_probe_v1"
    libraries = [
        "nf_physical_domains_f32_v1",
        "nf_gaussian_rgb_f32_v1",
        "nf_bounded_adjacency_f32_v1",
        "nf_neutral_gauge_f32_v1",
        "nf_ao6_context_f32_v2",
        "nf_ao6_display_f32_v4",
    ]
    _run(
        [
            clang,
            "--target=x86_64-linux-android21",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            include,
            root / "native/film_physics/nf_native_standard_android_probe_v1.c",
            "-L",
            build,
            *[f"-l{name}" for name in libraries],
            "-lm",
            "-Wl,-rpath,$ORIGIN",
            "-Wl,--build-id=none",
            "-o",
            probe,
        ]
    )
    hashes[probe.name] = sha256_file(probe)
    return probe, hashes


def _boot_and_run(
    *,
    sdk: Path,
    avd_home: Path,
    avd_name: str,
    port: int,
    boot_index: int,
    local_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    emulator_exe = sdk / "emulator/emulator.exe"
    adb = sdk / "platform-tools/adb.exe"
    env = _android_env(sdk, avd_home)
    serial = f"emulator-{port}"
    stdout_path = output_dir / f"boot{boot_index}_emulator.stdout.log"
    stderr_path = output_dir / f"boot{boot_index}_emulator.stderr.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
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
                "-wipe-data",
                "-no-snapshot",
                "-gpu",
                "swiftshader_indirect",
            ],
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        try:
            _wait_for_boot(adb, serial, emulator, env=env, timeout=180.0)
            remote = "/data/local/tmp/nf_p8bo1"
            _run([adb, "-s", serial, "shell", "rm", "-rf", remote], env=env)
            _run([adb, "-s", serial, "shell", "mkdir", "-p", remote], env=env)
            for directory in ("build", "fixture"):
                _run(
                    [
                        adb,
                        "-s",
                        serial,
                        "push",
                        str(local_dir / directory),
                        f"{remote}/{directory}",
                    ],
                    env=env,
                )
            _run(
                [
                    adb,
                    "-s",
                    serial,
                    "shell",
                    "chmod",
                    "755",
                    f"{remote}/build/nf_native_standard_android_probe_v1",
                ],
                env=env,
            )
            rows = []
            for process_index in range(2):
                remote_output = f"{remote}/output_{process_index}.f32"
                command = (
                    f"cd {remote}/build && "
                    f"LD_LIBRARY_PATH=. ./nf_native_standard_android_probe_v1 "
                    f"{remote}/fixture {remote_output}"
                )
                completed = _run([adb, "-s", serial, "shell", command], env=env)
                local_output = (
                    output_dir / f"boot{boot_index}_process{process_index}.f32"
                )
                _run([adb, "-s", serial, "pull", remote_output, local_output], env=env)
                rows.append(
                    {
                        "stdout": completed.stdout.strip(),
                        "output_sha256": sha256_file(local_output),
                        "output_path": local_output,
                    }
                )
            properties = {
                "model": _run(
                    [adb, "-s", serial, "shell", "getprop", "ro.product.model"],
                    env=env,
                ).stdout.strip(),
                "fingerprint": _run(
                    [
                        adb,
                        "-s",
                        serial,
                        "shell",
                        "getprop",
                        "ro.build.fingerprint",
                    ],
                    env=env,
                ).stdout.strip(),
            }
            _run([adb, "-s", serial, "shell", "rm", "-rf", remote], env=env)
            _run([adb, "-s", serial, "emu", "kill"], env=env)
            emulator.wait(timeout=30)
        finally:
            if emulator.poll() is None:
                emulator.terminate()
                try:
                    emulator.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    emulator.kill()
                    emulator.wait(timeout=15)
    forced = _finish_owned_emulator_processes(emulator_exe, avd_name, port)
    if _owned_emulator_processes(emulator_exe, avd_name, port):
        raise NativeStandardAndroidRuntimeError("owned emulator survived cleanup")
    return {
        "rows": rows,
        "properties": properties,
        "forced_cleanup": forced,
        "owned_process_survived_cleanup": False,
    }


def evaluate(
    *,
    root: Path,
    contract_path: Path,
    ndk: Path,
    sdk: Path,
    avd_home: Path,
    output_dir: Path,
    port: int,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema")
        != "neuro_film.u6_p8bo1_android_native_standard_runtime_contract.v1"
    ):
        raise NativeStandardAndroidRuntimeError("P8BO1 contract drift")
    parent, portable = _validate_parent(root, contract)
    runtime = contract["runtime"]
    if (
        sha256_file(
            sdk / "system-images/android-34/google_apis/x86_64/source.properties"
        )
        != runtime["system_image_source_properties_sha256"]
        or sha256_file(sdk / "emulator/emulator.exe") != runtime["emulator_sha256"]
        or sha256_file(sdk / "platform-tools/adb.exe") != runtime["adb_sha256"]
    ):
        raise NativeStandardAndroidRuntimeError("Android runtime asset drift")
    payload_dir = output_dir / "payload"
    source, expected, fixture_hashes = _prepare_fixture(root, payload_dir, portable)
    probe, build_hashes = _build_probe(root, portable, ndk, payload_dir)
    boots = []
    for index in range(runtime["wipe_data_cold_boots"]):
        boots.append(
            _boot_and_run(
                sdk=sdk,
                avd_home=avd_home,
                avd_name=runtime["avd_name"],
                port=port,
                boot_index=index,
                local_dir=payload_dir,
                output_dir=output_dir,
            )
        )
    flattened = [row for boot in boots for row in boot["rows"]]
    observed = [np.fromfile(row["output_path"], dtype=np.float32) for row in flattened]
    if any(values.size != expected.size for values in observed):
        raise NativeStandardAndroidRuntimeError("Android output sample count drift")
    maximum = max(
        float(np.max(np.abs(values.astype(np.float64) - expected)))
        for values in observed
    )
    rmse = max(
        float(np.sqrt(np.mean(np.square(values.astype(np.float64) - expected))))
        for values in observed
    )
    replay_exact = len({row["output_sha256"] for row in flattened}) == 1
    stdout_exact = {row["stdout"] for row in flattened} == {
        f"source_unchanged=1 invalid_atomic=1 samples={source.size}"
    }
    properties_exact = len({_canonical(boot["properties"]) for boot in boots}) == 1
    gates = {
        "maximum_absolute_error": maximum
        <= contract["fixture"]["maximum_absolute_error"],
        "rmse": rmse <= contract["fixture"]["maximum_rmse"],
        "all_finite": all(np.all(np.isfinite(values)) for values in observed),
        "replay_exact": replay_exact,
        "source_unchanged_and_invalid_atomic": stdout_exact,
        "runtime_properties_exact": properties_exact,
        "owned_process_cleanup": all(
            not boot["owned_process_survived_cleanup"] for boot in boots
        ),
        "parent_oracle_identity": hashlib.sha256(expected.tobytes()).hexdigest()
        == parent["result"]["oracle_output_sha256"],
    }
    passed = all(gates.values())
    stable = {
        "schema": "neuro_film.u6_p8bo1_android_native_standard_runtime_report.v1",
        "contract_sha256": sha256_file(contract_path),
        "probe_sha256": sha256_file(probe),
        "fixture_hashes": fixture_hashes,
        "build_hashes": build_hashes,
        "runtime": {
            "api_level": runtime["api_level"],
            "abi": runtime["abi"],
            "model": boots[0]["properties"]["model"],
            "build_fingerprint": boots[0]["properties"]["fingerprint"],
            "cold_boots": len(boots),
            "fresh_processes": len(flattened),
        },
        "oracle_output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "android_output_sha256": flattened[0]["output_sha256"],
        "maximum_absolute_error": maximum,
        "rmse": rmse,
        "gates": gates,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable["stable_evidence_id"] = hashlib.sha256(_canonical(stable)).hexdigest()
    return stable


__all__ = ["NativeStandardAndroidRuntimeError", "evaluate"]
