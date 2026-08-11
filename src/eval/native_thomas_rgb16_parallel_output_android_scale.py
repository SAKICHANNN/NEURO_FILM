"""Exact 12MP audit for within-tile parallel Thomas RGB16 output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_cached_android_scale import (
    SOURCE_PATHS,
    NativeThomasRgb16CachedScaleError,
    _canonical_bytes,
    _one_android_boot,
    build_android_probe,
    build_msvc_probe,
    run_host_probe,
)


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
        != "neuro_film.u6_p8cp_parallel_output_thomas_rgb16_contract.v1"
    ):
        raise NativeThomasRgb16CachedScaleError("P8CP contract drift")
    parent = contract["parent"]
    if sha256_file(root / parent["path"]) != parent["sha256"]:
        raise NativeThomasRgb16CachedScaleError("P8CO parent evidence drift")
    fixture = contract["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    row_partition = int(fixture["row_partition"])
    output_dir.mkdir(parents=True, exist_ok=True)
    host_build = build_msvc_probe(root, output_dir / "host_build")
    android_build = build_android_probe(
        root, ndk, output_dir / "android_build/nf_p8cp_probe"
    )
    modes = (
        ("cached", "cached", 3),
        ("output3a", "cached_output3", 3),
        ("output3b", "cached_output3", 3),
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
            modes=modes,
        )
        for index in (1, 2)
    ]
    host_stable = [_stable_run(row) for row in host_runs]
    android_stable = [
        [_stable_run(row) for row in boot["runs"]] for boot in android_boots
    ]
    all_runs = [*host_runs, *(row for boot in android_boots for row in boot["runs"])]
    output_runs = [row for row in all_runs if row["facts"]["mode"] == "cached_output3"]
    android_output_runs = [
        row
        for boot in android_boots
        for row in boot["runs"]
        if row["facts"]["mode"] == "cached_output3"
    ]
    gates = {
        "p8co_parallel_rgb16_byte_exact": len({row["raw_sha256"] for row in all_runs})
        == 1,
        "single_parallel_output_byte_exact": all(
            row["raw_sha256"] == host_runs[0]["raw_sha256"] for row in output_runs
        ),
        "host_repeat_exact": host_stable[1] == host_stable[2],
        "android_cold_boot_repeat_exact": android_stable[0] == android_stable[1],
        "host_android_exact": host_stable == android_stable[0],
        "android_parallel_output_wall_bounded": max(
            row["device_command_seconds"] for row in android_output_runs
        )
        <= float(contract["gates"]["max_android_parallel_output_command_seconds"]),
        "workspace_exact": all(
            row["facts"]["workspace"]
            == int(contract["gates"]["required_workspace_bytes"])
            for row in all_runs
        ),
        "invalid_input_zero_sink_calls": all(
            row["facts"]["invalid_status"] == 3 and row["facts"]["invalid_calls"] == 0
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
        "host_runs": host_stable,
        "android_runs": android_stable[0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cp_parallel_output_thomas_rgb16_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": automatic_pass,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "host_build": host_build,
        "android_build": android_build,
        "host_observations": host_runs,
        "android_observations": android_boots,
    }


__all__ = ["evaluate"]
