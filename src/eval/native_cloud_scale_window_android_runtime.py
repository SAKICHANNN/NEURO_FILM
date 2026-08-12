"""P4EV Android runtime for the bounded native cloud scale window."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.native_cloud_row_chain_android_runtime import evaluate as evaluate_runtime
from src.eval.native_msvc import sha256_file


def evaluate(root: Path, contract_path: Path, ndk: Path, host_clang: Path,
             sdk: Path, avd_home: Path, output: Path, port: int = 5586) -> dict:
    contract = json.loads(contract_path.read_text())
    parent = root / contract["parent"]["path"]
    if (sha256_file(parent) != contract["parent"]["sha256"] or
            json.loads(parent.read_text())["decision"] !=
            contract["parent"]["required_decision"]):
        raise RuntimeError("P4EV parent drift")
    runtime_contract = dict(contract)
    runtime_contract["parent"] = {
        "path": contract["parent"]["path"],
        "sha256": contract["parent"]["sha256"],
        "required_decision": contract["parent"]["required_decision"],
    }
    temporary_contract = output / "p4ev-runtime-contract.json"
    output.mkdir(parents=True, exist_ok=True)
    temporary_contract.write_text(json.dumps(runtime_contract))
    # The shared runner owns emulator lifecycle; select the P4EV probe through
    # its explicit probe-source parameter.
    report = evaluate_runtime(
        root, temporary_contract, ndk, host_clang, sdk, avd_home, output,
        port=port, probe_source="native/film_physics/nf_cloud_scale_window_runtime_probe_v1.c",
    )
    stable0 = report["stable"]
    gates = {
        "full_window": all("full_window=1" in row for row in report["stdout_rows"]),
        "host_android": stable0["gates"]["density"] and stable0["gates"]["transmittance"],
        "repeat": stable0["gates"]["repeat"],
        "workspace": stable0["gates"]["workspace"],
        "atomic": stable0["gates"]["atomic"],
        "abi": stable0["gates"]["abi"],
        "cleanup": stable0["gates"]["cleanup"],
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "host_density_sha256": stable0["host_density_sha256"],
        "host_transmittance_sha256": stable0["host_transmittance_sha256"],
        "maximum_density_absolute_error": stable0["maximum_density_absolute_error"],
        "density_differing_values": stable0["density_differing_values"],
        "gates": gates,
        "decision": contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4ev_native_cloud_scale_window_android_runtime.v1",
        "automatic_pass": all(gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
