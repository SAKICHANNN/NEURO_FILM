#!/usr/bin/env python3
"""Audit exact R1GX Sony ARQ unpacking without copying producer code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.p321-r1gx-sony-arq-no-copy-intake.v1"


class P321Error(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _sha_bytes(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify(repo: Path, commit: str, binding: dict[str, Any]) -> bytes:
    value = _git_bytes(repo, commit, binding["path"])
    observed = (len(value), _git_blob(repo, commit, binding["path"]), _sha_bytes(value))
    expected = (binding["bytes"], binding["git_blob"], binding["sha256"])
    if observed != expected:
        raise P321Error(f"producer artifact differs: {binding['path']}")
    return value


def _load(site: Path, module_bytes: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "sony_arq.py").write_bytes(module_bytes)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.sony_arq")
    finally:
        sys.path.remove(str(site))


def _reject(callable_: Any, source: object) -> bool:
    try:
        callable_(source)
    except (TypeError, ValueError):
        return True
    return False


def _output_sha(output: np.ndarray) -> str:
    if output.dtype != np.uint16 or not output.flags.c_contiguous:
        raise P321Error("output is not contiguous uint16")
    return _sha_bytes(memoryview(output).cast("B"))


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P321Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIRST_CONSUMER_DECODE":
        raise P321Error("P321 is not frozen")

    items = list(config["artifacts"].items())
    invocation = list(reversed(items)) if order == "reverse" else items
    artifact_values: dict[str, bytes] = {}
    verified_unsorted: dict[str, dict[str, Any]] = {}
    for name, binding in invocation:
        commit = config["producer"][binding["commit_role"]]
        value = _verify(producer_repo, commit, binding)
        artifact_values[name] = value
        verified_unsorted[name] = {
            "path": binding["path"],
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "sha256": _sha_bytes(value),
        }
    verified = {name: verified_unsorted[name] for name, _ in items}

    preregistration = json.loads(artifact_values["preregistration"])
    source_gate = json.loads(artifact_values["source_gate"])
    fixture = json.loads(artifact_values["canonical_fixture"])
    execution_lock = json.loads(artifact_values["execution_lock"])
    contract = json.loads(artifact_values["callable_contract"])
    evidence = json.loads(artifact_values["evidence"])
    source_binding = config["source"]
    source_path = producer_repo / source_binding["path"]
    source_before = {
        "bytes": source_path.stat().st_size,
        "sha256": _sha_file(source_path),
    }
    if source_before != {
        "bytes": source_binding["bytes"],
        "sha256": source_binding["sha256"],
    }:
        raise P321Error("R1GX ARQ source differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p321-"))
    module_names = ("zhuise.sony_arq", "zhuise")
    try:
        module = _load(temporary / "site", artifact_values["callable_module"])
        isolated = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )

        fixture_source = bytes.fromhex(fixture["source_hex"])
        fixture_output = module.decode_sony_arq(fixture_source)
        fixture_row = {
            "source_bytes": len(fixture_source),
            "source_sha256": _sha_bytes(fixture_source),
            "shape": [int(value) for value in fixture_output.shape],
            "dtype": str(fixture_output.dtype),
            "output_sha256": _output_sha(fixture_output),
            "output_hex": memoryview(fixture_output).cast("B").hex(),
            "owned_writable_c_contiguous": bool(
                fixture_output.flags.owndata
                and fixture_output.flags.writeable
                and fixture_output.flags.c_contiguous
            ),
        }
        controls: dict[str, bool] = {}
        control_items = [
            ("mutable-fixture-input", bytearray(fixture_source)),
            ("truncated-fixture-input", fixture_source[:-1]),
        ]
        if order == "reverse":
            control_items.reverse()
        for name, control_source in control_items:
            controls[name] = _reject(module.decode_sony_arq, control_source)

        source = source_path.read_bytes()
        controls["truncated-real-source"] = _reject(
            module.decode_sony_arq, source[:-1]
        )
        source_memory_sha_before = _sha_bytes(source)
        metadata = module.parse_sony_arq_metadata(source)
        output = module.decode_sony_arq(source)
        real_row = {
            "source_bytes": len(source),
            "source_sha256": source_memory_sha_before,
            "pixel_shift_group": metadata.pixel_shift_group,
            "shape": [int(value) for value in output.shape],
            "dtype": str(output.dtype),
            "minimum_code": int(output.min()),
            "maximum_code": int(output.max()),
            "output_uint16le_sha256": _output_sha(output),
            "owned_writable_c_contiguous": bool(
                output.flags.owndata
                and output.flags.writeable
                and output.flags.c_contiguous
            ),
            "source_immutable_in_memory": source_memory_sha_before
            == _sha_bytes(source),
        }

        source_after = {
            "bytes": source_path.stat().st_size,
            "sha256": _sha_file(source_path),
        }
        execution = config["execution"]
        core = config["artifacts"]["callable_module"]
        fixture_exact = (
            fixture_row["source_bytes"] == fixture["source_bytes"]
            and fixture_row["source_sha256"] == fixture["source_sha256"]
            and fixture_row["shape"] == fixture["expected_shape"]
            and fixture_row["dtype"] == fixture["expected_dtype"]
            and fixture_row["output_sha256"] == fixture["expected_sha256"]
            and fixture_row["output_hex"] == fixture["expected_u16le_hex"]
            and fixture_row["owned_writable_c_contiguous"]
        )
        real_exact = (
            real_row["source_bytes"] == source_binding["bytes"]
            and real_row["source_sha256"] == source_binding["sha256"]
            and real_row["pixel_shift_group"] == source_binding["pixel_shift_group"]
            and real_row["shape"] == source_binding["geometry_hw4"]
            and real_row["dtype"] == "uint16"
            and real_row["minimum_code"] >= 0
            and real_row["maximum_code"] <= 16383
            and real_row["output_uint16le_sha256"]
            == source_binding["output_uint16le_sha256"]
            and real_row["owned_writable_c_contiguous"]
            and real_row["source_immutable_in_memory"]
        )
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "producer-preregistration-exact": preregistration["status"]
            == "FROZEN_BEFORE_SOURCE_PAYLOAD_ACQUISITION",
            "producer-source-gate-exact": source_gate["status"]
            == "PASS_SOURCE_GATE_CANDIDATE_IMPLEMENTATION_AUTHORIZED",
            "producer-v2-lock-exact": execution_lock["schema"].endswith(".v2")
            and execution_lock["fixed_identity"]["git_blobs"]["core"]
            == core["git_blob"]
            and execution_lock["fixed_identity"]["output_sha256"]
            == source_binding["output_uint16le_sha256"],
            "producer-contract-exact": contract["protocol"] == config["protocol"]
            and contract["source_lock"]["formal_execution_lock_commit"]
            == config["producer"]["execution_lock_commit"]
            and contract["source_lock"]["formal_report_sha256"]
            == execution["producer_report_sha256"]
            and contract["source_lock"]["formal_stable_identity"]
            == execution["producer_stable_identity"],
            "producer-evidence-exact": evidence["status"]
            == "PASS_PRIVATE_ONE_GROUP_SONY_ARQ_STORED_CODE_CALLABLE"
            and evidence["stable_identity"]
            == execution["producer_stable_identity"]
            and evidence["gates"]["cyclic_arw_association_discriminates"]
            and evidence["cyclic_arw_association_control"]["fraction"] <= 0.10,
            "superseded-v1-not-consumed": verified["evidence"]["sha256"]
            != execution["superseded_report_sha256"],
            "isolated-git-object-import": isolated,
            "callable-identity-exact": module.CALLABLE_ID == config["protocol"],
            "canonical-fixture-exact": fixture_exact,
            "real-arq-output-exact": real_exact,
            "invalid-controls-reject": all(controls.values()),
            "source-immutable": source_after == source_before,
            "component-arw-files-unread": True,
        }
        scientific = {
            "protocol": config["protocol"],
            "fixture": fixture_row,
            "real_source": real_row,
            "controls": {key: controls[key] for key in sorted(controls)},
            "gates": gates,
        }
        report = {
            "schema": SCHEMA,
            "experiment_id": "P321",
            "status": "PASS_PRIVATE_R1GX_SONY_ARQ_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1GX_SONY_ARQ_NO_COPY_INTAKE",
            "bindings": {
                "artifacts": verified,
                "config_sha256": _sha_bytes(config_bytes),
                "source": source_before,
            },
            "scientific": scientific,
            "scientific_identity": _sha_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "component_arw_reads": 0,
                "consumer_core_copied": False,
                "default_loader_changed": False,
                "producer_runner_executed": False,
                "producer_reference_decoder_executed": False,
                "product_mapping": False,
                "source_rights": "raw.pixls.us CC0/Public Domain",
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            failed = sorted(name for name, value in gates.items() if not value)
            raise P321Error(f"P321 gates failed: {', '.join(failed)}")
        return report
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P321Error("temporary residue remains")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config, args.producer_repo, args.order)
    payload = _canonical(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "bytes": len(payload),
                "sha256": _sha_bytes(payload),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
