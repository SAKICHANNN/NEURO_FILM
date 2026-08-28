#!/usr/bin/env python3
"""Audit exact R1GL Nikon NEF unpacking without copying producer code."""

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

SCHEMA = "neuro-film.p315-r1gl-nikon-nef-no-copy-intake.v1"


class P315Error(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, check=True, capture_output=True
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
        raise P315Error(f"producer artifact differs: {binding['path']}")
    return value


def _load(site: Path, module_bytes: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "nikon_nef.py").write_bytes(module_bytes)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.nikon_nef")
    finally:
        sys.path.remove(str(site))


def _reject_truncation(module: Any, source: bytes) -> bool:
    try:
        module.decode_nikon_nef(source[:-1])
    except ValueError:
        return True
    return False


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P315Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIRST_CONSUMER_DECODE":
        raise P315Error("P315 is not frozen")

    artifacts: dict[str, bytes] = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["artifacts"].items():
        commit = config["producer"][binding["commit_role"]]
        value = _verify(producer_repo, commit, binding)
        artifacts[name] = value
        verified[name] = {
            "path": binding["path"],
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "sha256": _sha_bytes(value),
        }

    contract = json.loads(artifacts["callable_contract"])
    fixture = json.loads(artifacts["canonical_fixture"])
    preregistration = json.loads(artifacts["preregistration"])
    execution_lock = json.loads(artifacts["execution_lock"])
    evidence = json.loads(artifacts["evidence"])
    source_bindings = config["sources"]
    source_before = {
        row["id"]: {
            "bytes": (producer_repo / row["path"]).stat().st_size,
            "sha256": _sha_file(producer_repo / row["path"]),
        }
        for row in source_bindings
    }
    for row in source_bindings:
        expected = {key: row[key] for key in ("bytes", "sha256")}
        if source_before[row["id"]] != expected:
            raise P315Error(f"source differs: {row['id']}")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p315-"))
    module_names = ("zhuise.nikon_nef", "zhuise")
    try:
        module = _load(temporary / "site", artifacts["callable_module"])
        isolated = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        invocation = list(reversed(source_bindings)) if order == "reverse" else source_bindings
        rows: list[dict[str, Any]] = []
        controls: dict[str, bool] = {}
        for binding in invocation:
            path = producer_repo / binding["path"]
            source = path.read_bytes()
            source_hash = _sha_bytes(source)
            metadata = module.parse_nikon_nef_metadata(source)
            output = module.decode_nikon_nef(source)
            output_hash = _sha_bytes(np.asarray(output, dtype="<u2").tobytes())
            rows.append(
                {
                    "id": binding["id"],
                    "source_bytes": len(source),
                    "source_sha256": source_hash,
                    "geometry_hw": [int(output.shape[0]), int(output.shape[1])],
                    "bits_per_sample": int(metadata.bits_per_sample),
                    "compression": int(metadata.compression),
                    "output_dtype": str(output.dtype),
                    "output_uint16le_sha256": output_hash,
                    "minimum_code": int(output.min()),
                    "maximum_code": int(output.max()),
                    "owned_writable_c_contiguous": bool(
                        output.flags.owndata
                        and output.flags.writeable
                        and output.flags.c_contiguous
                    ),
                    "source_immutable_in_memory": source_hash == _sha_bytes(source),
                }
            )
            controls[f"{binding['id']}-truncate-one-byte"] = _reject_truncation(
                module, source
            )
        order_index = {row["id"]: index for index, row in enumerate(source_bindings)}
        rows.sort(key=lambda row: order_index[row["id"]])

        fixture_by_id = {row["id"]: row for row in fixture["rows"]}
        exact_rows = []
        for row, binding in zip(rows, source_bindings, strict=True):
            fixture_row = fixture_by_id[binding["id"]]
            exact_rows.append(
                row["source_bytes"] == binding["bytes"] == fixture_row["source_bytes"]
                and row["source_sha256"] == binding["sha256"] == fixture_row["source_sha256"]
                and row["geometry_hw"] == binding["geometry_hw"] == fixture_row["geometry_hw"]
                and row["bits_per_sample"]
                == binding["bits_per_sample"]
                == fixture_row["bits_per_sample"]
                and row["compression"] == 34713
                and row["output_dtype"] == "uint16"
                and row["output_uint16le_sha256"]
                == binding["output_uint16le_sha256"]
                == fixture_row["output_sha256"]
                and row["minimum_code"] >= 0
                and row["maximum_code"] < (1 << row["bits_per_sample"])
                and row["owned_writable_c_contiguous"]
                and row["source_immutable_in_memory"]
            )

        source_after = {
            row["id"]: {
                "bytes": (producer_repo / row["path"]).stat().st_size,
                "sha256": _sha_file(producer_repo / row["path"]),
            }
            for row in source_bindings
        }
        execution = config["execution"]
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "producer-contract-exact": contract["callable"]["id"] == config["protocol"]
            and contract["source_lock"]["implementation_commit"]
            == config["producer"]["implementation_commit"]
            and contract["source_lock"]["formal_execution_commit"]
            == config["producer"]["formal_execution_commit"],
            "producer-locks-exact": preregistration["candidate"]["callable_id"]
            == config["protocol"]
            and execution_lock["formal_execution_base_commit"]
            == config["producer"]["runner_commit"],
            "producer-evidence-exact": evidence["status"]
            == "PASS_PRIVATE_NIKON_NEF_HUFFMAN_CURVE_UNPACK_CALLABLE"
            and evidence["fixed_identity"]["formal_report_sha256"]
            == execution["producer_report_sha256"]
            and evidence["fixed_identity"]["stable_identity"]
            == execution["producer_stable_identity"],
            "isolated-git-object-import": isolated,
            "callable-identity-exact": module.CALLABLE_ID == config["protocol"],
            "two-output-rows-exact": len(rows) == 2 and all(exact_rows),
            "truncation-controls-reject": all(controls.values()),
            "sources-immutable": source_after == source_before,
        }
        scientific = {
            "protocol": config["protocol"],
            "rows": rows,
            "controls": controls,
            "gates": gates,
        }
        report = {
            "schema": SCHEMA,
            "experiment_id": "P315",
            "status": "PASS_PRIVATE_R1GL_NIKON_NEF_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1GL_NIKON_NEF_NO_COPY_INTAKE",
            "bindings": {
                "artifacts": verified,
                "config_sha256": _sha_bytes(config_bytes),
                "sources": source_before,
            },
            "scientific": scientific,
            "scientific_identity": _sha_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "consumer_core_copied": False,
                "default_loader_changed": False,
                "producer_runner_executed": False,
                "product_mapping": False,
                "source_rights": "raw.pixls.us CC0/Public Domain",
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            failed = sorted(name for name, value in gates.items() if not value)
            raise P315Error(f"P315 gates failed: {', '.join(failed)}")
        return report
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P315Error("temporary residue remains")


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
            {"bytes": len(payload), "sha256": _sha_bytes(payload), "status": report["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
