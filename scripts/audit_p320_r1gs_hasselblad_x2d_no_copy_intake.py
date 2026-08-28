#!/usr/bin/env python3
"""Audit exact R1GS Hasselblad X2D unpacking without copying producer code."""

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

SCHEMA = "neuro-film.p320-r1gs-hasselblad-x2d-no-copy-intake.v1"


class P320Error(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


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
        raise P320Error(f"producer artifact differs: {binding['path']}")
    return value


def _load(site: Path, module_bytes: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "hasselblad_x2d.py").write_bytes(module_bytes)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.hasselblad_x2d")
    finally:
        sys.path.remove(str(site))


def _reject_truncation(module: Any, source: bytes) -> bool:
    try:
        module.decode_hasselblad_x2d(source[:-1])
    except ValueError:
        return True
    return False


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P320Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIRST_CONSUMER_DECODE":
        raise P320Error("P320 is not frozen")

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

    preregistration = json.loads(artifacts["preregistration"])
    source_lock = json.loads(artifacts["source_lock"])
    contract = json.loads(artifacts["callable_contract"])
    fixture = json.loads(artifacts["canonical_fixture"])
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
            raise P320Error(f"source differs: {row['id']}")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p320-"))
    module_names = ("zhuise.hasselblad_x2d", "zhuise")
    try:
        module = _load(temporary / "site", artifacts["callable_module"])
        isolated = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        invocation = (
            list(reversed(source_bindings)) if order == "reverse" else source_bindings
        )
        rows: list[dict[str, Any]] = []
        controls: dict[str, bool] = {}
        for binding in invocation:
            path = producer_repo / binding["path"]
            source = path.read_bytes()
            source_hash = _sha_bytes(source)
            metadata = module.parse_hasselblad_x2d_metadata(source)
            output = module.decode_hasselblad_x2d(source)
            rows.append(
                {
                    "id": binding["id"],
                    "source_bytes": len(source),
                    "source_sha256": source_hash,
                    "geometry_hw": [int(value) for value in output.shape],
                    "container": str(metadata.container),
                    "compression": int(metadata.compression),
                    "output_dtype": str(output.dtype),
                    "output_uint16le_sha256": _sha_bytes(
                        np.asarray(output, dtype="<u2").tobytes()
                    ),
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
            del output, source
        order_index = {row["id"]: index for index, row in enumerate(source_bindings)}
        rows.sort(key=lambda row: order_index[row["id"]])

        fixture_by_id = {row["id"]: row for row in fixture["rows"]}
        exact_rows = []
        for row, binding in zip(rows, source_bindings, strict=True):
            fixture_row = fixture_by_id[binding["id"]]
            exact_rows.append(
                row["source_bytes"] == binding["bytes"] == fixture_row["source_bytes"]
                and row["source_sha256"]
                == binding["sha256"]
                == fixture_row["source_sha256"]
                and row["geometry_hw"]
                == binding["geometry_hw"]
                == fixture_row["geometry_hw"]
                and row["container"]
                == binding["container"]
                == fixture_row["container"]
                and row["compression"]
                == binding["compression"]
                == fixture_row["compression"]
                and row["output_dtype"] == "uint16"
                and row["output_uint16le_sha256"]
                == binding["output_uint16le_sha256"]
                == fixture_row["output_sha256"]
                and row["minimum_code"] >= 0
                and row["maximum_code"] <= 65535
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
        core_binding = config["artifacts"]["callable_module"]
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "producer-contract-exact": contract["callable"]["id"]
            == config["protocol"]
            and contract["source_lock"]["implementation_commit"]
            == config["producer"]["implementation_commit"],
            "producer-locks-exact": preregistration["status"]
            == "FROZEN_BEFORE_SOURCE_PAYLOAD_ACQUISITION"
            and source_lock["status"]
            == "SOURCE_CONTAINER_AND_DECODER_BRANCHES_LOCKED_BEFORE_RAW_SAMPLE_DECODE"
            and execution_lock["fixed_identity"]["git_blobs"]["core"]
            == core_binding["git_blob"],
            "producer-evidence-exact": evidence["status"]
            == "PASS_PRIVATE_HASSELBLAD_X2D_DUAL_CONTAINER_UNPACK_CALLABLE"
            and evidence["fixed_identity"]["formal_report_sha256"]
            == execution["producer_report_sha256"]
            and evidence["fixed_identity"]["stable_identity"]
            == execution["producer_stable_identity"]
            and evidence["fixed_identity"]["core_git_blob"]
            == core_binding["git_blob"],
            "isolated-git-object-import": isolated,
            "callable-identity-exact": module.CALLABLE_ID == config["protocol"],
            "four-output-rows-exact": len(rows) == 4 and all(exact_rows),
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
            "experiment_id": "P320",
            "status": "PASS_PRIVATE_R1GS_HASSELBLAD_X2D_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1GS_HASSELBLAD_X2D_NO_COPY_INTAKE",
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
                "producer_reference_decoder_executed": False,
                "product_mapping": False,
                "source_rights": "raw.pixls.us CC0/Public Domain",
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            failed = sorted(name for name, value in gates.items() if not value)
            raise P320Error(f"P320 gates failed: {', '.join(failed)}")
        return report
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P320Error("temporary residue remains")


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
