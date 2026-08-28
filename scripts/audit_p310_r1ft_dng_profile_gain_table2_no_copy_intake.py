#!/usr/bin/env python3
"""Audit the exact source-locked R1FT PGTM2 parser handoff."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "neuro-film.p310-r1ft-dng-profile-gain-table2-no-copy-intake.v1"


class P310Error(RuntimeError):
    """Raised when a frozen P310 identity or execution gate fails."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _verify_artifact(repo: Path, commit: str, binding: dict[str, Any]) -> bytes:
    value = _git_bytes(repo, commit, binding["path"])
    observed = {
        "bytes": len(value),
        "git_blob": _git_blob(repo, commit, binding["path"]),
        "sha256": _sha256_bytes(value),
    }
    if observed != {key: binding[key] for key in observed}:
        raise P310Error(f"producer artifact differs: {binding['path']}")
    return value


def _load_isolated_module(site: Path, module_bytes: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile_gain_table2.py").write_bytes(module_bytes)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.dng_profile_gain_table2")
    finally:
        sys.path.remove(str(site))


def _extract_payload(path: Path, binding: dict[str, Any]) -> tuple[str, bytes]:
    with path.open("rb") as stream:
        header = stream.read(8)
        if header[:2] == b"II":
            prefix, byteorder = "<", "little"
        elif header[:2] == b"MM":
            prefix, byteorder = ">", "big"
        else:
            raise P310Error("source is not classic TIFF")
        magic, ifd_offset = struct.unpack_from(prefix + "HI", header, 2)
        if magic != 42 or ifd_offset < 8:
            raise P310Error("source TIFF header differs")
        stream.seek(ifd_offset)
        count_bytes = stream.read(2)
        if len(count_bytes) != 2:
            raise P310Error("source TIFF directory is truncated")
        count = struct.unpack(prefix + "H", count_bytes)[0]
        matches: list[tuple[int, int, int]] = []
        for _ in range(count):
            entry = stream.read(12)
            if len(entry) != 12:
                raise P310Error("source TIFF directory is truncated")
            tag, field_type, value_count, value_offset = struct.unpack(
                prefix + "HHII", entry
            )
            if tag in {52525, 52544}:
                if field_type != 7:
                    raise P310Error("gain table tag type differs")
                matches.append((tag, value_count, value_offset))
        expected = (
            binding["tag"],
            binding["payload_bytes"],
            binding["payload_offset"],
        )
        if matches != [expected]:
            raise P310Error("gain table tag envelope differs")
        stream.seek(binding["payload_offset"])
        payload = stream.read(binding["payload_bytes"])
    if len(payload) != binding["payload_bytes"]:
        raise P310Error("gain table payload is truncated")
    if _sha256_bytes(payload) != binding["payload_sha256"]:
        raise P310Error("gain table payload identity differs")
    return byteorder, payload


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _map_summary(parsed: Any, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "points": list(parsed.points),
        "spacing": list(parsed.spacing),
        "origin": list(parsed.origin),
        "weights": [float(value) for value in parsed.input_weights],
        "data_type": parsed.data_type,
        "gamma": parsed.gamma,
        "gain_min": parsed.gain_min,
        "gain_max": parsed.gain_max,
        "table_points": parsed.table_points,
        "gains_f32le_sha256": _sha256_bytes(
            np.asarray(parsed.gains, dtype="<f4").tobytes()
        ),
        "gains_readonly": not parsed.gains.flags.writeable,
        "weights_readonly": not parsed.input_weights.flags.writeable,
    }


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P310Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_AFTER_EXCLUDED_UNCOMMITTED_DRY_RUN_BEFORE_FORMAL":
        raise P310Error("P310 is not source locked")

    producer = config["producer"]
    artifact_bytes: dict[str, bytes] = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["artifacts"].items():
        commit = producer[binding["commit_role"]]
        value = _verify_artifact(producer_repo, commit, binding)
        artifact_bytes[name] = value
        verified[name] = {
            "path": binding["path"],
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "sha256": _sha256_bytes(value),
        }

    evidence = json.loads(artifact_bytes["evidence"])
    preregistration = json.loads(artifact_bytes["preregistration"])
    sources = list(config["sources"])
    invocation = list(reversed(sources)) if order == "reverse" else sources
    source_paths = [producer_repo / binding["path"] for binding in sources]
    source_before = [
        {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in source_paths
    ]
    if source_before != [
        {"bytes": binding["bytes"], "sha256": binding["sha256"]}
        for binding in sources
    ]:
        raise P310Error("one or more source identities differ")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p310-"))
    module_name = "zhuise.dng_profile_gain_table2"
    try:
        module = _load_isolated_module(temporary / "site", artifact_bytes["module"])
        isolated_import = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        summaries: list[dict[str, Any]] = []
        first_payload: bytes | None = None
        for binding in invocation:
            byteorder, payload = _extract_payload(
                producer_repo / binding["path"], binding
            )
            parsed = module.parse_profile_gain_table_map_v2(
                payload, byteorder=byteorder
            )
            summaries.append(_map_summary(parsed, binding["role"]))
            if first_payload is None:
                first_payload = payload
        if first_payload is None:
            raise P310Error("no source payload executed")
        summaries.sort(key=lambda row: row["role"])

        prefix = "<"
        controls = {
            "invalid-byteorder": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload, byteorder="middle"
                )
            ),
            "wrong-endian": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload, byteorder="big"
                )
            ),
            "zero-dimension": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    struct.pack(prefix + "I", 0) + first_payload[4:]
                )
            ),
            "unsupported-data-type": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload[:64]
                    + struct.pack(prefix + "I", 4)
                    + first_payload[68:]
                )
            ),
            "invalid-gamma": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload[:68]
                    + struct.pack(prefix + "f", 0.0)
                    + first_payload[72:]
                )
            ),
            "invalid-gain-bounds": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload[:72]
                    + struct.pack(prefix + "ff", 5.0, 4.0)
                    + first_payload[80:]
                )
            ),
            "nonfinite-spacing": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(
                    first_payload[:8]
                    + struct.pack(prefix + "d", float("inf"))
                    + first_payload[16:]
                )
            ),
            "truncated": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(first_payload[:-1])
            ),
            "trailing": _expect_value_error(
                lambda: module.parse_profile_gain_table_map_v2(first_payload + b"\0")
            ),
        }

        expected = config["expected"]
        common_header_exact = all(
            row["points"] == expected["points"]
            and row["spacing"] == expected["spacing"]
            and row["origin"] == expected["origin"]
            and row["weights"] == expected["weights"]
            and row["gamma"] == expected["gamma"]
            and row["gain_min"] == expected["gain_min"]
            and row["gain_max"] == expected["gain_max"]
            and row["table_points"] == expected["table_points"]
            for row in summaries
        )
        source_after = [
            {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in source_paths
        ]
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "evidence-decision-exact": evidence["decision"] == expected["decision"],
            "evidence-protocol-exact": evidence["protocol"] == config["protocol"],
            "preregistration-protocol-exact": preregistration["protocol"]
            == config["protocol"],
            "producer-report-identity-exact": evidence["execution"]["forward_report"]
            == {
                "bytes": expected["producer_report_bytes"],
                "path": "outputs/reports/r1ft_dng_profile_gain_table2_forward.json",
                "sha256": expected["producer_report_sha256"],
            },
            "producer-scientific-identity-exact": evidence["execution"]
            ["scientific_stable_identity"]
            == f"sha256:{expected['producer_scientific_identity']}",
            "isolated-git-object-import": isolated_import,
            "frozen-slotted-result": dataclasses.is_dataclass(
                module.DngProfileGainTableMapV2
            )
            and module.DngProfileGainTableMapV2.__dataclass_params__.frozen
            and hasattr(module.DngProfileGainTableMapV2, "__slots__"),
            "three-encoding-headers-exact": common_header_exact,
            "three-encoding-gain-hashes-exact": all(
                row["gains_f32le_sha256"] == expected["gains_f32le_sha256"]
                for row in summaries
            ),
            "returned-arrays-readonly": all(
                row["gains_readonly"] and row["weights_readonly"]
                for row in summaries
            ),
            "all-invalid-controls-reject": all(controls.values()),
            "source-files-immutable": source_after == source_before,
        }
        scientific = {
            "controls": controls,
            "gates": gates,
            "maps": summaries,
            "protocol": config["protocol"],
        }
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "P310",
            "status": "PASS_PRIVATE_R1FT_DNG_PROFILE_GAIN_TABLE2_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1FT_DNG_PROFILE_GAIN_TABLE2_NO_COPY_INTAKE",
            "bindings": {
                "artifacts": verified,
                "config": {
                    "bytes": len(config_bytes),
                    "sha256": _sha256_bytes(config_bytes),
                },
                "producer_head": producer["repo_head"],
                "sources": [
                    {
                        "role": binding["role"],
                        "bytes": binding["bytes"],
                        "sha256": binding["sha256"],
                        "payload_sha256": binding["payload_sha256"],
                    }
                    for binding in sources
                ],
            },
            "scientific": scientific,
            "scientific_identity": _sha256_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "consumer_core_copied": False,
                "image_pixels_decoded": 0,
                "product_mapping": False,
                "public_capability": False,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            raise P310Error("one or more P310 gates failed")
        return report
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop("zhuise", None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P310Error("temporary residue remains")


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
                "sha256": _sha256_bytes(payload),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
