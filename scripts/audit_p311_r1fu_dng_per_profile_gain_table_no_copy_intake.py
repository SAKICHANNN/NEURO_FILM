#!/usr/bin/env python3
"""Audit the exact source-locked R1FU per-profile PGTM2 parser handoff."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "neuro-film.p311-r1fu-dng-per-profile-gain-table-no-copy-intake.v1"


class P311Error(RuntimeError):
    """Raised when a frozen P311 identity or execution gate fails."""


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
        raise P311Error(f"producer artifact differs: {binding['path']}")
    return value


def _load_isolated_module(site: Path, artifacts: dict[str, bytes]) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile_gain_table2.py").write_bytes(
        artifacts["pgtm2_dependency"]
    )
    (package / "dng_profile_gain_table_set.py").write_bytes(
        artifacts["association_module"]
    )
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.dng_profile_gain_table_set")
    finally:
        sys.path.remove(str(site))


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _profile_summary(binding: Any) -> dict[str, Any]:
    return {
        "role": binding.role,
        "name": binding.name,
        "profile_offset": binding.profile_offset,
        "name_payload_sha256": binding.name_payload_sha256,
        "color_matrix1_sha256": binding.color_matrix1_sha256,
        "forward_matrix1_sha256": binding.forward_matrix1_sha256,
        "tone_curve_sha256": binding.tone_curve_sha256,
        "table_payload_sha256": binding.table_payload_sha256,
        "gains_f32le_sha256": _sha256_bytes(
            np.asarray(binding.table.gains, dtype="<f4").tobytes()
        ),
        "gains_readonly": not binding.table.gains.flags.writeable,
        "weights_readonly": not binding.table.input_weights.flags.writeable,
    }


def _set_summary(value: Any) -> dict[str, Any]:
    return {
        "profiles": [_profile_summary(binding) for binding in value.profiles],
        "extra_profile_offsets": list(value.extra_profile_offsets),
        "raw_ifd_v1_table_sha256": value.raw_ifd_v1_table_sha256,
    }


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P311Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIRST_CONSUMER_SAMPLE_PARSE":
        raise P311Error("P311 is not source locked")

    producer = config["producer"]
    artifacts: dict[str, bytes] = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["artifacts"].items():
        commit = producer[binding["commit_role"]]
        value = _verify_artifact(producer_repo, commit, binding)
        artifacts[name] = value
        verified[name] = {
            "path": binding["path"],
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "sha256": _sha256_bytes(value),
        }

    evidence = json.loads(artifacts["evidence"])
    preregistration = json.loads(artifacts["preregistration"])
    source_binding = config["source"]
    source_path = producer_repo / source_binding["path"]
    source_before = {
        "bytes": source_path.stat().st_size,
        "sha256": _sha256_file(source_path),
    }
    if source_before != {
        "bytes": source_binding["bytes"],
        "sha256": source_binding["sha256"],
    }:
        raise P311Error("source identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p311-"))
    module_names = (
        "zhuise.dng_profile_gain_table_set",
        "zhuise.dng_profile_gain_table2",
        "zhuise",
    )
    try:
        module = _load_isolated_module(temporary / "site", artifacts)
        isolated_import = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        source_bytes = source_path.read_bytes()
        expected = config["expected"]
        table_locks = {
            profile["name"]: profile["table_payload_sha256"]
            for profile in expected["profiles"]
        }

        operations = {
            "parse-bytes": lambda: module.parse_dng_profile_gain_table_set(
                source_bytes,
                expected_table_sha256_by_name=table_locks,
            ),
            "read-path": lambda: module.read_dng_profile_gain_table_set(
                source_path,
                expected_source_sha256=source_binding["sha256"],
                expected_table_sha256_by_name=table_locks,
            ),
        }
        invocation = ["parse-bytes", "read-path"]
        if order == "reverse":
            invocation.reverse()
        summaries = {name: _set_summary(operations[name]()) for name in invocation}

        swapped_locks = {
            expected["profiles"][0]["name"]: expected["profiles"][1][
                "table_payload_sha256"
            ],
            expected["profiles"][1]["name"]: expected["profiles"][0][
                "table_payload_sha256"
            ],
        }
        invalid_magic = bytearray(source_bytes)
        extra_offset = expected["extra_profile_offsets"][0]
        invalid_magic[extra_offset : extra_offset + 4] = b"\0\0\0\0"
        controls = {
            "wrong-source-lock": _expect_value_error(
                lambda: module.read_dng_profile_gain_table_set(
                    source_path, expected_source_sha256="0" * 64
                )
            ),
            "swapped-profile-table-lock": _expect_value_error(
                lambda: module.parse_dng_profile_gain_table_set(
                    source_bytes,
                    expected_table_sha256_by_name=swapped_locks,
                )
            ),
            "missing-profile-lock": _expect_value_error(
                lambda: module.parse_dng_profile_gain_table_set(
                    source_bytes,
                    expected_table_sha256_by_name={
                        expected["profiles"][0]["name"]: expected["profiles"][0][
                            "table_payload_sha256"
                        ]
                    },
                )
            ),
            "truncated-extra-profile": _expect_value_error(
                lambda: module.parse_dng_profile_gain_table_set(source_bytes[:7500])
            ),
            "invalid-byte-order": _expect_value_error(
                lambda: module.parse_dng_profile_gain_table_set(
                    b"ZZ" + source_bytes[2:]
                )
            ),
            "invalid-extended-profile-magic": _expect_value_error(
                lambda: module.parse_dng_profile_gain_table_set(bytes(invalid_magic))
            ),
        }

        canonical_expected_profiles = [
            dict(profile, gains_readonly=True, weights_readonly=True)
            for profile in expected["profiles"]
        ]
        expected_summary = {
            "profiles": canonical_expected_profiles,
            "extra_profile_offsets": expected["extra_profile_offsets"],
            "raw_ifd_v1_table_sha256": expected["raw_ifd_v1_table_sha256"],
        }
        source_after = {
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "evidence-decision-exact": evidence["decision"] == expected["decision"],
            "evidence-report-identity-exact": evidence["execution"][
                "forward_report"
            ]
            == {
                "bytes": expected["producer_report_bytes"],
                "path": "outputs/reports/r1fu_dng_per_profile_gain_table_forward.json",
                "sha256": expected["producer_report_sha256"],
            },
            "evidence-scientific-identity-exact": evidence["execution"][
                "scientific_stable_identity"
            ]
            == f"sha256:{expected['producer_scientific_identity']}",
            "preregistration-protocol-exact": preregistration["protocol"]
            == config["protocol"],
            "isolated-git-object-import": isolated_import,
            "frozen-slotted-profile-set": dataclasses.is_dataclass(
                module.DngProfileGainTableSet
            )
            and module.DngProfileGainTableSet.__dataclass_params__.frozen
            and hasattr(module.DngProfileGainTableSet, "__slots__"),
            "parse-and-read-summaries-exact": all(
                summary == expected_summary for summary in summaries.values()
            ),
            "all-invalid-controls-reject": all(controls.values()),
            "source-file-immutable": source_after == source_before,
        }
        scientific = {
            "controls": controls,
            "gates": gates,
            "profile_set": expected_summary,
            "protocol": config["protocol"],
        }
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "P311",
            "status": "PASS_PRIVATE_R1FU_DNG_PER_PROFILE_GAIN_TABLE_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1FU_DNG_PER_PROFILE_GAIN_TABLE_NO_COPY_INTAKE",
            "bindings": {
                "artifacts": verified,
                "config": {
                    "bytes": len(config_bytes),
                    "sha256": _sha256_bytes(config_bytes),
                },
                "producer_head": producer["repo_head"],
                "source": source_before,
            },
            "scientific": scientific,
            "scientific_identity": _sha256_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "consumer_core_copied": False,
                "image_pixels_decoded": 0,
                "product_mapping": False,
                "public_capability": False,
                "sdk_renders_executed": 0,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            raise P311Error("one or more P311 gates failed")
        return report
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P311Error("temporary residue remains")


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
