#!/usr/bin/env python3
"""Audit exact R1FV PGTM2 stage arithmetic without copying producer code."""

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

SCHEMA = "neuro-film.p312-r1fv-dng-profile-gain-table2-stage-no-copy-intake.v1"


class P312Error(RuntimeError):
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
        raise P312Error(f"producer artifact differs: {binding['path']}")
    return value


def _load(site: Path, artifacts: dict[str, bytes]) -> tuple[Any, Any]:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile_gain_table2.py").write_bytes(artifacts["stage_module"])
    (package / "dng_profile_gain_table_set.py").write_bytes(
        artifacts["association_module"]
    )
    sys.path.insert(0, str(site))
    try:
        stage = importlib.import_module("zhuise.dng_profile_gain_table2")
        association = importlib.import_module("zhuise.dng_profile_gain_table_set")
        return stage, association
    finally:
        sys.path.remove(str(site))


def _probe() -> np.ndarray:
    row, col = np.indices((13, 17), dtype=np.int32)
    red = (row * 13 + col * 7 - 35).astype(np.float32) / np.float32(80.0)
    green = ((row * 11 + col * 17) % 137 - 20).astype(np.float32) / np.float32(90.0)
    blue = ((row * 19 + col * 5) % 149 - 25).astype(np.float32) / np.float32(95.0)
    return np.ascontiguousarray(np.stack((red, green, blue), axis=-1))


def _reject(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P312Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIRST_CONSUMER_STAGE_EXECUTION":
        raise P312Error("P312 is not frozen")

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

    evidence = json.loads(artifacts["evidence"])
    preregistration = json.loads(artifacts["preregistration"])
    execution_lock = json.loads(artifacts["execution_lock"])
    source_binding = config["source"]
    source_path = producer_repo / source_binding["path"]
    source_before = {"bytes": source_path.stat().st_size, "sha256": _sha_file(source_path)}
    if source_before != {key: source_binding[key] for key in ("bytes", "sha256")}:
        raise P312Error("source differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p312-"))
    module_names = (
        "zhuise.dng_profile_gain_table_set",
        "zhuise.dng_profile_gain_table2",
        "zhuise",
    )
    try:
        stage, association = _load(temporary / "site", artifacts)
        isolated = all(
            Path(module.__file__).resolve().is_relative_to((temporary / "site").resolve())
            for module in (stage, association)
        )
        data = source_path.read_bytes()
        table_locks = {
            row["name"]: row["table_payload_sha256"] for row in source_binding["profiles"]
        }
        profile_set = association.parse_dng_profile_gain_table_set(
            data, expected_table_sha256_by_name=table_locks
        )
        by_name = {binding.name: binding.table for binding in profile_set.profiles}
        main = by_name[source_binding["profiles"][0]["name"]]
        extra = by_name[source_binding["profiles"][1]["name"]]
        gamma2 = dataclasses.replace(main, gamma=2.0)
        probe = _probe()
        probe_before = _sha_bytes(np.asarray(probe, dtype="<f4").tobytes())
        execution = config["execution"]
        cases = [
            ("main-profile-strict", main, False),
            ("main-profile-overrange", main, True),
            ("extra-profile-strict", extra, False),
            ("extra-profile-overrange", extra, True),
            ("gamma2-strict", gamma2, False),
            ("gamma2-overrange", gamma2, True),
        ]
        invocation = list(reversed(cases)) if order == "reverse" else cases
        rows = []
        for role, gain_map, overrange in invocation:
            output = stage.apply_profile_gain_table_map_v2(
                probe,
                gain_map,
                exposure_weight_gain=execution["exposure_weight_gain"],
                image_area=tuple(execution["image_area"]),
                support_overrange=overrange,
            )
            rows.append(
                {
                    "role": role,
                    "gamma": float(gain_map.gamma),
                    "support_overrange": overrange,
                    "minimum": float(np.min(output)),
                    "maximum": float(np.max(output)),
                    "output_f32le_sha256": _sha_bytes(
                        np.asarray(output, dtype="<f4").tobytes()
                    ),
                    "float32_finite": output.dtype == np.float32
                    and bool(np.isfinite(output).all()),
                    "owned_writable_c_contiguous": bool(
                        output.flags.owndata
                        and output.flags.writeable
                        and output.flags.c_contiguous
                    ),
                }
            )
        order_index = {row["role"]: index for index, row in enumerate(execution["expected_rows"])}
        rows.sort(key=lambda row: order_index[row["role"]])

        wrong_locks = dict(table_locks)
        names = list(wrong_locks)
        wrong_locks[names[0]], wrong_locks[names[1]] = wrong_locks[names[1]], wrong_locks[names[0]]
        controls = {
            "nonfinite-rgb": _reject(
                lambda: stage.apply_profile_gain_table_map_v2(
                    np.full((1, 1, 3), np.nan, dtype=np.float32), main
                )
            ),
            "nonpositive-exposure": _reject(
                lambda: stage.apply_profile_gain_table_map_v2(
                    probe, main, exposure_weight_gain=0.0
                )
            ),
            "mismatched-image-area": _reject(
                lambda: stage.apply_profile_gain_table_map_v2(
                    probe, main, image_area=(0, 0, 1, 1)
                )
            ),
            "invalid-gamma": _reject(
                lambda: stage.apply_profile_gain_table_map_v2(
                    probe, dataclasses.replace(main, gamma=0.0)
                )
            ),
            "cross-profile-table-lock": _reject(
                lambda: association.parse_dng_profile_gain_table_set(
                    data, expected_table_sha256_by_name=wrong_locks
                )
            ),
        }
        expected_rows = execution["expected_rows"]
        exact_rows = [
            {key: row[key] for key in expected}
            == expected
            and row["float32_finite"]
            and row["owned_writable_c_contiguous"]
            for row, expected in zip(rows, expected_rows, strict=True)
        ]
        source_after = {"bytes": source_path.stat().st_size, "sha256": _sha_file(source_path)}
        gates = {
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "producer-evidence-exact": evidence["decision"].startswith("PASS_PRIVATE_DNG_PROFILE_GAIN_TABLE2")
            and evidence["execution"]["forward_report"]["sha256"]
            == execution["producer_report_sha256"]
            and evidence["execution"]["scientific_stable_identity"]
            == f"sha256:{execution['producer_scientific_identity']}",
            "producer-locks-exact": preregistration["protocol"] == config["protocol"]
            and execution_lock["implementation_commit"]
            == config["producer"]["implementation_commit"],
            "isolated-git-object-import": isolated,
            "probe-identity-exact": probe_before == execution["probe_f32le_sha256"],
            "six-output-rows-exact": all(exact_rows),
            "input-immutable": probe_before
            == _sha_bytes(np.asarray(probe, dtype="<f4").tobytes()),
            "all-invalid-controls-reject": all(controls.values()),
            "gamma2-remains-nondiscriminating": execution["gamma2_discriminating"] is False
            and rows[0]["output_f32le_sha256"] == rows[4]["output_f32le_sha256"]
            and rows[1]["output_f32le_sha256"] == rows[5]["output_f32le_sha256"],
            "source-immutable": source_after == source_before,
        }
        scientific = {"protocol": config["protocol"], "rows": rows, "controls": controls, "gates": gates}
        report = {
            "schema": SCHEMA,
            "experiment_id": "P312",
            "status": "PASS_PRIVATE_R1FV_DNG_PROFILE_GAIN_TABLE2_STAGE_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1FV_DNG_PROFILE_GAIN_TABLE2_STAGE_NO_COPY_INTAKE",
            "bindings": {"artifacts": verified, "config_sha256": _sha_bytes(config_bytes), "source": source_before},
            "scientific": scientific,
            "scientific_identity": _sha_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "consumer_core_copied": False,
                "general_gamma_claim": False,
                "image_pixels_decoded": 0,
                "product_mapping": False,
                "sdk_oracle_executed": False,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            failed = sorted(name for name, value in gates.items() if not value)
            raise P312Error(f"P312 gates failed: {', '.join(failed)}")
        return report
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P312Error("temporary residue remains")


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
    print(json.dumps({"bytes": len(payload), "sha256": _sha_bytes(payload), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
