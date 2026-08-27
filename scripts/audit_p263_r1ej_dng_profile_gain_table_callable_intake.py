#!/usr/bin/env python3
"""Audit the source-locked R1EJ ProfileGainTableMap callable handoff."""

from __future__ import annotations

import argparse
import copy
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

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "neuro-film.p263-r1ej-dng-profile-gain-table-callable-intake.v1"


class P263Error(RuntimeError):
    """Raised when a frozen source or execution gate fails."""


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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(value, dtype="<f4").tobytes())


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


def _verify_artifact(
    producer_repo: Path,
    commit: str,
    binding: dict[str, Any],
) -> bytes:
    value = _git_bytes(producer_repo, commit, binding["path"])
    observed = {
        "bytes": len(value),
        "git_blob": _git_blob(producer_repo, commit, binding["path"]),
        "sha256": _sha256(value),
    }
    if observed != {key: binding[key] for key in observed}:
        raise P263Error(f"producer artifact differs: {binding['path']}")
    return value


def _load_isolated_callable(site: Path, core: bytes, wrapper: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile_gain_table.py").write_bytes(core)
    (package / "dng_profile_gain_table_callable.py").write_bytes(wrapper)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.dng_profile_gain_table_callable")
    finally:
        sys.path.remove(str(site))


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _payload_control(payload: dict[str, Any], name: str) -> dict[str, Any]:
    candidate = copy.deepcopy(payload)
    if name == "unexpected-field":
        candidate["extra"] = 1
    elif name == "wrong-schema":
        candidate["schema"] = "wrong"
    elif name == "bad-points":
        candidate["points"] = [0, 2]
    elif name == "bad-spacing":
        candidate["spacing"] = [0.0, 0.5]
    elif name == "bad-weight-count":
        candidate["input_weights"] = candidate["input_weights"][:-1]
    elif name == "bad-gain-count":
        candidate["gains"] = candidate["gains"][:-1]
    elif name == "gain-out-of-range":
        candidate["gains"][0] = 4097.0
    elif name == "bad-exposure":
        candidate["exposure_weight_gain"] = 0.0
    elif name == "bad-area":
        candidate["image_area"] = [10, 20, 11, 22]
    elif name == "nonboolean-overrange":
        candidate["support_overrange"] = 1
    else:
        raise AssertionError(f"unknown payload control: {name}")
    return candidate


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P263Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT":
        raise P263Error("P263 is not source locked")
    producer = config["producer"]
    artifacts = config["artifacts"]
    verified: dict[str, dict[str, Any]] = {}
    artifact_bytes: dict[str, bytes] = {}
    for name, binding in artifacts.items():
        commit = producer[binding["commit_role"]]
        value = _verify_artifact(producer_repo, commit, binding)
        artifact_bytes[name] = value
        verified[name] = {
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "path": binding["path"],
            "sha256": _sha256(value),
        }

    fixture = json.loads(artifact_bytes["fixture"])
    schema = json.loads(artifact_bytes["schema"])
    jsonschema.Draft202012Validator(schema).validate(fixture["payload"])
    source = np.asarray(fixture["input_romm_rgb"], dtype=np.float32)
    expected = np.asarray(fixture["expected_output_romm_rgb"], dtype=np.float32)
    source_before = source.copy()
    payload_before = copy.deepcopy(fixture["payload"])

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p263-"))
    gates: dict[str, bool] = {}
    try:
        module = _load_isolated_callable(
            temporary / "site",
            artifact_bytes["arithmetic_core"],
            artifact_bytes["callable"],
        )
        apply = module.apply_dng_profile_gain_table_callable_v1
        parse = module.parse_dng_profile_gain_table_payload_v1
        direct = importlib.import_module(
            "zhuise.dng_profile_gain_table"
        ).apply_profile_gain_table_map_v1
        isolated_import = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        names = [
            "main",
            "direct-core-parity",
            "default-sdr-parity",
            "nonfinite-input",
            "wrong-rank-input",
            "unexpected-field",
            "wrong-schema",
            "bad-points",
            "bad-spacing",
            "bad-weight-count",
            "bad-gain-count",
            "gain-out-of-range",
            "bad-exposure",
            "bad-area",
            "nonboolean-overrange",
        ]
        if order == "reverse":
            names.reverse()
        controls: dict[str, bool] = {}
        output: np.ndarray | None = None
        for name in names:
            if name == "main":
                output = apply(source, fixture["payload"])
                controls[name] = np.array_equal(output, expected)
            elif name == "direct-core-parity":
                controls[name] = np.array_equal(
                    apply(source, fixture["payload"]),
                    direct(source, **parse(fixture["payload"])),
                )
            elif name == "default-sdr-parity":
                payload = copy.deepcopy(fixture["payload"])
                payload["support_overrange"] = False
                controls[name] = np.array_equal(
                    apply(source, payload), direct(source, **parse(payload))
                )
            elif name == "nonfinite-input":
                invalid = source.copy()
                invalid[0, 0, 0] = np.nan
                controls[name] = _expect_value_error(
                    lambda invalid=invalid: apply(invalid, fixture["payload"])
                )
            elif name == "wrong-rank-input":
                controls[name] = _expect_value_error(
                    lambda: apply(source.reshape(-1, 3), fixture["payload"])
                )
            else:
                invalid_payload = _payload_control(fixture["payload"], name)
                controls[name] = _expect_value_error(
                    lambda invalid_payload=invalid_payload: apply(
                        source, invalid_payload
                    )
                )
        if output is None:
            raise AssertionError("main control did not execute")

        payload_hash = _sha256(
            json.dumps(
                fixture["payload"],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )
        execution_lock = json.loads(artifact_bytes["execution_lock"])
        evidence = json.loads(artifact_bytes["evidence"])
        gates.update(
            {
                "all-controls-pass": all(controls.values()),
                "artifact-identities-exact": len(verified) == len(artifacts),
                "callable-id-exact": module.CALLABLE_ID == config["callable"]["id"],
                "fixture-input-hash-exact": _array_sha256(source)
                == fixture["input_f32le_sha256"],
                "fixture-output-hash-exact": _array_sha256(output)
                == fixture["expected_output_f32le_sha256"],
                "fixture-payload-hash-exact": payload_hash
                == fixture["payload_canonical_sha256"],
                "input-and-payload-unchanged": np.array_equal(source, source_before)
                and fixture["payload"] == payload_before,
                "isolated-git-object-import": isolated_import,
                "output-owned-contiguous-writable-float32": bool(
                    output.dtype == np.float32
                    and output.flags.owndata
                    and output.flags.c_contiguous
                    and output.flags.writeable
                ),
                "producer-formal-identities-exact": evidence["fixture"]
                ["output_f32le_sha256"]
                == fixture["expected_output_f32le_sha256"]
                and evidence["formal_execution"]["report_sha256"]
                == config["producer_summary_bindings"]["formal_report_sha256"]
                if "report_sha256" in evidence["formal_execution"]
                else evidence["formal_execution"]["forward_sha256"]
                == config["producer_summary_bindings"]["formal_report_sha256"]
                and evidence["formal_execution"]["reverse_sha256"]
                == config["producer_summary_bindings"]["formal_report_sha256"],
                "producer-execution-lock-exact": execution_lock["status"]
                == "FROZEN_BEFORE_FORMAL_HANDOFF_EXECUTION",
            }
        )
        scientific = {
            "artifacts": dict(sorted(verified.items())),
            "callable_id": module.CALLABLE_ID,
            "consumer_config_sha256": _sha256(config_bytes),
            "controls": dict(sorted(controls.items())),
            "fixture_input_f32le_sha256": _array_sha256(source),
            "fixture_output_f32le_sha256": _array_sha256(output),
            "fixture_payload_canonical_sha256": payload_hash,
            "gates": gates,
            "network_reads": 0,
            "new_raw_dng_pixel_target_reads": 0,
            "producer_head": producer["repo_head"],
        }
    finally:
        for name in (
            "zhuise.dng_profile_gain_table_callable",
            "zhuise.dng_profile_gain_table",
            "zhuise",
        ):
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)

    gates["zero-temporary-residue"] = not temporary.exists()
    scientific["zero_temporary_residue"] = not temporary.exists()
    status = (
        "PASS_PRIVATE_R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_INTAKE"
        if all(gates.values())
        else "FAIL_CLOSED"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scientific": scientific,
        "stable_identity": f"sha256:{_sha256(_canonical(scientific))}",
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve(), args.producer_repo.resolve(), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
