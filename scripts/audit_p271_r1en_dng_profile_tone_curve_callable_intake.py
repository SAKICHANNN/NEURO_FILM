#!/usr/bin/env python3
"""Audit the source-locked R1EN ProfileToneCurve callable handoff."""

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

REPORT_SCHEMA = "neuro-film.p271-r1en-dng-profile-tone-curve-callable-intake.v1"


class P271Error(RuntimeError):
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


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    return _sha256(np.ascontiguousarray(value, dtype=dtype).tobytes())


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
    producer_repo: Path, commit: str, binding: dict[str, Any]
) -> bytes:
    value = _git_bytes(producer_repo, commit, binding["path"])
    observed: dict[str, object] = {
        "bytes": len(value),
        "git_blob": _git_blob(producer_repo, commit, binding["path"]),
    }
    if "sha256" in binding:
        observed["sha256"] = _sha256(value)
    if observed != {key: binding[key] for key in observed}:
        raise P271Error(f"producer artifact differs: {binding['path']}")
    return value


def _load_isolated_callable(site: Path, core: bytes, wrapper: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile.py").write_bytes(core)
    (package / "dng_profile_tone_curve_callable.py").write_bytes(wrapper)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.dng_profile_tone_curve_callable")
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
    elif name == "odd-coordinate-count":
        candidate["coordinates"] = candidate["coordinates"][:-1]
    elif name == "too-few-points":
        candidate["coordinates"] = [0.0, 0.0]
    elif name == "coordinate-out-of-range":
        candidate["coordinates"][0] = -0.1
    elif name == "nonincreasing-x":
        candidate["coordinates"][2] = candidate["coordinates"][0]
    elif name == "boolean-coordinate":
        candidate["coordinates"][0] = True
    elif name == "nonfinite-coordinate":
        candidate["coordinates"][0] = float("nan")
    else:
        raise AssertionError(f"unknown payload control: {name}")
    return candidate


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P271Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT":
        raise P271Error("P271 is not source locked")

    producer = config["producer"]
    artifacts = config["artifacts"]
    verified: dict[str, dict[str, Any]] = {}
    artifact_bytes: dict[str, bytes] = {}
    for name, binding in artifacts.items():
        value = _verify_artifact(
            producer_repo, producer[binding["commit_role"]], binding
        )
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
    source = np.asarray(fixture["input_linear_romm"], dtype=np.float64)
    expected = np.asarray(fixture["expected_output_linear_romm"], dtype=np.float64)
    source_before = source.copy()
    payload_before = copy.deepcopy(fixture["payload"])

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p271-"))
    gates: dict[str, bool] = {}
    try:
        module = _load_isolated_callable(
            temporary / "site",
            artifact_bytes["arithmetic_core"],
            artifact_bytes["callable"],
        )
        apply = module.apply_dng_profile_tone_curve_callable_v1
        parse = module.parse_dng_profile_tone_curve_payload_v1
        direct = importlib.import_module("zhuise.dng_profile").apply_profile_tone_curve
        isolated_import = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        names = [
            "main",
            "direct-core-parity",
            "scalar-clamp",
            "nonmonotone-y",
            "nonfinite-input",
            "unexpected-field",
            "wrong-schema",
            "odd-coordinate-count",
            "too-few-points",
            "coordinate-out-of-range",
            "nonincreasing-x",
            "boolean-coordinate",
            "nonfinite-coordinate",
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
                    direct(source, parse(fixture["payload"])),
                )
            elif name == "scalar-clamp":
                scalar = apply(-2.0, fixture["payload"])
                high = apply(2.0, fixture["payload"])
                controls[name] = bool(
                    scalar.shape == ()
                    and high.shape == ()
                    and scalar.item() == expected[0, 0]
                    and high.item() == expected[-1, -1]
                )
            elif name == "nonmonotone-y":
                payload = {
                    "schema": config["callable"]["id"],
                    "coordinates": [0.0, 0.0, 0.5, 0.8, 1.0, 0.2],
                }
                samples = np.asarray([0.0, 0.5, 1.0])
                candidate = apply(samples, payload)
                controls[name] = bool(
                    np.all(np.isfinite(candidate))
                    and np.array_equal(candidate, direct(samples, parse(payload)))
                )
            elif name == "nonfinite-input":
                controls[name] = _expect_value_error(
                    lambda: apply(np.asarray([np.nan]), fixture["payload"])
                )
            else:
                invalid = _payload_control(fixture["payload"], name)
                controls[name] = _expect_value_error(lambda invalid=invalid: apply(source, invalid))
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
        coordinates_hash = _array_sha256(
            np.asarray(fixture["payload"]["coordinates"], dtype=np.float32), "<f4"
        )
        execution_lock = json.loads(artifact_bytes["execution_lock"])
        evidence = json.loads(artifact_bytes["evidence"])
        gates.update(
            {
                "all-controls-pass": all(controls.values()),
                "artifact-identities-exact": len(verified) == len(artifacts),
                "callable-id-exact": module.CALLABLE_ID == config["callable"]["id"],
                "fixture-input-hash-exact": _array_sha256(source, "<f8")
                == fixture["input_f64le_sha256"],
                "fixture-output-hash-exact": _array_sha256(output, "<f8")
                == fixture["expected_output_f64le_sha256"],
                "fixture-payload-hash-exact": payload_hash
                == fixture["payload_canonical_sha256"],
                "real-curve-points-exact": coordinates_hash
                == config["producer_summary_bindings"]["real_curve_points_f32le_sha256"],
                "input-and-payload-unchanged": np.array_equal(source, source_before)
                and fixture["payload"] == payload_before,
                "isolated-git-object-import": isolated_import,
                "output-owned-contiguous-writable-float64": bool(
                    output.dtype == np.float64
                    and output.flags.owndata
                    and output.flags.c_contiguous
                    and output.flags.writeable
                ),
                "producer-formal-identities-exact": evidence["fixed_identity"]
                ["formal_report_sha256"]
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
            "fixture_input_f64le_sha256": _array_sha256(source, "<f8"),
            "fixture_output_f64le_sha256": _array_sha256(output, "<f8"),
            "fixture_payload_canonical_sha256": payload_hash,
            "gates": gates,
            "network_reads": 0,
            "new_raw_dng_pixel_target_reads": 0,
            "producer_head": producer["repo_head"],
            "real_curve_points_f32le_sha256": coordinates_hash,
        }
    finally:
        for name in (
            "zhuise.dng_profile_tone_curve_callable",
            "zhuise.dng_profile",
            "zhuise",
        ):
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)

    gates["zero-temporary-residue"] = not temporary.exists()
    scientific["zero_temporary_residue"] = not temporary.exists()
    status = (
        "PASS_PRIVATE_R1EN_DNG_PROFILE_TONE_CURVE_CALLABLE_INTAKE"
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
