"""Audit the source-locked R1EC DNG ProfileHueSatMap callable handoff."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_profile_huesatmap_audit import (
    parse_profile_huesatmap_exif,
)

REPORT_SCHEMA = "neuro-film.p254-r1ec-dng-profile-huesat-callable-intake.v1"


class P254Error(RuntimeError):
    """Raised when a frozen P254 source or execution gate fails."""


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
    expected = {key: binding[key] for key in observed}
    if observed != expected:
        raise P254Error(f"producer artifact differs: {binding['path']}")
    return value


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _load_isolated_callable(
    site: Path,
    callable_bytes: bytes,
    core_bytes: bytes,
) -> tuple[Any, Any]:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_profile.py").write_bytes(core_bytes)
    (package / "dng_profile_huesat_callable.py").write_bytes(callable_bytes)
    sys.path.insert(0, str(site))
    try:
        callable_module = importlib.import_module("zhuise.dng_profile_huesat_callable")
        core_module = importlib.import_module("zhuise.dng_profile")
    finally:
        sys.path.remove(str(site))
    return callable_module, core_module


def _payload_control(
    payload: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    candidate = copy.deepcopy(payload)
    if name == "nonfinite-table":
        candidate["data_1"][1] = float("nan")
    elif name == "wrong-length-table":
        candidate["data_1"] = candidate["data_1"][:-1]
    elif name == "malformed-dimensions":
        candidate["dimensions"] = [2, 1, 2]
    elif name == "negative-saturation-scale":
        candidate["data_1"][1] = -0.1
    elif name == "negative-value-scale":
        candidate["data_1"][2] = -0.1
    elif name == "nonunit-zero-saturation-value-scale":
        candidate["data_1"][2] = 0.9
    elif name == "invalid-weight-low":
        candidate["calibration_1_weight"] = -0.1
    elif name == "invalid-weight-high":
        candidate["calibration_1_weight"] = 1.1
    elif name == "unsupported-encoding":
        candidate["encoding"] = 1
    elif name == "nonboolean-overrange":
        candidate["support_overrange"] = 1
    elif name == "unexpected-field":
        candidate["extra"] = 1
    else:
        raise AssertionError(f"unknown payload control: {name}")
    return candidate


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P254Error("order must be forward or reverse")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "STAGE_B_SOURCE_LOCKED_READY_FOR_FIXTURE_EXECUTION":
        raise P254Error("P254 Stage B is not source locked")
    stage_a = config["stage_a_bindings"]
    stage_b = config["stage_b_bindings"]
    source_lock = ROOT / stage_b["source_lock_path"]
    if (
        source_lock.stat().st_size != stage_b["source_lock_bytes"]
        or _sha256(source_lock.read_bytes()) != stage_b["source_lock_sha256"]
    ):
        raise P254Error("P254 Stage B source lock differs")

    implementation = stage_b["producer_implementation_commit"]
    artifacts = stage_b["artifacts"]
    verified: dict[str, dict[str, Any]] = {}
    artifact_bytes: dict[str, bytes] = {}
    for name, binding in artifacts.items():
        commit = (
            stage_b["producer_evidence_commit"]
            if name == "evidence"
            else implementation
        )
        value = _verify_artifact(producer_repo, commit, binding)
        artifact_bytes[name] = value
        verified[name] = {
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "path": binding["path"],
            "sha256": _sha256(value),
        }

    fixture = json.loads(artifact_bytes["fixture"].decode("utf-8"))
    payload_schema = json.loads(artifact_bytes["schema"].decode("utf-8"))
    jsonschema.Draft202012Validator(payload_schema).validate(fixture["payload"])
    source = np.asarray(fixture["input_xyz_d50"], dtype=np.float64)
    source_before = source.copy()
    payload_before = copy.deepcopy(fixture["payload"])

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p254-"))
    try:
        callable_module, core_module = _load_isolated_callable(
            temporary / "site",
            artifact_bytes["callable"],
            artifact_bytes["arithmetic_dependency"],
        )
        apply = callable_module.apply_dng_profile_huesat_callable_v1
        parse = callable_module.parse_dng_profile_huesat_payload_v1
        direct = core_module.apply_profile_hue_sat_map_xyz_d50
        module_path = Path(callable_module.__file__).resolve()
        isolated_import = module_path.is_relative_to((temporary / "site").resolve())
        expected = np.asarray(fixture["expected_output_xyz_d50"], dtype=np.float64)

        names = [
            "main",
            "default-sdr-parity",
            "single-table-parity",
            "nonfinite-input",
            "wrong-rank-input",
            "nonfinite-table",
            "wrong-length-table",
            "malformed-dimensions",
            "negative-saturation-scale",
            "negative-value-scale",
            "nonunit-zero-saturation-value-scale",
            "invalid-weight-low",
            "invalid-weight-high",
            "unsupported-encoding",
            "nonboolean-overrange",
            "unexpected-field",
        ]
        if order == "reverse":
            names.reverse()
        controls: dict[str, bool] = {}
        output: np.ndarray | None = None
        for name in names:
            if name == "main":
                output = apply(source, fixture["payload"])
                controls[name] = np.array_equal(output, expected)
            elif name == "default-sdr-parity":
                safe = np.asarray([[[0.18, 0.19, 0.20]]], dtype=np.float64)
                payload = copy.deepcopy(fixture["payload"])
                payload["support_overrange"] = False
                parsed = parse(payload)
                controls[name] = np.array_equal(apply(safe, payload), direct(safe, **parsed))
            elif name == "single-table-parity":
                payload = copy.deepcopy(fixture["payload"])
                payload["data_2"] = None
                parsed = parse(payload)
                controls[name] = np.array_equal(apply(source, payload), direct(source, **parsed))
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
                    lambda invalid_payload=invalid_payload: apply(source, invalid_payload)
                )
        if output is None:
            raise AssertionError("main callable control did not execute")

        exif = ROOT / stage_a["consumer_dji_exif_path"]
        profile = parse_profile_huesatmap_exif(exif.read_text(encoding="utf-8"))
        local_metadata = {
            "data1_f32le_sha256": _array_sha256(profile.data1, "<f4"),
            "data2_f32le_sha256": _array_sha256(profile.data2, "<f4"),
            "dimensions": list(profile.dimensions),
            "encoding": profile.encoding,
            "exif_bytes": exif.stat().st_size,
            "exif_sha256": _sha256(exif.read_bytes()),
        }
        local_metadata_exact = local_metadata == {
            "data1_f32le_sha256": stage_a["consumer_dji_data1_f32le_sha256"],
            "data2_f32le_sha256": stage_a["consumer_dji_data2_f32le_sha256"],
            "dimensions": [6, 6, 3],
            "encoding": 0,
            "exif_bytes": stage_a["consumer_dji_exif_bytes"],
            "exif_sha256": stage_a["consumer_dji_exif_sha256"],
        }
        output_hash = _array_sha256(output, "<f8")
        payload_hash = _sha256(
            json.dumps(
                fixture["payload"],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )
        gates = {
            "all-controls-pass": all(controls.values()),
            "artifact-identities-exact": len(verified) == len(artifacts),
            "fixture-input-hash-exact": _array_sha256(source, "<f8")
            == stage_b["fixture_identities"]["input_f64le_sha256"],
            "fixture-output-hash-exact": output_hash
            == stage_b["fixture_identities"]["output_f64le_sha256"],
            "fixture-payload-hash-exact": payload_hash
            == stage_b["fixture_identities"]["payload_canonical_sha256"],
            "input-and-payload-unchanged": np.array_equal(source, source_before)
            and fixture["payload"] == payload_before,
            "isolated-git-object-import": isolated_import,
            "local-metadata-exact": local_metadata_exact,
            "output-owned-contiguous-writable-f64": bool(
                output.dtype == np.float64
                and output.flags.owndata
                and output.flags.c_contiguous
                and output.flags.writeable
            ),
        }
        scientific = {
            "artifacts": dict(sorted(verified.items())),
            "callable_id": callable_module.CALLABLE_ID,
            "controls": dict(sorted(controls.items())),
            "fixture_input_f64le_sha256": _array_sha256(source, "<f8"),
            "fixture_output_f64le_sha256": output_hash,
            "fixture_payload_canonical_sha256": payload_hash,
            "gates": gates,
            "local_metadata": local_metadata,
            "network_reads": 0,
            "new_raw_dng_pixel_target_reads": 0,
            "producer_implementation_commit": implementation,
        }
    finally:
        for name in ("zhuise.dng_profile_huesat_callable", "zhuise.dng_profile", "zhuise"):
            sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)

    scientific["zero_temporary_residue"] = not temporary.exists()
    gates["zero-temporary-residue"] = scientific["zero_temporary_residue"]
    status = (
        "PASS_PRIVATE_R1EC_DNG_PROFILE_HUESAT_CALLABLE_INTAKE"
        if all(gates.values())
        else "FAIL_CLOSED"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scientific": scientific,
        "stable_identity": f"sha256:{_sha256(_canonical(scientific))}",
        "claim_ceiling": config["stage_b_bindings"]["rights_and_claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(
        args.config.resolve(), args.producer_repo.resolve(), args.order
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
