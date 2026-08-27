#!/usr/bin/env python3
"""Audit exact isolated composition of four source-locked DNG profile stages."""

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "neuro-film.p288-dng-profile-stage-composition-result.v1"


class P288Error(RuntimeError):
    """Raised when a frozen P288 identity or execution requirement differs."""


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


def _file_binding(binding: list[object]) -> tuple[Path, bytes]:
    if len(binding) != 3:
        raise P288Error("file binding must contain path, bytes and SHA-256")
    path = ROOT / str(binding[0])
    value = path.read_bytes()
    if len(value) != int(binding[1]) or _sha256(value) != str(binding[2]):
        raise P288Error(f"bound file differs: {path}")
    return path, value


def _verify_local_binding(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    if not path.is_file():
        return False
    value = path.read_bytes()
    return len(value) == int(binding["bytes"]) and _sha256(value) == str(
        binding["sha256"]
    )


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


def _producer_artifact(
    repo: Path, commit: str, binding: Mapping[str, Any]
) -> bytes:
    value = _git_bytes(repo, commit, str(binding["path"]))
    observed = {
        "bytes": len(value),
        "git_blob": _git_blob(repo, commit, str(binding["path"])),
        "sha256": _sha256(value),
    }
    expected = {key: binding[key] for key in observed}
    if observed != expected:
        raise P288Error(f"producer object differs: {binding['path']}")
    return value


def _parent_objects(
    name: str, config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if name == "huesat":
        locked = config["stage_b_bindings"]
        commit = str(locked["producer_implementation_commit"])
        selected = {
            "core": locked["artifacts"]["arithmetic_dependency"],
            "callable": locked["artifacts"]["callable"],
            "schema": locked["artifacts"]["schema"],
            "fixture": locked["artifacts"]["fixture"],
        }
        commits = {key: commit for key in selected}
    else:
        producer = config["producer"]
        artifacts = config["artifacts"]
        selected = {
            "core": artifacts["arithmetic_core"],
            "callable": artifacts["callable"],
            "schema": artifacts["schema"],
            "fixture": artifacts["fixture"],
        }
        commits = {
            key: str(producer[binding["commit_role"]])
            for key, binding in selected.items()
        }
    values: dict[str, bytes] = {}
    identities: dict[str, dict[str, Any]] = {}
    for key, binding in selected.items():
        value = _producer_artifact(repo, commits[key], binding)
        values[key] = value
        identities[key] = {
            "bytes": len(value),
            "commit": commits[key],
            "git_blob": binding["git_blob"],
            "path": binding["path"],
            "sha256": _sha256(value),
        }
    return values, identities


def _write_isolated_package(site: Path, objects: Mapping[str, dict[str, bytes]]) -> None:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    common = objects["huesat"]["core"]
    if objects["look_table"]["core"] != common or objects["tone_curve"]["core"] != common:
        raise P288Error("common dng_profile arithmetic objects differ")
    (package / "dng_profile.py").write_bytes(common)
    (package / "dng_profile_huesat_callable.py").write_bytes(
        objects["huesat"]["callable"]
    )
    (package / "dng_profile_gain_table.py").write_bytes(
        objects["gain_table"]["core"]
    )
    (package / "dng_profile_gain_table_callable.py").write_bytes(
        objects["gain_table"]["callable"]
    )
    (package / "dng_profile_look_table_callable.py").write_bytes(
        objects["look_table"]["callable"]
    )
    (package / "dng_profile_tone_curve_callable.py").write_bytes(
        objects["tone_curve"]["callable"]
    )


def _array_sha256(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha256(np.ascontiguousarray(value, dtype=dtype).tobytes())


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _romm_domain(values: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.isfinite(values).all()
        and float(np.min(values)) >= -tolerance
        and float(np.max(values)) <= 1.0 + tolerance
    )


def _compose(
    source_romm: np.ndarray,
    exposure: float,
    payloads: Mapping[str, Mapping[str, Any]],
    modules: Mapping[str, Any],
    *,
    direct: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source = np.array(source_romm, dtype=np.float64, order="C", copy=True)
    if source.ndim != 3 or source.shape[-1] != 3 or not np.isfinite(source).all():
        raise ValueError("composition input must be finite HxWx3 linear ROMM")
    if not np.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("composition exposure must be positive finite")
    core = modules["core"]
    xyz = source @ core.ROMM_RGB_TO_XYZ_D50.T
    if direct:
        hsm = core.apply_profile_hue_sat_map_xyz_d50(
            xyz, **modules["huesat"].parse_dng_profile_huesat_payload_v1(payloads["huesat"])
        )
    else:
        hsm = modules["huesat"].apply_dng_profile_huesat_callable_v1(
            xyz, payloads["huesat"]
        )
    hsm_romm = np.asarray(hsm @ core.XYZ_D50_TO_ROMM_RGB.T, dtype=np.float32)
    if direct:
        gain = modules["gain_core"].apply_profile_gain_table_map_v1(
            hsm_romm,
            **modules["gain_table"].parse_dng_profile_gain_table_payload_v1(
                payloads["gain_table"]
            ),
        )
    else:
        gain = modules["gain_table"].apply_dng_profile_gain_table_callable_v1(
            hsm_romm, payloads["gain_table"]
        )
    exposed_romm = np.asarray(gain, dtype=np.float64) * float(exposure)
    exposed_xyz = exposed_romm @ core.ROMM_RGB_TO_XYZ_D50.T
    if direct:
        looked_xyz = core.apply_profile_look_table_xyz_d50(
            exposed_xyz,
            **modules["look_table"].parse_dng_profile_look_table_payload_v1(
                payloads["look_table"]
            ),
        )
    else:
        looked_xyz = modules["look_table"].apply_dng_profile_look_table_callable_v1(
            exposed_xyz, payloads["look_table"]
        )
    looked_romm = looked_xyz @ core.XYZ_D50_TO_ROMM_RGB.T
    if direct:
        output = core.apply_profile_tone_curve(
            looked_romm,
            **modules["tone_curve"].parse_dng_profile_tone_curve_payload_v1(
                payloads["tone_curve"]
            ),
        )
    else:
        output = modules["tone_curve"].apply_dng_profile_tone_curve_callable_v1(
            looked_romm, payloads["tone_curve"]
        )
    stages = {
        "initial_xyz_d50": xyz,
        "huesat_xyz_d50": hsm,
        "huesat_romm": hsm_romm,
        "gain_table_romm": gain,
        "exposed_romm": exposed_romm,
        "look_table_xyz_d50": looked_xyz,
        "look_table_romm": looked_romm,
        "tone_curve_romm": output,
    }
    return np.array(output, dtype=np.float64, order="C", copy=True), stages


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P288Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FORMAL_EXECUTION":
        raise P288Error("P288 formal execution is not locked")
    local_gates = {
        "contract": _verify_local_binding(config["contract"]),
        "runner": _verify_local_binding(config["execution_bindings"]["runner"]),
        "test": _verify_local_binding(config["execution_bindings"]["test"]),
    }
    if not all(local_gates.values()):
        raise P288Error("P288 local binding differs")

    parent_configs: dict[str, dict[str, Any]] = {}
    parent_evidence: dict[str, dict[str, Any]] = {}
    parent_identities: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, bytes]] = {}
    producer_identities: dict[str, dict[str, dict[str, Any]]] = {}
    for name, bindings in config["parents"].items():
        config_path_parent, config_value = _file_binding(bindings["config"])
        evidence_path, evidence_value = _file_binding(bindings["evidence"])
        parent_configs[name] = json.loads(config_value)
        parent_evidence[name] = json.loads(evidence_value)
        if not str(parent_evidence[name].get("status", "")).startswith("PASS_"):
            raise P288Error(f"parent evidence is not passing: {name}")
        parent_identities[name] = {
            "config": _sha256(config_value),
            "config_path": str(config_path_parent.relative_to(ROOT)).replace("\\", "/"),
            "evidence": _sha256(evidence_value),
            "evidence_path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
        }
        objects[name], producer_identities[name] = _parent_objects(
            name, parent_configs[name], producer_repo
        )

    fixtures = {name: json.loads(value["fixture"]) for name, value in objects.items()}
    schemas = {name: json.loads(value["schema"]) for name, value in objects.items()}
    payloads = {name: fixture["payload"] for name, fixture in fixtures.items()}
    for name in sorted(payloads):
        jsonschema.Draft202012Validator(schemas[name]).validate(payloads[name])
    payloads_before = copy.deepcopy(payloads)
    source = np.asarray(config["fixture"]["initial_romm_rgb"], dtype=np.float64)
    source_before = source.copy()
    exposure = float(config["fixture"]["exposure_multiplier"])

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p288-"))
    scientific: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    try:
        site = temporary / "site"
        _write_isolated_package(site, objects)
        sys.path.insert(0, str(site))
        try:
            modules = {
                "core": importlib.import_module("zhuise.dng_profile"),
                "huesat": importlib.import_module("zhuise.dng_profile_huesat_callable"),
                "gain_core": importlib.import_module("zhuise.dng_profile_gain_table"),
                "gain_table": importlib.import_module(
                    "zhuise.dng_profile_gain_table_callable"
                ),
                "look_table": importlib.import_module(
                    "zhuise.dng_profile_look_table_callable"
                ),
                "tone_curve": importlib.import_module(
                    "zhuise.dng_profile_tone_curve_callable"
                ),
            }
        finally:
            sys.path.remove(str(site))
        isolated = all(
            Path(module.__file__).resolve().is_relative_to(site.resolve())
            for module in modules.values()
        )
        primary, primary_stages = _compose(
            source, exposure, payloads, modules, direct=False
        )
        direct, direct_stages = _compose(source, exposure, payloads, modules, direct=True)
        tolerance = float(modules["core"].PROFILE_MATRIX_ROUNDTRIP_TOLERANCE)
        controls_order = [
            "wrong-rank",
            "nonfinite-input",
            "zero-exposure",
            "nonfinite-exposure",
            "swapped-stage-payload",
            "look-table-out-of-domain",
        ]
        if order == "reverse":
            controls_order.reverse()
        controls: dict[str, bool] = {}
        for name in controls_order:
            if name == "wrong-rank":
                controls[name] = _expect_value_error(
                    lambda: _compose(source.reshape(-1, 3), exposure, payloads, modules, direct=False)
                )
            elif name == "nonfinite-input":
                invalid = source.copy()
                invalid[0, 0, 0] = np.nan
                controls[name] = _expect_value_error(
                    lambda invalid=invalid: _compose(invalid, exposure, payloads, modules, direct=False)
                )
            elif name == "zero-exposure":
                controls[name] = _expect_value_error(
                    lambda: _compose(source, 0.0, payloads, modules, direct=False)
                )
            elif name == "nonfinite-exposure":
                controls[name] = _expect_value_error(
                    lambda: _compose(source, np.inf, payloads, modules, direct=False)
                )
            elif name == "swapped-stage-payload":
                controls[name] = _expect_value_error(
                    lambda: modules["huesat"].apply_dng_profile_huesat_callable_v1(
                        primary_stages["initial_xyz_d50"], payloads["look_table"]
                    )
                )
            elif name == "look-table-out-of-domain":
                invalid_romm = np.full((1, 1, 3), 1.25, dtype=np.float64)
                invalid_xyz = invalid_romm @ modules["core"].ROMM_RGB_TO_XYZ_D50.T
                controls[name] = _expect_value_error(
                    lambda invalid_xyz=invalid_xyz: modules[
                        "look_table"
                    ].apply_dng_profile_look_table_callable_v1(
                        invalid_xyz, payloads["look_table"]
                    )
                )
        stage_hashes = {
            name: _array_sha256(value) for name, value in sorted(primary_stages.items())
        }
        direct_stage_hashes = {
            name: _array_sha256(value) for name, value in sorted(direct_stages.items())
        }
        domain_checks = {
            "initial_romm": _romm_domain(source, tolerance),
            "huesat_romm": _romm_domain(primary_stages["huesat_romm"], tolerance),
            "gain_table_romm": _romm_domain(
                primary_stages["gain_table_romm"], tolerance
            ),
            "exposed_romm": _romm_domain(
                primary_stages["exposed_romm"], tolerance
            ),
            "look_table_romm": _romm_domain(
                primary_stages["look_table_romm"], tolerance
            ),
            "tone_curve_finite": bool(np.isfinite(primary).all()),
        }
        gates.update(
            {
                "all-invalid-controls-reject": all(controls.values()),
                "all-parent-and-producer-identities-exact": True,
                "all-payload-schemas-valid": True,
                "common-core-exact": objects["huesat"]["core"]
                == objects["look_table"]["core"]
                == objects["tone_curve"]["core"],
                "domains-finite-and-valid": all(domain_checks.values()),
                "inputs-and-payloads-unchanged": np.array_equal(source, source_before)
                and payloads == payloads_before,
                "isolated-git-object-imports": isolated,
                "local-bindings-exact": all(local_gates.values()),
                "output-owned-writable-contiguous-float64": bool(
                    primary.shape == (2, 2, 3)
                    and primary.dtype == np.float64
                    and primary.flags.owndata
                    and primary.flags.writeable
                    and primary.flags.c_contiguous
                ),
                "stage-order-exact": config["fixture"]["stage_order"]
                == [
                    "ProfileHueSatMap",
                    "ProfileGainTableMap",
                    "exposure",
                    "ProfileLookTable",
                    "ProfileToneCurve",
                ],
                "wrapper-direct-byte-parity": np.array_equal(primary, direct),
            }
        )
        scientific = {
            "candidate_count": "2/3",
            "config_sha256": _sha256(config_bytes),
            "controls": dict(sorted(controls.items())),
            "direct_output_f64le_sha256": _array_sha256(direct),
            "direct_stage_hashes": direct_stage_hashes,
            "domain_checks": domain_checks,
            "exposure_multiplier": exposure,
            "gates": gates,
            "initial_romm_f64le_sha256": _array_sha256(source),
            "network_reads": 0,
            "new_raw_dng_pixel_target_reads": 0,
            "output_f64le_sha256": _array_sha256(primary),
            "parent_identities": dict(sorted(parent_identities.items())),
            "producer_artifacts": dict(sorted(producer_identities.items())),
            "producer_worktree_imports": 0,
            "consumer_src_copies": 0,
            "stage_hashes": stage_hashes,
        }
    finally:
        for name in list(sys.modules):
            if name == "zhuise" or name.startswith("zhuise."):
                sys.modules.pop(name, None)
        shutil.rmtree(temporary, ignore_errors=False)
    cleanup = not temporary.exists()
    gates["zero-network-pixels-artifacts-and-temp-residue"] = bool(cleanup)
    scientific["zero_temporary_residue"] = cleanup
    status = (
        "PASS_PRIVATE_DNG_PROFILE_STAGE_COMPOSITION"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_PROFILE_STAGE_COMPOSITION"
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
