"""Compile and exhaustively verify the P6AL scanner log-OECF profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.scanner_oecf import LogScannerOecfProfile

CONTRACT_SCHEMA = "neuro_film.u6_p6al_scanner_log_oecf_profile_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6al_scanner_log_oecf_profile_report.v1"


class ScannerOecfProfileError(RuntimeError):
    """Raised when the P6AL contract or profile does not satisfy its gates."""


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ScannerOecfProfileError("unsupported P6AL contract")
    parent = payload["parent"]
    parent_path = root / parent["evidence_path"]
    if hash_file(parent_path) != parent["evidence_sha256"]:
        raise ScannerOecfProfileError("P6AK evidence binding mismatch")
    return payload


def compile_profile(config: dict[str, Any]) -> LogScannerOecfProfile:
    profile = dict(config["profile"])
    profile["parent_evidence_sha256"] = config["parent"]["evidence_sha256"]
    return LogScannerOecfProfile.from_dict(
        {
            "schema": "neuro_film.log_scanner_oecf_profile.v1",
            **profile,
        }
    )


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    profile = compile_profile(config)
    codes = np.arange(65536, dtype=np.float64)
    valid = codes[codes > profile.black_code]
    density = profile.code_to_density(valid)
    code_roundtrip = profile.density_to_code(
        np.clip(
            density,
            profile.minimum_calibrated_density,
            profile.maximum_calibrated_density,
        )
    )
    calibrated_mask = (density >= profile.minimum_calibrated_density) & (
        density <= profile.maximum_calibrated_density
    )
    code_error = float(
        np.max(np.abs(code_roundtrip[calibrated_mask] - valid[calibrated_mask]))
    )
    density_grid = np.linspace(
        profile.minimum_calibrated_density,
        profile.maximum_calibrated_density,
        4097,
        dtype=np.float64,
    )
    density_roundtrip = profile.code_to_density(
        profile.density_to_code(density_grid)
    )
    density_error = float(np.max(np.abs(density_roundtrip - density_grid)))
    serialized = profile.canonical_bytes()
    restored = LogScannerOecfProfile.from_dict(json.loads(serialized))
    gates = config["gates"]
    checks = {
        "all_65536_integer_codes_preflighted": len(codes) == 65536,
        "valid_code_domain_strictly_decreasing_density": bool(
            np.all(np.diff(density) < 0.0)
        ),
        "calibrated_density_inverse_roundtrip": density_error
        <= float(gates["calibrated_density_inverse_roundtrip_max_abs_error"]),
        "valid_code_roundtrip": code_error
        <= float(gates["valid_code_roundtrip_max_abs_error"]),
        "serialization_byte_exact": restored.canonical_bytes() == serialized,
        "identity_byte_exact": restored.profile_sha256 == profile.profile_sha256,
        "claims_fail_closed": not restored.independent_confirmation
        and not restored.production_eligible,
    }
    if not all(checks.values()):
        raise ScannerOecfProfileError("P6AL scanner OECF verification failed")
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": "pass-exact-workflow-typed-log-oecf-profile",
        "decision": "retain_research_only_scanner_log_oecf_profile",
        "parent_evidence_sha256": config["parent"]["evidence_sha256"],
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "verification": {
            "integer_codes_preflighted": int(codes.size),
            "valid_integer_codes": int(valid.size),
            "calibrated_integer_codes": int(np.count_nonzero(calibrated_mask)),
            "valid_density_minimum": float(np.min(density)),
            "valid_density_maximum": float(np.max(density)),
            "calibrated_code_roundtrip_max_abs_error": code_error,
            "density_grid_points": int(density_grid.size),
            "calibrated_density_roundtrip_max_abs_error": density_error,
            "checks": checks,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ScannerOecfProfileError",
    "canonical_json",
    "compile_profile",
    "evaluate",
    "hash_file",
    "load_contract",
]
