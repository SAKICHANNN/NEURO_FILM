"""Emit deterministic U6.P6ZI scanner-chain serialization evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile
from src.film_physics.scanner_glare import compile_scanner_glare_kernel

CONTRACT = ROOT / "configs" / "u6_p6zi_scanner_chain_profile_serialization_v1.json"
P6ZG_CONTRACT = ROOT / "configs" / "u6_p6zg_scanner_glare_typed_chain_v1.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rejects(payload: dict[str, Any]) -> bool:
    try:
        ScannerChainProfile.from_dict(payload)
    except (TypeError, ValueError):
        return True
    return False


def evaluate() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p6zg = load_contract(P6ZG_CONTRACT)
    profile = ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, p6zg),
        glare_profile=glare_profile(),
    )
    raw = profile.canonical_bytes()
    reconstructed = ScannerChainProfile.from_json_bytes(raw)
    kernel = compile_scanner_glare_kernel(
        profile.glare_profile, kernel_size=profile.glare_kernel_size
    )
    reconstructed_kernel = compile_scanner_glare_kernel(
        reconstructed.glare_profile,
        kernel_size=reconstructed.glare_kernel_size,
    )

    parent_hashes = {
        prefix: _sha256(ROOT / contract["parents"][f"{prefix}_path"])
        for prefix in ("p6zh_evidence", "p6zh_implementation")
    }
    parent_hashes_exact = all(
        parent_hashes[prefix] == contract["parents"][f"{prefix}_sha256"]
        for prefix in parent_hashes
    )

    top_unknown = profile.to_dict()
    top_unknown["unknown"] = 1
    nested_unknown = profile.to_dict()
    nested_unknown["scanner_profile"]["unknown"] = 1
    domain_drift = profile.to_dict()
    domain_drift["input_domain"] = "display-linear-light"
    stage_drift = profile.to_dict()
    stage_drift["stage_order"] = [
        "spectral",
        "dmax",
        "p6za-multiscale-glare",
        "mtf",
        "noise",
    ]
    claim_drift = profile.to_dict()
    claim_drift["scanner_calibrated"] = True
    integer_coercion = profile.to_dict()
    integer_coercion["glare_profile"]["components"][0]["weight"] = 1

    duplicate_raw = raw[:-1] + (
        b',"schema":"neuro_film.generic_scanner_chain_profile.v1"}'
    )
    try:
        ScannerChainProfile.from_json_bytes(duplicate_raw)
        duplicate_rejected = False
    except ValueError:
        duplicate_rejected = True
    nonfinite = profile.to_dict()
    nonfinite["pixel_pitch_um"] = float("nan")
    try:
        ScannerChainProfile.from_json_bytes(
            json.dumps(nonfinite, separators=(",", ":")).encode("ascii")
        )
        nonfinite_rejected = False
    except ValueError:
        nonfinite_rejected = True

    decisions = {
        "canonical_roundtrip_exact": reconstructed.canonical_bytes() == raw,
        "identity_roundtrip_exact": reconstructed.profile_sha256
        == profile.profile_sha256,
        "strict_top_level_and_nested_fields": _rejects(top_unknown)
        and _rejects(nested_unknown)
        and _rejects(integer_coercion),
        "duplicate_json_keys_rejected": duplicate_rejected,
        "nonfinite_json_constants_rejected": nonfinite_rejected,
        "domain_stage_claim_and_execution_drift_rejected": _rejects(domain_drift)
        and _rejects(stage_drift)
        and _rejects(claim_drift),
        "reconstructed_profile_and_kernel_exact": reconstructed == profile
        and np.array_equal(reconstructed_kernel, kernel),
        "p6zh_parent_hashes_exact": parent_hashes_exact,
    }
    core = {
        "schema": "neuro_film.u6_p6zi_scanner_chain_profile_report.v1",
        "node": "U6.P6ZI",
        "contract_sha256": _sha256(CONTRACT),
        "automatic_pass": all(decisions.values()),
        "decisions": decisions,
        "profile_sha256": profile.profile_sha256,
        "canonical_profile_bytes": len(raw),
        "kernel_sha256": hashlib.sha256(kernel.tobytes(order="C")).hexdigest(),
        "parent_hashes": parent_hashes,
        "claim_ceiling": contract["claim_ceiling"],
        "branch": contract["branch_rule"][
            "pass" if all(decisions.values()) else "fail"
        ],
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
