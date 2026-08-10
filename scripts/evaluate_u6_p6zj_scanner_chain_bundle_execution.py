"""Emit deterministic U6.P6ZJ bundle-driven execution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.eval.scanner_chain_bundle_execution as execution
from src.eval.scanner_glare_tiled_downstream import (
    apply_typed_scanner_glare_chain_tiled_downstream,
)
from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile

CONTRACT = ROOT / "configs" / "u6_p6zj_scanner_chain_bundle_execution_v1.json"
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


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = (
        array.dtype.str.encode("ascii")
        + b"|"
        + json.dumps(array.shape, separators=(",", ":")).encode("ascii")
        + b"|"
        + array.tobytes(order="C")
    )
    return hashlib.sha256(identity).hexdigest()


def _rejected_before_execution(call: Callable[[], np.ndarray]) -> bool:
    original = execution.apply_typed_scanner_glare_chain_tiled_downstream
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("pixel executor was called")

    execution.apply_typed_scanner_glare_chain_tiled_downstream = forbidden
    try:
        try:
            call()
        except (TypeError, ValueError):
            return calls == 0
        return False
    finally:
        execution.apply_typed_scanner_glare_chain_tiled_downstream = original


def evaluate() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p6zg = load_contract(P6ZG_CONTRACT)
    profile = ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, p6zg), glare_profile=glare_profile()
    )
    raw = profile.canonical_bytes()
    source = np.random.default_rng(6206106).uniform(0.03, 0.97, (321, 513, 3))
    direct = apply_typed_scanner_glare_chain_tiled_downstream(
        source,
        profile.scanner_profile,
        pixel_pitch_um=profile.pixel_pitch_um,
        glare_row_chunk=profile.glare_row_chunk,
        downstream_tile_rows=profile.downstream_tile_rows,
    )
    first = execution.apply_scanner_chain_from_bundle(
        source, raw, expected_profile_sha256=profile.profile_sha256
    )
    second = execution.apply_scanner_chain_from_bundle(
        source, raw, expected_profile_sha256=profile.profile_sha256
    )

    parent_hashes = {
        prefix: _sha256(ROOT / contract["parents"][f"{prefix}_path"])
        for prefix in ("p6zi_evidence", "profile_serializer", "p6zh_executor")
    }
    parent_hashes_exact = all(
        parent_hashes[prefix] == contract["parents"][f"{prefix}_sha256"]
        for prefix in parent_hashes
    )

    wrong_identity = _rejected_before_execution(
        lambda: execution.apply_scanner_chain_from_bundle(
            source, raw, expected_profile_sha256="0" * 64
        )
    )
    noncanonical = _rejected_before_execution(
        lambda: execution.apply_scanner_chain_from_bundle(
            source, raw + b"\n", expected_profile_sha256=profile.profile_sha256
        )
    )
    tampered = raw.replace(b'"downstream_tile_rows":512', b'"downstream_tile_rows":256')
    tamper_rejected = _rejected_before_execution(
        lambda: execution.apply_scanner_chain_from_bundle(
            source, tampered, expected_profile_sha256=profile.profile_sha256
        )
    )
    invalid_inputs = (
        source.astype(np.float32),
        source[..., 0],
        np.zeros_like(source),
        np.full_like(source, np.nan),
    )
    inputs_rejected = all(
        _rejected_before_execution(
            lambda invalid=invalid: execution.apply_scanner_chain_from_bundle(
                invalid, raw, expected_profile_sha256=profile.profile_sha256
            )
        )
        for invalid in invalid_inputs
    )

    decisions = {
        "parent_hashes_exact": parent_hashes_exact,
        "bundle_identity_exact": profile.profile_sha256
        == contract["parents"]["profile_sha256"],
        "canonical_bytes_exact": ScannerChainProfile.from_json_bytes(
            raw
        ).canonical_bytes()
        == raw,
        "direct_vs_bundle_output_bit_exact": np.array_equal(first, direct),
        "repeat_output_exact": np.array_equal(second, first),
        "wrong_identity_rejected_before_execution": wrong_identity,
        "noncanonical_and_tampered_bundle_rejected_before_execution": noncanonical
        and tamper_rejected,
        "dtype_shape_domain_rejected_before_execution": inputs_rejected,
    }
    core = {
        "schema": "neuro_film.u6_p6zj_scanner_chain_bundle_execution_report.v1",
        "node": "U6.P6ZJ",
        "contract_sha256": _sha256(CONTRACT),
        "automatic_pass": all(decisions.values()),
        "decisions": decisions,
        "profile_sha256": profile.profile_sha256,
        "source_sha256": _hash_array(source),
        "output_sha256": _hash_array(first),
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
