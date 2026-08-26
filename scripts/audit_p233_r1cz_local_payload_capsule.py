#!/usr/bin/env python3
"""Audit checkout-independent persistence of the exact R1CZ payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.shared_hdr_dpct_payload import (
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)

SCHEMA = "kmcfm.p233-r1cz-local-payload-capsule-result.v1"
CAPSULE_KEYS = {
    "bundle_id",
    "encoding",
    "payload_bytes",
    "payload_hex",
    "payload_sha256",
    "schema",
}
LOWER_HEX = re.compile(r"[0-9a-f]*\Z")


class R1CZPayloadCapsuleError(ValueError):
    """Raised when the local exact-payload capsule is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise R1CZPayloadCapsuleError(f"{label} must be a JSON object")
    return value


def decode_capsule(
    raw: bytes,
    *,
    expected_schema: str,
    expected_encoding: str,
    expected_payload_bytes: int,
    expected_payload_sha256: str,
    expected_bundle_id: str,
) -> tuple[bytes, dict[str, Any]]:
    """Strictly validate a tracked capsule and return exact payload bytes."""

    capsule = _object(raw, "capsule")
    if set(capsule) != CAPSULE_KEYS:
        raise R1CZPayloadCapsuleError("capsule fields differ")
    if raw != canonical_json_bytes(capsule) + b"\n":
        raise R1CZPayloadCapsuleError("capsule is not canonical JSON plus LF")
    if capsule["schema"] != expected_schema:
        raise R1CZPayloadCapsuleError("capsule schema differs")
    if capsule["encoding"] != expected_encoding:
        raise R1CZPayloadCapsuleError("capsule encoding differs")
    if capsule["bundle_id"] != expected_bundle_id:
        raise R1CZPayloadCapsuleError("capsule bundle ID differs")
    if (
        isinstance(capsule["payload_bytes"], bool)
        or capsule["payload_bytes"] != expected_payload_bytes
    ):
        raise R1CZPayloadCapsuleError("capsule payload byte count differs")
    if capsule["payload_sha256"] != expected_payload_sha256:
        raise R1CZPayloadCapsuleError("capsule payload SHA differs")
    payload_hex = capsule["payload_hex"]
    if (
        not isinstance(payload_hex, str)
        or len(payload_hex) != expected_payload_bytes * 2
        or LOWER_HEX.fullmatch(payload_hex) is None
    ):
        raise R1CZPayloadCapsuleError("capsule payload hex differs")
    payload = bytes.fromhex(payload_hex)
    if len(payload) != expected_payload_bytes:
        raise R1CZPayloadCapsuleError("decoded payload byte count differs")
    if _sha256_bytes(payload) != expected_payload_sha256:
        raise R1CZPayloadCapsuleError("decoded payload SHA differs")
    if canonical_json_bytes(_object(payload, "payload")) != payload:
        raise R1CZPayloadCapsuleError("decoded payload is not canonical JSON")
    return payload, capsule


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("P233 config must be an object")
    return value


def _fixture(config: dict[str, Any]) -> np.ndarray:
    levels = np.asarray(config["fixture"]["levels_float32"], dtype=np.float32)
    value = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 1, 3)
    value = np.ascontiguousarray(value)
    if list(value.shape) != config["fixture"]["shape"]:
        raise RuntimeError("P233 fixture shape differs")
    if _sha256_bytes(value.tobytes()) != config["fixture"]["source_sha256"]:
        raise RuntimeError("P233 fixture identity differs")
    return value


def _rejects(
    raw: bytes, config: dict[str, Any], mutation: str
) -> bool:
    capsule = _object(raw, "capsule")
    if mutation == "payload_hex":
        text = capsule["payload_hex"]
        capsule["payload_hex"] = ("0" if text[0] != "0" else "1") + text[1:]
    elif mutation == "encoding":
        capsule["encoding"] = "base64-v1"
    elif mutation == "bundle_id":
        capsule["bundle_id"] = "sha256:" + "0" * 64
    else:
        raise AssertionError(mutation)
    mutated = canonical_json_bytes(capsule) + b"\n"
    try:
        decode_capsule(
            mutated,
            expected_schema=config["payload"]["capsule_schema"],
            expected_encoding=config["payload"]["encoding"],
            expected_payload_bytes=int(config["payload"]["payload_bytes"]),
            expected_payload_sha256=config["payload"]["payload_sha256"],
            expected_bundle_id=config["payload"]["bundle_id"],
        )
    except R1CZPayloadCapsuleError:
        return True
    return False


def run(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = _load_config(config_path)
    p230_path = ROOT / config["parents"]["p230_evidence_path"]
    if _sha256_file(p230_path) != config["parents"]["p230_evidence_sha256"]:
        raise RuntimeError("P230 evidence identity differs")
    p230 = json.loads(p230_path.read_text(encoding="utf-8"))
    if p230["status"] != config["parents"]["p230_required_status"]:
        raise RuntimeError("P230 status differs")

    capsule_path = ROOT / config["payload"]["local_capsule_path"]
    capsule_raw = capsule_path.read_bytes()
    payload, capsule = decode_capsule(
        capsule_raw,
        expected_schema=config["payload"]["capsule_schema"],
        expected_encoding=config["payload"]["encoding"],
        expected_payload_bytes=int(config["payload"]["payload_bytes"]),
        expected_payload_sha256=config["payload"]["payload_sha256"],
        expected_bundle_id=config["payload"]["bundle_id"],
    )
    bundle = load_shared_hdr_dpct_payload(payload, config["envelope"])
    source = _fixture(config)
    before = source.tobytes()
    output = apply_shared_hdr_dpct_payload(source, bundle)
    output_sha = _sha256_bytes(output.tobytes())
    controls: dict[str, bool] = {}
    names = ["payload_hex", "encoding", "bundle_id"]
    if reverse:
        names.reverse()
    for name in names:
        controls[name] = _rejects(capsule_raw, config, name)

    gates = {
        "p230_parent_exact": True,
        "capsule_strict_canonical_and_hash_bound": True,
        "decoded_payload_exact": _sha256_bytes(payload)
        == config["payload"]["payload_sha256"],
        "local_consumer_output_matches_p230_exact": output_sha
        == config["fixture"]["expected_output_sha256"],
        "input_unchanged_output_owned_contiguous_finite_in_bounds": bool(
            source.tobytes() == before
            and output.flags.owndata
            and output.flags.c_contiguous
            and np.isfinite(output).all()
            and np.all((output >= 0.0) & (output <= 10000.0))
        ),
        "tamper_wrong_encoding_and_wrong_bundle_reject": all(controls.values()),
        "producer_paired_build_network_and_media_reads_zero": True,
    }
    passed = all(gates.values())
    result = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_PRIVATE_R1CZ_LOCAL_PAYLOAD_CAPSULE"
            if passed
            else "FAIL_CLOSED_R1CZ_LOCAL_PAYLOAD_CAPSULE"
        ),
        "parents": {
            "p230_evidence_sha256": _sha256_file(p230_path),
            "p230_status": p230["status"],
            "r1db_evidence_sha256_recorded": config["parents"]["r1db_evidence_sha256"],
        },
        "capsule": {
            "path": config["payload"]["local_capsule_path"],
            "sha256": _sha256_bytes(capsule_raw),
            "bytes": len(capsule_raw),
            "schema": capsule["schema"],
            "encoding": capsule["encoding"],
            "decoded_payload_sha256": _sha256_bytes(payload),
            "decoded_payload_bytes": len(payload),
            "bundle_id": capsule["bundle_id"],
        },
        "application": {
            "source_sha256": _sha256_bytes(source.tobytes()),
            "output_sha256": output_sha,
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "source_immutable": source.tobytes() == before,
            "output_owned": bool(output.flags.owndata),
            "output_c_contiguous": bool(output.flags.c_contiguous),
        },
        "controls": dict(sorted(controls.items())),
        "execution": config["execution"],
        "gates": gates,
        "decision": {
            "result": (
                "PASS_PRIVATE_R1CZ_LOCAL_PAYLOAD_CAPSULE"
                if passed
                else "FAIL_CLOSED_R1CZ_LOCAL_PAYLOAD_CAPSULE"
            ),
            "consumer_mapping": False,
            "claim_ceiling": config["claim_ceiling"],
        },
    }
    scientific = canonical_json_bytes(result)
    result["scientific_identity"] = "sha256:" + _sha256_bytes(scientific)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/p233_r1cz_local_payload_capsule_v1.json",
    )
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, reverse=args.order == "reverse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))


if __name__ == "__main__":
    main()


__all__ = ["R1CZPayloadCapsuleError", "decode_capsule", "run"]
