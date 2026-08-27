#!/usr/bin/env python3
"""Audit official libavif gain-map AVIF fixtures through pinned libultrahdr/P87."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    prepare_ultrahdr_match_view_v1,
)
from src.preprocess import load_working_image
from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p278-libavif-gainmap-compatibility-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def build_decoder_command(executable: Path, input_name: str, output_name: str) -> list[str]:
    return [
        str(executable), "-m", "1", "-j", input_name,
        "-o", "0", "-O", "4", "-z", output_name,
    ]


def _run_decoder(executable: Path, input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_decoder_command(executable, input_path.name, output_path.name),
        cwd=input_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [line.strip() for line in (completed.stdout + "\n" + completed.stderr).splitlines() if line.strip()]
    return lines[-1][-300:] if lines else ""


def _production_rejection(path: Path) -> dict[str, str | bool]:
    try:
        load_working_image(path)
    except (OSError, RuntimeError, ValueError) as error:
        return {"accepted": False, "error_type": type(error).__name__, "message": str(error)}
    return {"accepted": True, "error_type": "", "message": ""}


def _validate_bindings(config: dict[str, Any], decoder_app: Path, producer_evidence: Path) -> dict[str, str]:
    decoder_sha = _sha256_file(decoder_app)
    if decoder_sha != config["decoder"]["executable_sha256"]:
        raise ValueError("decoder executable SHA-256 mismatch")
    producer_sha = _sha256_file(producer_evidence)
    if producer_sha != config["bindings"]["producer_r1bl_evidence_sha256"]:
        raise ValueError("producer R1BL evidence SHA-256 mismatch")
    producer = _load_object(producer_evidence)
    if producer["status"] != "PASS_PRIVATE_PUBLIC_C_API_EMBEDDING":
        raise ValueError("producer R1BL status mismatch")
    p87_path = ROOT / "docs/evidence/P87_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_RESULT.json"
    p87_sha = _sha256_file(p87_path)
    if p87_sha != config["bindings"]["p87_evidence_sha256"]:
        raise ValueError("P87 evidence SHA-256 mismatch")
    return {
        "decoder_executable_sha256": decoder_sha,
        "p87_evidence_sha256": p87_sha,
        "producer_r1bl_evidence_sha256": producer_sha,
        "producer_r1bl_stable_identity": producer["execution"]["stable_identity"],
    }


def run(config_path: Path, decoder_app: Path, producer_evidence: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    bindings = _validate_bindings(config, decoder_app, producer_evidence)
    source_root = ROOT / config["source"]["root"]
    manifest_path = ROOT / config["source"]["manifest"]
    if _sha256_file(manifest_path) != config["source"]["manifest_sha256"]:
        raise ValueError("source manifest SHA-256 mismatch")
    manifest = _load_object(manifest_path)
    manifest_rows = {row["name"]: row for row in manifest["files"]}
    expected_names = {"LICENSE", "README.md"} | {
        row["name"] for row in config["valid_fixtures"] + config["invalid_fixtures"]
    }
    if set(manifest_rows) != expected_names:
        raise ValueError("source retained-member set mismatch")

    roles = [("valid", row) for row in config["valid_fixtures"]] + [
        ("invalid", row) for row in config["invalid_fixtures"]
    ]
    if reverse:
        roles.reverse()
    records: list[dict[str, Any]] = []
    source_snapshots: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="p278_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for role, row in roles:
            source_path = source_root / row["name"]
            source_bytes = source_path.read_bytes()
            source_snapshots[row["name"]] = source_bytes
            if _sha256_bytes(source_bytes) != row["sha256"]:
                raise ValueError(f"source SHA-256 mismatch: {row['name']}")
            if manifest_rows[row["name"]]["sha256"] != row["sha256"]:
                raise ValueError(f"manifest/config mismatch: {row['name']}")
            local_input = scratch / row["name"]
            local_output = scratch / f"{row['name']}.rgba16f"
            local_input.write_bytes(source_bytes)
            completed = _run_decoder(decoder_app, local_input, local_output)
            output_exists = local_output.is_file()
            record: dict[str, Any] = {
                "decoder_diagnostic": _diagnostic(completed),
                "decoder_returncode": completed.returncode,
                "name": row["name"],
                "output_exists": output_exists,
                "role": role,
                "source_sha256": row["sha256"],
            }
            if role == "valid":
                expected_length = int(row["width"]) * int(row["height"]) * 4 * 2
                decoded = local_output.read_bytes() if output_exists else b""
                decode_success = completed.returncode == 0 and len(decoded) == expected_length
                record.update({
                    "decode_success": decode_success,
                    "decoded_byte_length": len(decoded),
                    "expected_byte_length": expected_length,
                    "production_rejection": _production_rejection(source_path),
                })
                if decode_success:
                    prepared = prepare_ultrahdr_match_view_v1(
                        source_asset=source_bytes,
                        expected_source_sha256=row["sha256"],
                        decoded_rgba16f=decoded,
                        width=int(row["width"]),
                        height=int(row["height"]),
                        decoder_version=ULTRAHDR_DECODER_VERSION,
                        producer_profile_id=ULTRAHDR_EXTERNAL_PROFILE_ID,
                    )
                    record.update({
                        "decoded_payload_sha256": prepared.decoded_payload_sha256,
                        "ingress_id": prepared.ingress_id,
                        "maximum_nits": float(prepared.pixels.max()),
                        "minimum_nits": float(prepared.pixels.min()),
                        "pixel_sha256": prepared.descriptor.pixel_sha256,
                        "profile_id": prepared.descriptor.profile_id,
                    })
            else:
                record.update({
                    "atomic_rejection": completed.returncode != 0 and not output_exists,
                    "reason": row["reason"],
                })
            records.append(record)

        first = (source_root / config["valid_fixtures"][0]["name"]).read_bytes()
        truncated = scratch / "truncated.avif"
        truncated_output = scratch / "truncated.rgba16f"
        truncated.write_bytes(first[:64])
        invalid = _run_decoder(decoder_app, truncated, truncated_output)
        truncated_rejected = invalid.returncode != 0 and not truncated_output.exists()

    records.sort(key=lambda value: value["name"])
    sources_unchanged = all((source_root / name).read_bytes() == value for name, value in source_snapshots.items())
    valid_records = [row for row in records if row["role"] == "valid"]
    invalid_records = [row for row in records if row["role"] == "invalid"]
    gates = {
        "all_invalid_atomic_rejection": len(invalid_records) == config["gates"]["required_invalid_count"] and all(row["atomic_rejection"] for row in invalid_records),
        "all_valid_decode": len(valid_records) == config["gates"]["required_valid_count"] and all(row["decode_success"] for row in valid_records),
        "production_loader_fail_closed": all(not row["production_rejection"]["accepted"] for row in valid_records),
        "source_immutable": sources_unchanged,
        "truncation_rejected_atomically": truncated_rejected,
    }
    passed = all(gates.values())
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "gates": gates,
        "records": records,
        "source_commit": config["source"]["commit"],
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "status": "PASS_PRIVATE_LIBAVIF_GAINMAP_AVIF_DECODER_COMPATIBILITY" if passed else "FAIL_CLOSED_LIBAVIF_GAINMAP_AVIF_DECODER_COMPATIBILITY",
    }
    scientific_bytes = canonical_json_bytes(scientific)
    return {
        "schema": REPORT_SCHEMA,
        "status": scientific["status"],
        "scientific": scientific,
        "execution": {
            "scientific_sha256": _sha256_bytes(scientific_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decoder-app", type=Path, required=True)
    parser.add_argument("--producer-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.decoder_app, args.producer_evidence, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
