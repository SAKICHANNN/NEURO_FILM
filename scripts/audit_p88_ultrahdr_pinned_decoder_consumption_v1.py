#!/usr/bin/env python3
"""Formal P88 pinned libultrahdr executable to P87 consumption audit."""

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

REPORT_SCHEMA = "neuro-film.p88-ultrahdr-pinned-decoder-result.v1"


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


def build_decoder_command(
    executable: Path,
    input_name: str,
    output_name: str,
) -> list[str]:
    """Return the frozen official-app HDR RGBA16F decode invocation."""

    return [
        str(executable),
        "-m",
        "1",
        "-j",
        input_name,
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        output_name,
    ]


def _run_decoder(
    executable: Path,
    input_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_decoder_command(executable, input_path.name, output_path.name),
        cwd=input_path.parent,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _validate_bindings(
    config: dict[str, Any],
    decoder_app: Path,
    producer_evidence: Path,
) -> dict[str, str]:
    decoder_sha = _sha256_file(decoder_app)
    if decoder_sha != config["official"]["decoder_executable_sha256"]:
        raise ValueError("decoder executable SHA-256 mismatch")
    producer_sha = _sha256_file(producer_evidence)
    if producer_sha != config["bindings"]["producer_r1bl_evidence_sha256"]:
        raise ValueError("producer R1BL evidence SHA-256 mismatch")
    producer = _load_object(producer_evidence)
    if producer["status"] != "PASS_PRIVATE_PUBLIC_C_API_EMBEDDING":
        raise ValueError("producer R1BL status mismatch")
    if (
        producer["official_and_build_identity"]["official_commit"]
        != config["official"]["commit"]
    ):
        raise ValueError("producer official commit mismatch")
    if (
        producer["official_and_build_identity"]["library_version"]
        != ULTRAHDR_DECODER_VERSION
    ):
        raise ValueError("producer decoder version mismatch")
    p87_path = (
        ROOT / "docs/evidence/P87_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_RESULT.json"
    )
    p87_sha = _sha256_file(p87_path)
    if p87_sha != config["bindings"]["p87_evidence_sha256"]:
        raise ValueError("P87 evidence SHA-256 mismatch")
    return {
        "decoder_executable_sha256": decoder_sha,
        "p87_evidence_sha256": p87_sha,
        "producer_r1bl_evidence_sha256": producer_sha,
        "producer_r1bl_stable_identity": producer["execution"]["stable_identity"],
    }


def _production_rejection(path: Path) -> str:
    try:
        load_working_image(path)
    except ValueError as error:
        message = str(error)
        if "HDR/gain-map reconstruction is not implemented" not in message:
            raise AssertionError(
                "production loader failed for another reason"
            ) from error
        return message
    raise AssertionError("production loader accepted pinned gain-map fixture")


def run(
    config_path: Path,
    decoder_app: Path,
    producer_evidence: Path,
    *,
    reverse: bool,
) -> dict[str, Any]:
    config = _load_object(config_path)
    bindings = _validate_bindings(config, decoder_app, producer_evidence)
    fixture_root = ROOT / "tests/fixtures/u1_5c_libultrahdr"
    source_manifest = ROOT / config["bindings"]["fixture_source_manifest"]
    source = _load_object(source_manifest)
    if source["license"] != "CC-BY-4.0":
        raise ValueError("fixture license fact mismatch")
    rows = list(config["fixtures"])
    if reverse:
        rows.reverse()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p88_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            fixture = fixture_root / row["name"]
            fixture_bytes = fixture.read_bytes()
            if _sha256_bytes(fixture_bytes) != row["sha256"]:
                raise ValueError(f"fixture SHA-256 mismatch: {row['name']}")
            local_input = scratch / row["name"]
            local_output = scratch / f"{Path(row['name']).stem}.rgba16f"
            local_input.write_bytes(fixture_bytes)
            completed = _run_decoder(decoder_app, local_input, local_output)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"decoder failed for {row['name']}: {completed.stderr[-500:]}"
                )
            decoded = local_output.read_bytes()
            expected_length = int(row["width"]) * int(row["height"]) * 4 * 2
            prepared = prepare_ultrahdr_match_view_v1(
                source_asset=fixture_bytes,
                expected_source_sha256=row["sha256"],
                decoded_rgba16f=decoded,
                width=int(row["width"]),
                height=int(row["height"]),
                decoder_version=ULTRAHDR_DECODER_VERSION,
                producer_profile_id=ULTRAHDR_EXTERNAL_PROFILE_ID,
            )
            records.append(
                {
                    "decoded_byte_length": len(decoded),
                    "decoded_length_exact": len(decoded) == expected_length,
                    "decoded_payload_sha256": prepared.decoded_payload_sha256,
                    "fixture_name": row["name"],
                    "fixture_sha256": row["sha256"],
                    "ingress_id": prepared.ingress_id,
                    "maximum_nits": float(prepared.pixels.max()),
                    "minimum_nits": float(prepared.pixels.min()),
                    "pixel_sha256": prepared.descriptor.pixel_sha256,
                    "profile_id": prepared.descriptor.profile_id,
                    "production_rejection": _production_rejection(fixture),
                    "reference_white_nits": prepared.descriptor.reference_white_nits,
                    "source_unchanged": fixture.read_bytes() == fixture_bytes,
                    "view_id": prepared.descriptor.view_id,
                }
            )

        truncated = scratch / "truncated.jpg"
        truncated_output = scratch / "truncated.rgba16f"
        first_bytes = (fixture_root / config["fixtures"][0]["name"]).read_bytes()
        truncated.write_bytes(first_bytes[:64])
        invalid = _run_decoder(decoder_app, truncated, truncated_output)
        truncated_rejected = invalid.returncode != 0 and not truncated_output.exists()

    records.sort(key=lambda value: value["fixture_name"])
    gates = {
        "all_decoded_lengths_exact": all(
            row["decoded_length_exact"] for row in records
        ),
        "all_match_views_exact": all(
            row["profile_id"]
            == "neuro-film.display-absolute-linear-rec2020-d65-cdm2.v1"
            and row["reference_white_nits"] == 203.0
            for row in records
        ),
        "all_sources_unchanged": all(row["source_unchanged"] for row in records),
        "bindings_exact": len(bindings) == 4,
        "production_loader_still_rejects": all(
            "HDR/gain-map reconstruction is not implemented"
            in row["production_rejection"]
            for row in records
        ),
        "truncated_input_atomic": truncated_rejected,
    }
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha256_file(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "records": records,
        "schema": REPORT_SCHEMA,
        "status": (
            "PASS_PRIVATE_PINNED_ULTRAHDR_DECODER_TO_P87"
            if all(gates.values())
            else "FAIL_CLOSED_PINNED_ULTRAHDR_DECODER_TO_P87"
        ),
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p88_ultrahdr_pinned_decoder_consumption_v1.json"),
    )
    parser.add_argument("--decoder-app", type=Path, required=True)
    parser.add_argument("--producer-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    decoder = args.decoder_app.resolve()
    producer = args.producer_evidence.resolve()
    report = run(config, decoder, producer, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": _sha256_bytes(payload),
                "stable_evidence_id": report["stable_evidence_id"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
