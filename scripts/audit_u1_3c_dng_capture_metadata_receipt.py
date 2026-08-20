#!/usr/bin/env python3
"""Formal U1.3C metadata-only DNG receipt audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_metadata import (
    build_dng_capture_metadata_receipt,
    canonical_json_bytes,
)

REPORT_SCHEMA = "neuro_film.u1_3c_dng_capture_metadata_receipt_result.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _manifest_row(manifest: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("candidates", []) if row.get("id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {source_id}, found {len(matches)}")
    return matches[0]


def _binding_facts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bindings = config["bindings"]
    checked: dict[str, Any] = {}
    for key in ("contract", "implementation", "runner"):
        path = ROOT / bindings[f"{key}_path"]
        actual = _sha256_file(path)
        expected = bindings[f"{key}_sha256"]
        if actual != expected:
            raise ValueError(f"{key} SHA-256 mismatch: {actual} != {expected}")
        checked[f"{key}_path"] = bindings[f"{key}_path"]
        checked[f"{key}_sha256"] = actual

    manifests: list[dict[str, Any]] = []
    checked_manifests: list[dict[str, str]] = []
    for binding in bindings["source_manifests"]:
        path = ROOT / binding["path"]
        actual = _sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"source manifest SHA-256 mismatch: {binding['path']}")
        manifests.append(_load_json(path))
        checked_manifests.append({"path": binding["path"], "sha256": actual})
    checked["source_manifests"] = checked_manifests
    checked["implementation_commit"] = bindings["implementation_commit"]
    return manifests, checked


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    manifests, bindings = _binding_facts(config)
    rows = list(config["rows"])
    if reverse:
        rows.reverse()

    receipts: list[dict[str, Any]] = []
    manifest_matches = 0
    source_matches = 0
    digest_matches = 0
    for row in rows:
        source = ROOT / row["logical_path"]
        manifest = manifests[int(row["source_manifest_index"])]
        source_fact = _manifest_row(manifest, row["source_id"])
        if (
            source_fact.get("path") == row["logical_path"]
            and source_fact.get("sha256") == row["source_sha256"]
            and manifest.get("source", {}).get("declared_license")
            == "CC0/Public Domain per each exact repository row"
        ):
            manifest_matches += 1

        receipt = build_dng_capture_metadata_receipt(
            source,
            source_id=row["source_id"],
            logical_path=row["logical_path"],
        )
        if (
            receipt["source_sha256"] == row["source_sha256"]
            and receipt["source_bytes"] == row["source_bytes"]
            and receipt["raw_kind"] == row["expected_raw_kind"]
        ):
            source_matches += 1
        body = {key: value for key, value in receipt.items() if key != "receipt_body_sha256"}
        if hashlib.sha256(canonical_json_bytes(body)).hexdigest() == receipt["receipt_body_sha256"]:
            digest_matches += 1
        receipts.append(receipt)

    receipts.sort(key=lambda value: value["source_id"])
    count = len(receipts)
    metrics = {
        "cfa_count": sum(row["raw_kind"] == "cfa" for row in receipts),
        "linear_raw_count": sum(row["raw_kind"] == "linear_raw" for row in receipts),
        "manifest_binding_rate": manifest_matches / count if count else 0.0,
        "noise_profile_rate": sum(row["noise_profile_present"] for row in receipts) / count
        if count
        else 0.0,
        "raster_decode_calls": sum(row["raster_decode_calls"] for row in receipts),
        "receipt_digest_rate": digest_matches / count if count else 0.0,
        "row_count": count,
        "source_hash_match_rate": source_matches / count if count else 0.0,
    }
    gate_config = config["gates"]
    gates = {
        "cfa_count": metrics["cfa_count"] == gate_config["required_cfa_count"],
        "linear_raw_count": metrics["linear_raw_count"]
        == gate_config["required_linear_raw_count"],
        "manifest_binding": metrics["manifest_binding_rate"]
        == gate_config["required_manifest_binding_rate"],
        "noise_profile": (not gate_config["require_all_noise_profiles"])
        or metrics["noise_profile_rate"] == 1.0,
        "raster_decode": metrics["raster_decode_calls"]
        <= gate_config["maximum_raster_decode_calls"],
        "receipt_digest": metrics["receipt_digest_rate"]
        == gate_config["required_receipt_digest_rate"],
        "row_count": metrics["row_count"] == gate_config["required_row_count"],
        "source_hash": metrics["source_hash_match_rate"]
        == gate_config["required_source_hash_match_rate"],
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": metrics,
        "node": config["node"],
        "receipts": receipts,
        "schema": REPORT_SCHEMA,
        "source_rights_ceiling": config["source_rights_ceiling"],
        "status": "PASS_PRIVATE_DNG_CAPTURE_METADATA_RECEIPT"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_CAPTURE_METADATA_RECEIPT",
    }
    scientific["stable_evidence_id"] = hashlib.sha256(canonical_json_bytes(scientific)).hexdigest()
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u1_3c_dng_capture_metadata_receipt_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config_path, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
