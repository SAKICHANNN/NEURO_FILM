#!/usr/bin/env python3
"""Formal P91 canonical PQ writer audit over the exact P90 rows."""

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

import scripts.audit_p90_dng_aces2_hdr_pq_png_v1 as p90
from src.preprocess.aces2_canonical_pq_png import (
    publish_working_image_aces2_canonical_hdr_pq_png_v1,
)
from src.preprocess.dng_metadata import canonical_json_bytes

SCHEMA = "neuro-film.p91-canonical-pq-partition-invariance-result.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load(config_path)
    evidence_path = ROOT / "docs/evidence/P90_DNG_ACES2_HDR_PQ_PNG_RESULT.json"
    if _sha(evidence_path) != config["p90_evidence_sha256"]:
        raise ValueError("P90 evidence identity mismatch")
    p90.publish_working_image_aces2_hdr_pq_png_v1 = (
        publish_working_image_aces2_canonical_hdr_pq_png_v1
    )
    component = p90.run(
        ROOT / "configs/p90_dng_aces2_hdr_pq_png_v1.json",
        reverse=reverse,
    )
    gates = dict(component["gates"])
    gates["p90_failure_preserved"] = _load(evidence_path)["status"] == (
        "FAIL_CLOSED_PARTITION_BYTE_REPLAY"
    )
    report: dict[str, Any] = {
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "records": component["records"],
        "schema": SCHEMA,
        "status": "PASS_PRIVATE_CANONICAL_PQ_PARTITION_INVARIANCE"
        if all(gates.values())
        else "FAIL_CLOSED_CANONICAL_PQ_PARTITION_INVARIANCE",
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p91_canonical_pq_png_partition_invariance_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(json.dumps({"report_sha256": hashlib.sha256(payload).hexdigest(), "stable_evidence_id": report["stable_evidence_id"], "status": report["status"]}, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
