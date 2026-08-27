#!/usr/bin/env python3
"""Audit two exact HDR+ merged DNGs for silent-thumbnail ingress."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_working_image
from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p279-hdrplus-merged-dng-ingress-result.v1"


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


def _attempt(
    path: Path, *, expected_width: int, expected_height: int
) -> dict[str, Any]:
    try:
        working = load_working_image(path)
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "accepted": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "safe_outcome": True,
        }
    pixels = working.pixels
    full_dimensions = pixels.shape == (expected_height, expected_width, 3)
    owned = bool(pixels.flags.owndata)
    contiguous = bool(pixels.flags.c_contiguous)
    writable = bool(pixels.flags.writeable)
    finite = bool(np.isfinite(pixels).all())
    return {
        "accepted": True,
        "c_contiguous": contiguous,
        "finite": finite,
        "full_dimensions": full_dimensions,
        "maximum": float(np.max(pixels)),
        "minimum": float(np.min(pixels)),
        "owned": owned,
        "pixel_sha256": _sha256_bytes(np.ascontiguousarray(pixels).tobytes()),
        "safe_outcome": full_dimensions
        and owned
        and contiguous
        and writable
        and finite,
        "shape": list(pixels.shape),
        "working_space": working.working_space,
        "writeable": writable,
    }


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    parent = ROOT / "docs/evidence/P269_HDRPLUS_ONE_BURST_ACQUISITION_RESULT.json"
    if _sha256_file(parent) != config["parent_p269_evidence_sha256"]:
        raise ValueError("P269 parent evidence SHA-256 mismatch")
    root = ROOT / config["source_root"]
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    snapshots: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        path = root / row["path"]
        source_bytes = path.read_bytes()
        snapshots[row["path"]] = source_bytes
        if (
            len(source_bytes) != row["bytes"]
            or _sha256_bytes(source_bytes) != row["sha256"]
        ):
            raise ValueError(f"source identity mismatch: {row['version']}")
        record = {
            "bytes": row["bytes"],
            "path": row["path"],
            "sha256": row["sha256"],
            "version": row["version"],
            **_attempt(
                path,
                expected_width=config["expected_full_width"],
                expected_height=config["expected_full_height"],
            ),
        }
        records.append(record)

    with tempfile.TemporaryDirectory(prefix="p279_", dir=ROOT / "tmp") as directory:
        truncated = Path(directory) / "truncated.dng"
        truncated.write_bytes(next(iter(snapshots.values()))[:64])
        truncated_result = _attempt(
            truncated,
            expected_width=config["expected_full_width"],
            expected_height=config["expected_full_height"],
        )

    records.sort(key=lambda value: value["version"])
    gates = {
        "all_rows_safe": len(records) == config["gates"]["required_row_count"]
        and all(row["safe_outcome"] for row in records),
        "no_silent_known_thumbnail": all(
            not row["accepted"]
            or row.get("shape")
            != [
                config["known_metadata_page_height"],
                config["known_metadata_page_width"],
                3,
            ]
            for row in records
        ),
        "source_immutable": all(
            (root / name).read_bytes() == value for name, value in snapshots.items()
        ),
        "truncation_rejected": not truncated_result["accepted"],
    }
    passed = all(gates.values())
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "gates": gates,
        "parent_p269_evidence_sha256": config["parent_p269_evidence_sha256"],
        "records": records,
        "status": "PASS_PRIVATE_HDRPLUS_MERGED_DNG_INGRESS_SAFETY"
        if passed
        else "FAIL_CLOSED_HDRPLUS_MERGED_DNG_INGRESS_SAFETY",
        "truncation": truncated_result,
    }
    scientific_bytes = canonical_json_bytes(scientific)
    return {
        "schema": REPORT_SCHEMA,
        "status": scientific["status"],
        "scientific": scientific,
        "execution": {"scientific_sha256": _sha256_bytes(scientific_bytes)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
