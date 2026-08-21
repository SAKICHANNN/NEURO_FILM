#!/usr/bin/env python3
"""Formal P90 four-DNG ACES 2 HDR PQ PNG publication audit."""

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
from src.preprocess.aces2_pq_png import publish_working_image_aces2_hdr_pq_png_v1
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.raw_decode import load_raw_working_image

REPORT_SCHEMA = "neuro-film.p90-dng-aces2-hdr-pq-png-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _reject_pq(path: Path) -> str:
    try:
        load_working_image(path)
    except ValueError as error:
        message = str(error)
        if "unsupported PNG cICP color/dynamic-range signalling" not in message:
            raise AssertionError("production loader failed for another reason") from error
        return message
    raise AssertionError("production loader accepted private PQ output")


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load(config_path)
    bound = {
        "u1_4f_config_sha256": ROOT
        / "configs/u1_4f_dng_aces2_photographic_smoke_v1.json",
        "u1_4f_evidence_sha256": ROOT
        / "docs/evidence/U1_4F_DNG_ACES2_PHOTOGRAPHIC_SMOKE_RESULT.json",
        "u1_4g_evidence_sha256": ROOT
        / "docs/evidence/U1_4G_REC2100_PQ_PNG_RAIL_RESULT.json",
        "aces_adapter_sha256": ROOT / "src/preprocess/ocio_aces2_output.py",
    }
    bindings: dict[str, str] = {"p90_config_sha256": _sha256_file(config_path)}
    for key, path in bound.items():
        actual = _sha256_file(path)
        if actual != config["bindings"][key]:
            raise ValueError(f"binding mismatch: {key}")
        bindings[key] = actual
    source_config = _load(ROOT / config["source_config"])
    u1f_evidence = _load(bound["u1_4f_evidence_sha256"])
    report_path = ROOT / u1f_evidence["bindings"]["formal_report_path"]
    if _sha256_file(report_path) != u1f_evidence["bindings"]["formal_report_sha256"]:
        raise ValueError("U1.4F formal report identity mismatch")
    u1f_report = _load(report_path)
    frozen = {
        row["source_id"]: row["targets"]["hdr_rec2020_pq"]["output_sha256"]
        for row in u1f_report["metrics"]["rows"]
    }
    rows = list(source_config["rows"])
    if reverse:
        rows.reverse()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p90_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            source = ROOT / row["logical_path"]
            before = _sha256_file(source)
            if before != row["source_sha256"]:
                raise ValueError(f"source identity drift: {row['source_id']}")
            working = load_raw_working_image(source)
            output = scratch / f"{row['source_id']}.png"
            png_sha, samples, encoded = publish_working_image_aces2_hdr_pq_png_v1(
                working,
                output,
                row_count=int(config["row_count"]),
                reverse_partition=reverse,
            )
            sample_sha = _sha256_bytes(samples.tobytes())
            records.append(
                {
                    "encoded_maximum": float(encoded.max()),
                    "encoded_minimum": float(encoded.min()),
                    "encoded_output_sha256": _array_sha256(encoded),
                    "encoded_output_matches_u1_4f": _array_sha256(encoded)
                    == frozen[row["source_id"]],
                    "finite": bool(np.isfinite(encoded).all()),
                    "in_unit": bool(np.all((encoded >= 0.0) & (encoded <= 1.0))),
                    "png_file_sha256": png_sha,
                    "production_rejection": _reject_pq(output),
                    "sample_readback_exact": sha256_rec2100_pq_rgb16_png_samples(
                        output,
                        width=working.pixels.shape[1],
                        height=working.pixels.shape[0],
                    )
                    == sample_sha,
                    "sample_sha256": sample_sha,
                    "shape": list(encoded.shape),
                    "source_id": row["source_id"],
                    "source_sha256": before,
                    "source_unchanged": _sha256_file(source) == before,
                    "transfer_state": working.transfer_state,
                    "working_space": working.working_space,
                }
            )
            del encoded, samples, working
    records.sort(key=lambda row: row["source_id"])
    gates = {
        "bindings_exact": len(bindings) == 5,
        "all_encoded_outputs_match_u1_4f": all(
            row["encoded_output_matches_u1_4f"] for row in records
        ),
        "all_finite_in_unit": all(row["finite"] and row["in_unit"] for row in records),
        "all_sample_readbacks_exact": all(row["sample_readback_exact"] for row in records),
        "all_sources_unchanged": all(row["source_unchanged"] for row in records),
        "all_working_boundaries_exact": all(
            row["working_space"] == "linear_srgb"
            and row["transfer_state"] == "scene_linear"
            for row in records
        ),
        "production_loader_still_rejects": all(
            "unsupported PNG cICP color/dynamic-range signalling"
            in row["production_rejection"]
            for row in records
        ),
        "required_rows_exact": len(records) == int(config["required_rows"]),
    }
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "contract_id": config["contract_id"],
        "gates": gates,
        "records": records,
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_FOUR_DNG_ACES2_HDR_PQ_PNG"
        if all(gates.values())
        else "FAIL_CLOSED_FOUR_DNG_ACES2_HDR_PQ_PNG",
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p90_dng_aces2_hdr_pq_png_v1.json"),
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
    print(json.dumps({"report_sha256": _sha256_bytes(payload), "stable_evidence_id": report["stable_evidence_id"], "status": report["status"]}, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
