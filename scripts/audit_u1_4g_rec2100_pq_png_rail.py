#!/usr/bin/env python3
"""Formal U1.4G deterministic Rec.2100 PQ RGB16 PNG rail audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.color_management import REC2020_SDR_CICP, REC2100_PQ_CICP
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.png_stream import (
    StreamingRec2020PngWriter,
    StreamingRec2100PqPngWriter,
    sha256_rec2020_rgb16_png_samples,
    sha256_rec2100_pq_rgb16_png_samples,
)

REPORT_SCHEMA = "neuro_film.u1_4g_rec2100_pq_png_rail_result.v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key in ("contract", "implementation", "runner"):
        relative = config["bindings"][f"{key}_path"]
        actual = _sha256_file(ROOT / relative)
        if actual != config["bindings"][f"{key}_sha256"]:
            raise ValueError(f"binding mismatch for {key}")
        checked[f"{key}_path"] = relative
        checked[f"{key}_sha256"] = actual
    return checked


def _fixture() -> np.ndarray:
    samples = np.random.default_rng(1407).integers(
        0, 65536, size=(61, 97, 3), dtype=np.uint16
    )
    samples[0, 0] = [0, 32768, 65535]
    samples[-1, -1] = [65535, 1, 0]
    return samples


def _chunk_facts(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    offset = 8
    chunks: list[dict[str, Any]] = []
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        stored = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        chunks.append(
            {
                "crc_valid": stored == zlib.crc32(kind + data) & 0xFFFFFFFF,
                "kind": kind.decode("ascii"),
                "payload_hex": data.hex() if kind == b"cICP" else None,
            }
        )
        offset += length + 12
    return {"chunks": chunks, "file_sha256": hashlib.sha256(payload).hexdigest()}


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    fixture = _fixture()
    records = [(index, np.ascontiguousarray(fixture[index : index + 1])) for index in range(61)]
    if reverse:
        records.reverse()
    records.sort(key=lambda item: item[0])
    with tempfile.TemporaryDirectory(prefix="u1_4g_", dir=ROOT / "tmp") as directory:
        directory_path = Path(directory)
        pq_path = directory_path / "pq.png"
        sdr_path = directory_path / "sdr.png"
        pq_writer = StreamingRec2100PqPngWriter(pq_path, width=97, height=61)
        sdr_writer = StreamingRec2020PngWriter(sdr_path, width=97, height=61, bit_depth=16)
        for index, row in records:
            pq_writer.write_rows(index, row)
            sdr_writer.write_rows(index, row)
        pq_writer.finish()
        sdr_writer.finish()
        expected_sample_sha256 = hashlib.sha256(fixture.tobytes()).hexdigest()
        pq_sample_sha256 = sha256_rec2100_pq_rgb16_png_samples(
            pq_path, width=97, height=61
        )
        sdr_sample_sha256 = sha256_rec2020_rgb16_png_samples(
            sdr_path, width=97, height=61
        )
        pq_facts = _chunk_facts(pq_path)
        sdr_facts = _chunk_facts(sdr_path)
    pq_chunks = pq_facts["chunks"]
    sdr_chunks = sdr_facts["chunks"]
    gates = {
        "all_crc_valid": all(
            chunk["crc_valid"] for chunk in pq_chunks + sdr_chunks
        ),
        "exact_pq_cicp": pq_chunks[1]["kind"] == "cICP"
        and pq_chunks[1]["payload_hex"] == REC2100_PQ_CICP.hex(),
        "exact_sdr_cicp": sdr_chunks[1]["kind"] == "cICP"
        and sdr_chunks[1]["payload_hex"] == REC2020_SDR_CICP.hex(),
        "pq_chunk_order": [chunk["kind"] for chunk in pq_chunks]
        == ["IHDR", "cICP", "IDAT", "IEND"],
        "pq_sample_exact": pq_sample_sha256 == expected_sample_sha256,
        "sdr_sample_exact": sdr_sample_sha256 == expected_sample_sha256,
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": {
            "expected_sample_sha256": expected_sample_sha256,
            "fixture_shape": list(fixture.shape),
            "pq_file_sha256": pq_facts["file_sha256"],
            "pq_sample_sha256": pq_sample_sha256,
            "sdr_file_sha256": sdr_facts["file_sha256"],
            "sdr_sample_sha256": sdr_sample_sha256,
        },
        "node": config["node"],
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_REC2100_PQ_PNG_RAIL"
        if all(gates.values())
        else "FAIL_CLOSED_REC2100_PQ_PNG_RAIL",
    }
    scientific["stable_evidence_id"] = hashlib.sha256(
        canonical_json_bytes(scientific)
    ).hexdigest()
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u1_4g_rec2100_pq_png_rail_v1.json"),
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
