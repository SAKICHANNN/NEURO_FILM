#!/usr/bin/env python3
"""Formal P238 canonical P3-D65 Rec.2100-PQ RGB16 PNG audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p226_r1cv_rec2100_pq_runtime_compatibility import (
    build_p226_parity_fixture,
)
from src.preprocess import aces2_p3d65_canonical_pq_png as publisher
from src.preprocess.aces2_p3d65_canonical_pq_png import (
    publish_acescg_p3d65_1000nit_canonical_pq_png_v1,
)
from src.preprocess.ocio_aces2_output import (
    OCIO_VERSION,
    apply_aces2_output_packed,
    build_aces2_numeric_fixture,
)
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples

CONFIG = ROOT / "configs/p238_aces2_p3d65_canonical_pq_png_v1.json"
SCHEMA = "neuro-film.p238-aces2-p3d65-canonical-pq-png-contract.v1"
EVIDENCE_SHA = "2be3c6da6a80ab4ab75f3ecf7fa2756a383b34dea03822f6d123b1853b0a6b2a"
EXISTING_HASH = "c95554153a53588fccb121380afd1b34c7bd6a3f5e8171ab280592a5b30df923"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cicp(path: Path) -> str:
    payload = path.read_bytes()
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        body = payload[offset + 8 : offset + 8 + length]
        if kind == b"cICP":
            return body.hex()
        offset += 12 + length
    raise RuntimeError("cICP chunk missing")


def _validate_contract() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_IMPLEMENTATION":
        raise RuntimeError("P238 contract is not frozen")
    if config["inputs"]["p226_evidence_sha256"] != EVIDENCE_SHA:
        raise RuntimeError("P238 P226 evidence identity differs")
    if sha256_file(ROOT / config["inputs"]["p226_evidence"]) != EVIDENCE_SHA:
        raise RuntimeError("P238 P226 evidence bytes differ")
    for key in ("ocio_adapter", "canonical_writer", "strict_reader"):
        if sha256_file(ROOT / config["inputs"][key]) != config["inputs"][f"{key}_sha256"]:
            raise RuntimeError(f"P238 frozen {key} identity differs")
    if config["regression"]["output_sha256"] != EXISTING_HASH:
        raise RuntimeError("P238 existing-target regression identity differs")
    return config


def _atomic_failure(source: np.ndarray, scratch: Path) -> dict[str, bool]:
    invalid = scratch / "invalid.png"
    try:
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
            source.astype(np.float64), invalid
        )
    except ValueError:
        invalid_rejected = not invalid.exists()
    else:
        invalid_rejected = False

    existing = scratch / "existing.png"
    existing.write_bytes(b"retained")
    try:
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(source, existing)
    except FileExistsError:
        create_only = existing.read_bytes() == b"retained"
    else:
        create_only = False

    failed = scratch / "failed.png"
    original = publisher.CanonicalStreamingRec2100PqPngWriter.write_rows
    calls = 0

    def fail_second(self, row_start: int, samples: np.ndarray) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected P238 writer failure")
        original(self, row_start, samples)

    publisher.CanonicalStreamingRec2100PqPngWriter.write_rows = fail_second
    try:
        try:
            publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
                source, failed, row_count=7
            )
        except RuntimeError as exc:
            failure_atomic = "injected P238" in str(exc)
        else:
            failure_atomic = False
    finally:
        publisher.CanonicalStreamingRec2100PqPngWriter.write_rows = original
    failure_atomic = (
        failure_atomic
        and not failed.exists()
        and not failed.with_suffix(".png.stream.tmp").exists()
    )
    return {
        "invalid_input_rejected": invalid_rejected,
        "create_only": create_only,
        "failure_atomic": failure_atomic,
    }


def run(order: str, scratch: Path) -> dict[str, Any]:
    config = _validate_contract()
    source = np.ascontiguousarray(build_p226_parity_fixture().reshape(29, 34, 3))
    source_before = source.tobytes()
    names = ["forward", "reverse"]
    if order == "reverse":
        names.reverse()
    publications: dict[str, dict[str, Any]] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        path = scratch / f"{name}.png"
        digest, samples, encoded = publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
            source,
            path,
            row_count=config["publication"]["row_count"],
            reverse_partition=name == "reverse",
        )
        arrays[name] = (samples, encoded)
        publications[name] = {
            "png_bytes": path.stat().st_size,
            "png_sha256": digest,
            "sample_sha256": sha256_bytes(samples.tobytes()),
            "strict_readback_sha256": sha256_rec2100_pq_rgb16_png_samples(
                path, width=34, height=29
            ),
            "cicp_hex": _cicp(path),
            "minimum_code": int(samples.min()),
            "maximum_code": int(samples.max()),
        }

    forward_samples, forward_encoded = arrays["forward"]
    reverse_samples, reverse_encoded = arrays["reverse"]
    expected = np.ascontiguousarray(
        np.floor(forward_encoded.astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    )
    inherited = build_aces2_numeric_fixture()
    existing = apply_aces2_output_packed(inherited, "hdr_rec2020_pq")
    atomic = _atomic_failure(source, scratch)
    gates = {
        "fixture_rows_exact": source.reshape(-1, 3).shape[0] == 986,
        "input_immutable": source.tobytes() == source_before,
        "float_output_f32_contiguous_finite_in_0_1": bool(
            forward_encoded.dtype == np.float32
            and forward_encoded.flags.c_contiguous
            and np.isfinite(forward_encoded).all()
            and np.all(forward_encoded >= 0.0)
            and np.all(forward_encoded <= 1.0)
        ),
        "quantized_samples_exact": bool(np.array_equal(forward_samples, expected)),
        "partition_arrays_exact": bool(
            np.array_equal(forward_samples, reverse_samples)
            and np.array_equal(forward_encoded, reverse_encoded)
        ),
        "partition_png_bytes_exact": publications["forward"]["png_sha256"]
        == publications["reverse"]["png_sha256"],
        "strict_sample_readback_exact": all(
            item["sample_sha256"] == item["strict_readback_sha256"]
            for item in publications.values()
        ),
        "cicp_exact": all(
            item["cicp_hex"] == config["publication"]["cicp_hex"]
            for item in publications.values()
        ),
        "existing_target_hash_unchanged": sha256_bytes(existing.tobytes()) == EXISTING_HASH,
        **atomic,
    }
    return {
        "schema": "neuro-film.p238-aces2-p3d65-canonical-pq-png-report.v1",
        "experiment_id": "P238",
        "status": "PASS_PRIVATE_ACES2_P3D65_CANONICAL_PQ_PNG"
        if all(gates.values())
        else "FAIL_CLOSED_ACES2_P3D65_CANONICAL_PQ_PNG",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencolorio": OCIO_VERSION,
        },
        "identities": {
            "contract_sha256": sha256_file(CONFIG),
            "p226_evidence_sha256": EVIDENCE_SHA,
            "publisher_sha256": sha256_file(
                ROOT / "src/preprocess/aces2_p3d65_canonical_pq_png.py"
            ),
        },
        "mechanism": {
            "fixture_sha256": sha256_bytes(source.tobytes()),
            "float_output_sha256": sha256_bytes(forward_encoded.tobytes()),
            "float_minimum": float(forward_encoded.min()),
            "float_maximum": float(forward_encoded.max()),
            "official_black_float32": float(np.float32(7.30942701920867e-7)),
            "official_black_quantized_code": int(
                np.floor(np.float64(np.float32(7.30942701920867e-7)) * 65535.0 + 0.5)
            ),
            "existing_target_output_sha256": sha256_bytes(existing.tobytes()),
            "publications": {key: publications[key] for key in sorted(publications)},
        },
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    report = run(args.order, args.scratch)
    args.report.write_bytes(canonical_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
