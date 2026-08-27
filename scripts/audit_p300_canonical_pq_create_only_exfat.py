#!/usr/bin/env python3
"""Formal P300 canonical-P/exFAT create-only canonical PQ audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
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
from src.preprocess.aces2_p3d65_canonical_pq_png import (
    publish_acescg_p3d65_1000nit_canonical_pq_png_v1,
)
from src.preprocess.canonical_pq_png import CanonicalStreamingRec2100PqPngWriter
from src.preprocess.color_management import REC2100_PQ_CICP
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples

SCHEMA = "neuro_film.p300_canonical_pq_create_only_exfat_result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_git_file(commit: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return _sha256_bytes(payload)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


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


def _write(
    writer: CanonicalStreamingRec2100PqPngWriter,
    samples: np.ndarray,
    sizes: list[int],
) -> None:
    start = 0
    for size in sizes:
        writer.write_rows(start, np.ascontiguousarray(samples[start : start + size]))
        start += size


def _failure_control(path: Path, samples: np.ndarray, *, late: bool) -> dict[str, Any]:
    foreign = b"late-foreign" if late else b"existing-foreign"
    if not late:
        path.write_bytes(foreign)
    writer = CanonicalStreamingRec2100PqPngWriter(
        path, width=samples.shape[1], height=samples.shape[0]
    )
    _write(writer, samples, [7, 7, 7, 7, 1])
    if late:
        path.write_bytes(foreign)
    rejected = False
    error_type = ""
    try:
        writer.finish()
    except FileExistsError as exc:
        rejected = True
        error_type = type(exc).__name__
    result = {
        "rejected": rejected,
        "error_type": error_type,
        "destination_preserved": path.read_bytes() == foreign,
        "stage_residue": writer.temporary.exists(),
    }
    path.unlink()
    return result


def run(config_path: Path, output_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    fixture = config["formal_fixture"]
    bindings = config["bindings"]
    root = ROOT / fixture["logical_output_root"]
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError("P300 formal output root must start empty")

    source = np.ascontiguousarray(build_p226_parity_fixture().reshape(29, 34, 3))
    destination = root / "probe.png"
    reported_sha, samples, encoded = publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
        source,
        destination,
        row_count=7,
        reverse_partition=reverse,
    )
    png_bytes = destination.read_bytes()
    png_sha = _sha256_bytes(png_bytes)
    sample_sha = _sha256_bytes(samples.tobytes())
    decoded_sha = sha256_rec2100_pq_rgb16_png_samples(destination, width=34, height=29)
    cicp_hex = _cicp(destination)
    destination.unlink()

    existing = _failure_control(root / "existing.png", samples, late=False)
    late = _failure_control(root / "late.png", samples, late=True)
    residue = sorted(path.name for path in root.iterdir())
    resolved_root = Path(os.path.realpath(root))
    root.rmdir()

    parent_exact = (
        _sha256_git_file(bindings["parent_commit"], bindings["canonical_writer_path"])
        == bindings["canonical_writer_parent_sha256"]
    )
    static_bindings_exact = all(
        (
            _sha256_file(ROOT / bindings["create_only_path"])
            == bindings["create_only_sha256"],
            _sha256_file(ROOT / bindings["p238_evidence_path"])
            == bindings["p238_evidence_sha256"],
            _sha256_file(ROOT / bindings["p299_evidence_path"])
            == bindings["p299_evidence_sha256"],
        )
    )
    gates = {
        "parent_bindings_exact": parent_exact and static_bindings_exact,
        "source_exact": _sha256_bytes(source.tobytes()) == fixture["source_sha256"],
        "existing_destination_preserved": existing["rejected"]
        and existing["destination_preserved"],
        "late_destination_preserved": late["rejected"]
        and late["destination_preserved"],
        "canonical_png_bytes_exact": reported_sha
        == png_sha
        == fixture["expected_png_sha256"],
        "rgb16_samples_exact": decoded_sha
        == sample_sha
        == fixture["expected_sample_sha256"],
        "rec2100_pq_cicp_exact": cicp_hex == REC2100_PQ_CICP.hex(),
        "stage_and_media_residue_zero": not residue
        and not existing["stage_residue"]
        and not late["stage_residue"],
        "p_backed_output": resolved_root.drive.casefold() == "p:",
        "encoded_finite_in_range": bool(
            encoded.dtype == np.float32
            and encoded.flags.c_contiguous
            and np.isfinite(encoded).all()
            and np.all(encoded >= 0.0)
            and np.all(encoded <= 1.0)
        ),
    }
    status = (
        "PASS_PRIVATE_CANONICAL_PQ_CREATE_ONLY_EXFAT"
        if all(gates.values())
        else "FAIL_CLOSED_CANONICAL_PQ_CREATE_ONLY_EXFAT"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P300",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "canonical_writer_sha256": _sha256_file(
                ROOT / bindings["canonical_writer_path"]
            ),
            "create_only_sha256": _sha256_file(ROOT / bindings["create_only_path"]),
            "runner_sha256": _sha256_file(ROOT / bindings["runner_path"]),
        },
        "fixture": {
            "shape": list(source.shape),
            "source_sha256": _sha256_bytes(source.tobytes()),
            "float_output_sha256": _sha256_bytes(encoded.tobytes()),
            "sample_sha256": sample_sha,
            "png_bytes": len(png_bytes),
            "png_sha256": png_sha,
            "cicp_hex": cicp_hex,
            "resolved_volume": resolved_root.drive.upper(),
        },
        "controls": {"existing": existing, "late": late},
        "residue": residue,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, reverse=args.reverse)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
