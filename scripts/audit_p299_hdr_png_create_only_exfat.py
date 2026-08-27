"""Formal P299 canonical-P/exFAT create-only HDR PNG transaction audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.color_management import REC2100_PQ_CICP
from src.preprocess.png_stream import (
    StreamingRec2100PqPngWriter,
    sha256_rec2100_pq_rgb16_png_samples,
)

SCHEMA = "neuro_film.p299_hdr_png_create_only_exfat_result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write(
    writer: StreamingRec2100PqPngWriter, samples: np.ndarray, sizes: list[int]
) -> None:
    start = 0
    for size in sizes:
        writer.write_rows(start, samples[start : start + size])
        start += size


def _failure_control(path: Path, samples: np.ndarray, *, late: bool) -> dict[str, Any]:
    foreign = b"late-foreign" if late else b"existing-foreign"
    if not late:
        path.write_bytes(foreign)
    writer = StreamingRec2100PqPngWriter(
        path, width=samples.shape[1], height=samples.shape[0]
    )
    writer.write_rows(0, samples)
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
    root = ROOT / fixture["logical_output_root"]
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(fixture["seed"]))
    samples = rng.integers(
        0,
        65536,
        size=(int(fixture["height"]), int(fixture["width"]), 3),
        dtype=np.uint16,
    )
    row_count = int(fixture["row_count"])
    sizes = [row_count] * (samples.shape[0] // row_count)
    if samples.shape[0] % row_count:
        sizes.append(samples.shape[0] % row_count)
    if reverse:
        sizes.reverse()

    existing = _failure_control(root / "existing.png", samples, late=False)
    late = _failure_control(root / "late.png", samples, late=True)
    destination = root / "probe.png"
    writer = StreamingRec2100PqPngWriter(
        destination,
        width=samples.shape[1],
        height=samples.shape[0],
    )
    _write(writer, samples, sizes)
    reported_sha = writer.finish()
    png_bytes = destination.read_bytes()
    file_sha = _sha256_bytes(png_bytes)
    sample_sha = _sha256_bytes(samples.tobytes())
    decoded_sha = sha256_rec2100_pq_rgb16_png_samples(
        destination, width=samples.shape[1], height=samples.shape[0]
    )
    resolved_root = Path(os.path.realpath(root))
    stage_residue = sorted(path.name for path in root.glob("*.stream.tmp"))
    destination.unlink()

    bindings = config["bindings"]
    gates = {
        "parent_bindings_exact": all(
            (
                _sha256_file(ROOT / bindings["create_only_path"])
                == bindings["create_only_sha256"],
                _sha256_file(ROOT / bindings["p91_evidence_path"])
                == bindings["p91_evidence_sha256"],
                _sha256_file(ROOT / bindings["exfat_parent_evidence_path"])
                == bindings["exfat_parent_evidence_sha256"],
            )
        ),
        "existing_destination_preserved": existing["rejected"]
        and existing["destination_preserved"],
        "late_destination_preserved": late["rejected"]
        and late["destination_preserved"],
        "successful_png_bytes_exact": reported_sha == file_sha,
        "rgb16_samples_exact": decoded_sha == sample_sha,
        "rec2100_pq_cicp_exact": b"cICP" + REC2100_PQ_CICP in png_bytes,
        "stage_residue_zero": not stage_residue
        and not existing["stage_residue"]
        and not late["stage_residue"],
        "p_backed_output": resolved_root.drive.casefold() == "p:",
    }
    status = (
        "PASS_PRIVATE_HDR_PNG_CREATE_ONLY_EXFAT"
        if all(gates.values())
        else "FAIL_CLOSED_HDR_PNG_CREATE_ONLY_EXFAT"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P299",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "png_stream_sha256": _sha256_file(ROOT / bindings["png_stream_path"]),
            "create_only_sha256": _sha256_file(ROOT / bindings["create_only_path"]),
            "runner_sha256": _sha256_file(ROOT / bindings["runner_path"]),
        },
        "fixture": {
            "width": samples.shape[1],
            "height": samples.shape[0],
            "sample_sha256": sample_sha,
            "png_bytes": len(png_bytes),
            "png_sha256": file_sha,
            "resolved_volume": resolved_root.drive.upper(),
        },
        "controls": {"existing": existing, "late": late},
        "stage_residue": stage_residue,
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
