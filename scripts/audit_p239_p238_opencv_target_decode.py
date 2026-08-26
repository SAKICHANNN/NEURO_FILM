#!/usr/bin/env python3
"""Formal target-runtime decode audit for the exact P238 PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import cv2
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
from src.preprocess.opencv_rec2100_pq_decode import (
    OpenCvRec2100PqDecodeError,
    decode_validated_rec2100_pq_rgb16_png_opencv_v1,
)

CONFIG = ROOT / "configs/p239_p238_opencv_target_decode_v1.json"
SCHEMA = "neuro-film.p239-p238-opencv-target-decode-contract.v1"


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


def _validate_contract() -> tuple[dict[str, Any], Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_P238_PIXEL_DECODE":
        raise RuntimeError("P239 contract is not frozen")
    evidence = ROOT / config["p238"]["evidence_path"]
    if sha256_file(evidence) != config["p238"]["evidence_sha256"]:
        raise RuntimeError("P239 P238 evidence identity differs")
    cv2_binary = Path(cv2.__file__).resolve().parent / "cv2.pyd"
    if cv2.__version__ != config["runtime"]["opencv_version"]:
        raise RuntimeError("P239 OpenCV version differs")
    if sha256_file(cv2_binary) != config["runtime"]["cv2_pyd_sha256"]:
        raise RuntimeError("P239 OpenCV binary differs")
    build = cv2.getBuildInformation()
    if config["runtime"]["opencv_source_revision"] not in build:
        raise RuntimeError("P239 OpenCV source revision differs")
    if f"PNG:                         build (ver {config['runtime']['libpng_version']})" not in build:
        raise RuntimeError("P239 libpng version differs")
    return config, cv2_binary


def _replace_cicp(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = bytes(payload[offset + 4 : offset + 8])
        if kind == b"cICP":
            body_start = offset + 8
            payload[body_start] = 1
            crc = zlib.crc32(kind + payload[body_start : body_start + length]) & 0xFFFFFFFF
            payload[body_start + length : body_start + length + 4] = struct.pack(">I", crc)
            path.write_bytes(payload)
            return
        offset += 12 + length
    raise RuntimeError("P239 source cICP missing")


def _rejects(path: Path) -> bool:
    try:
        decode_validated_rec2100_pq_rgb16_png_opencv_v1(
            path, width=34, height=29, mode="memory"
        )
    except OpenCvRec2100PqDecodeError:
        return True
    return False


def run(order: str, scratch: Path) -> dict[str, Any]:
    config, cv2_binary = _validate_contract()
    source = np.ascontiguousarray(build_p226_parity_fixture().reshape(29, 34, 3))
    png = scratch / "p238.png"
    png_sha, samples, _ = publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
        source, png
    )
    png_before = png.read_bytes()
    modes = ["file", "memory"]
    if order == "memory_first":
        modes.reverse()
    decoded: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for mode in modes:
        decoded[mode], hashes[mode] = decode_validated_rec2100_pq_rgb16_png_opencv_v1(
            png, width=34, height=29, mode=mode  # type: ignore[arg-type]
        )

    wrong_cicp = scratch / "wrong-cicp.png"
    wrong_cicp.write_bytes(png_before)
    _replace_cicp(wrong_cicp)
    bad_crc = scratch / "bad-crc.png"
    crc_bytes = bytearray(png_before)
    crc_bytes[-5] ^= 1
    bad_crc.write_bytes(crc_bytes)
    truncated = scratch / "truncated.png"
    truncated.write_bytes(png_before[:-7])

    expected_sample_sha = config["p238"]["sample_sha256"]
    gates = {
        "p238_png_sha_exact": png_sha == config["p238"]["png_sha256"],
        "p238_sample_sha_exact": sha256_bytes(samples.tobytes()) == expected_sample_sha,
        "file_decode_uint16_shape_exact": bool(
            decoded["file"].dtype == np.uint16
            and decoded["file"].shape == (29, 34, 3)
            and decoded["file"].flags.c_contiguous
        ),
        "memory_decode_uint16_shape_exact": bool(
            decoded["memory"].dtype == np.uint16
            and decoded["memory"].shape == (29, 34, 3)
            and decoded["memory"].flags.c_contiguous
        ),
        "file_sample_hash_exact": hashes["file"] == expected_sample_sha,
        "memory_sample_hash_exact": hashes["memory"] == expected_sample_sha,
        "file_memory_arrays_exact": bool(
            np.array_equal(decoded["file"], decoded["memory"])
            and np.array_equal(decoded["file"], samples)
        ),
        "source_immutable": png.read_bytes() == png_before,
        "wrong_cicp_rejected": _rejects(wrong_cicp),
        "bad_crc_rejected": _rejects(bad_crc),
        "truncated_rejected": _rejects(truncated),
    }
    return {
        "schema": "neuro-film.p239-p238-opencv-target-decode-report.v1",
        "experiment_id": "P239",
        "status": "PASS_PRIVATE_P238_OPENCV_TARGET_DECODE"
        if all(gates.values())
        else "FAIL_CLOSED_P238_OPENCV_TARGET_DECODE",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "cv2_pyd_sha256": sha256_file(cv2_binary),
            "libpng": config["runtime"]["libpng_version"],
            "opencv_cicp_visibility": False,
        },
        "mechanism": {
            "png_bytes": len(png_before),
            "png_sha256": png_sha,
            "sample_sha256": expected_sample_sha,
            "file_rgb_sha256": sha256_bytes(decoded["file"].tobytes()),
            "memory_rgb_sha256": sha256_bytes(decoded["memory"].tobytes()),
            "minimum_code": int(decoded["file"].min()),
            "maximum_code": int(decoded["file"].max()),
        },
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("file_first", "memory_first"), required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    report = run(args.order, args.scratch)
    args.report.write_bytes(canonical_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
