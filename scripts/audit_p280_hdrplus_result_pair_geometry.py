#!/usr/bin/env python3
"""Test whether exact HDR+ merged-DNG/final-JPEG results share geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_working_image
from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p280-hdrplus-result-pair-geometry-result.v1"
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


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


def _gradient_luma(rgb: np.ndarray, *, width: int, height: int) -> np.ndarray:
    luma = np.asarray(rgb, dtype=np.float32) @ LUMA
    small = cv2.resize(luma, (width, height), interpolation=cv2.INTER_AREA)
    dx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(dx, dy, dtype=np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64, copy=False).ravel()
    y = right.astype(np.float64, copy=False).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    parent = ROOT / "docs/evidence/P279_HDRPLUS_MERGED_DNG_INGRESS_RESULT.json"
    if _sha256_file(parent) != config["parent_p279_evidence_sha256"]:
        raise ValueError("P279 parent evidence SHA-256 mismatch")
    root = ROOT / config["source_root"]
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    snapshots: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        observation_path = root / row["observation"]
        target_path = root / row["target"]
        observation_bytes = observation_path.read_bytes()
        target_bytes = target_path.read_bytes()
        snapshots[row["observation"]] = observation_bytes
        snapshots[row["target"]] = target_bytes
        if _sha256_bytes(observation_bytes) != row["observation_sha256"]:
            raise ValueError(f"observation identity mismatch: {row['version']}")
        if (
            len(target_bytes) != row["target_bytes"]
            or _sha256_bytes(target_bytes) != row["target_sha256"]
        ):
            raise ValueError(f"target identity mismatch: {row['version']}")

        working = load_working_image(observation_path)
        with Image.open(target_path) as image:
            target = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(
                255.0
            )
        expected_shape = (config["expected_height"], config["expected_width"], 3)
        dimensions_exact = (
            working.pixels.shape == expected_shape and target.shape == expected_shape
        )
        observation_gradient = _gradient_luma(
            working.pixels,
            width=config["downsample_width"],
            height=config["downsample_height"],
        )
        target_gradient = _gradient_luma(
            target,
            width=config["downsample_width"],
            height=config["downsample_height"],
        )
        identity = _correlation(observation_gradient, target_gradient)
        flip = _correlation(observation_gradient, np.fliplr(target_gradient))
        shifted = _correlation(
            observation_gradient,
            np.roll(target_gradient, config["shift_columns"], axis=1),
        )
        records.append(
            {
                "dimensions_exact": dimensions_exact,
                "flip_correlation": flip,
                "flip_margin": identity - flip,
                "identity_correlation": identity,
                "observation_sha256": row["observation_sha256"],
                "shift_correlation": shifted,
                "shift_margin": identity - shifted,
                "target_sha256": row["target_sha256"],
                "version": row["version"],
            }
        )

    records.sort(key=lambda value: value["version"])
    gates = {
        "dimensions_exact": all(row["dimensions_exact"] for row in records),
        "identity_correlation": all(
            row["identity_correlation"]
            >= config["gates"]["minimum_identity_correlation"]
            for row in records
        ),
        "flip_margin": all(
            row["flip_margin"] >= config["gates"]["minimum_flip_margin"]
            for row in records
        ),
        "shift_margin": all(
            row["shift_margin"] >= config["gates"]["minimum_shift_margin"]
            for row in records
        ),
        "source_immutable": all(
            (root / name).read_bytes() == value for name, value in snapshots.items()
        ),
    }
    passed = len(records) == config["gates"]["required_row_count"] and all(
        gates.values()
    )
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "gates": gates,
        "parent_p279_evidence_sha256": config["parent_p279_evidence_sha256"],
        "records": records,
        "status": "PASS_PRIVATE_HDRPLUS_RESULT_PAIR_GEOMETRY"
        if passed
        else "FAIL_CLOSED_HDRPLUS_RESULT_PAIR_GEOMETRY",
        "target_pixel_decode_count": len(records),
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
