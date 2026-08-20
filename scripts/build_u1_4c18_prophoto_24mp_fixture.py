#!/usr/bin/env python3
"""Build a deterministic natural-ProPhoto scale fixture for U1.4C18."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fixture(
    source: Path, output: Path, *, width: int = 6000, height: int = 4000
) -> dict[str, object]:
    source = Path(source)
    output = Path(output)
    if not source.is_file() or output.exists() or width <= 0 or height <= 0:
        raise ValueError("fixture source/output/geometry is invalid")
    with tifffile.TiffFile(source) as document:
        if len(document.pages) != 1 or 34675 not in document.pages[0].tags:
            raise ValueError("fixture source must be a single-page ICC RGB TIFF")
        profile = bytes(document.pages[0].tags[34675].value)
        pixels = document.pages[0].asarray()
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("fixture source must be RGB16")
    resized = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_LANCZOS4)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            output,
            resized,
            photometric="rgb",
            metadata=None,
            extratags=[(34675, "B", len(profile), profile, False)],
        )
        with tifffile.TiffFile(output) as document:
            stored_profile = bytes(document.pages[0].tags[34675].value)
            stored = document.pages[0].asarray()
        if stored_profile != profile or not np.array_equal(stored, resized):
            raise ValueError("fixture exact readback failed")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "schema": "neuro-film.u1-4c18-prophoto-24mp-fixture.v1",
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "embedded_icc_sha256": hashlib.sha256(profile).hexdigest(),
        "width": width,
        "height": height,
        "pixels": width * height,
        "dtype": "uint16",
        "channels": 3,
        "resampling": "opencv-inter-lanczos4-encoded-prophoto-rgb16",
        "exact_sample_and_icc_readback": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build_fixture(
        args.source, args.output, width=args.width, height=args.height
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        if args.report.exists():
            raise ValueError("fixture report must be create-only")
        args.report.write_text(encoded, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
