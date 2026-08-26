#!/usr/bin/env python3
"""Measure exact bounded tile parallelism on the three-stock baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import load_render_profile, render_resolved_safe_lab_rgb

STYLES = ("velvia_50", "portra_400", "ektar_100")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=3000)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if min(args.width, args.height, args.tile_size) < 1 or args.workers < 2:
        raise ValueError("dimensions must be positive and workers must be at least 2")
    source_bytes = args.source.read_bytes()
    with Image.open(args.source) as image:
        rgb8 = np.asarray(
            image.convert("RGB").resize(
                (args.width, args.height), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
    source = np.ascontiguousarray(rgb8.astype(np.float32) / 255.0)
    profile = load_render_profile(
        ROOT / "configs" / "render_profiles" / "safe_rich_v1.json", root=ROOT
    )
    statistics = json.loads(
        (ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8")
    )["styles"]
    styles = STYLES if args.order == "forward" else tuple(reversed(STYLES))
    rows: list[dict[str, object]] = []
    for style in styles:
        kwargs = {
            "style": style,
            "style_statistics": statistics[style],
            "style_parameters": profile["style_parameters"][style],
            "guardrails": load_guardrail_config(
                ROOT / "configs" / "color_guardrails.json", style
            ),
            "seed": 1729,
            "tile_size": args.tile_size,
        }
        start = time.perf_counter()
        serial = render_resolved_safe_lab_rgb(source, **kwargs, tile_workers=1)
        serial_seconds = time.perf_counter() - start
        start = time.perf_counter()
        parallel = render_resolved_safe_lab_rgb(
            source, **kwargs, tile_workers=args.workers
        )
        parallel_seconds = time.perf_counter() - start
        if not np.array_equal(serial, parallel):
            raise RuntimeError(f"parallel output drifted for {style}")
        rows.append(
            {
                "style": style,
                "output_float32_sha256": _sha256_bytes(serial.tobytes()),
                "output_srgb16_sha256": _sha256_bytes(
                    np.rint(serial * 65535.0).astype(np.uint16).tobytes()
                ),
                "serial_seconds": serial_seconds,
                "parallel_seconds": parallel_seconds,
                "wall_ratio": parallel_seconds / serial_seconds,
            }
        )
    stable = {
        "schema": "neuro-film.three-stock-exact-tile-parallel-benchmark.v1",
        "source_file_sha256": _sha256_bytes(source_bytes),
        "resized_rgb8_sha256": _sha256_bytes(rgb8.tobytes()),
        "width": args.width,
        "height": args.height,
        "tile_size": args.tile_size,
        "workers": args.workers,
        "outputs": {
            row["style"]: {
                "output_float32_sha256": row["output_float32_sha256"],
                "output_srgb16_sha256": row["output_srgb16_sha256"],
            }
            for row in rows
        },
    }
    report = {
        **stable,
        "order": args.order,
        "rows": rows,
        "stable_identity": _sha256_bytes(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
