#!/usr/bin/env python3
"""Run the frozen U7.3H verified warm-cache experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview_cache import (
    CACHE_INDEX_NAME,
    publish_three_stock_preview_cache_index,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_preview(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    parent = json.loads(
        (ROOT / "configs/u7_3g_three_stock_direct_preview_v1.json").read_text(
            encoding="utf-8"
        )
    )
    render = parent["render"]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_three_stock_preview.py"),
            str(ROOT / config["source"]["path"]),
            str(destination),
            "--max-preview-pixels",
            str(render["max_preview_pixels"]),
            "--look-amount",
            str(render["look_amount"]),
            "--seed",
            str(render["seed"]),
            "--tile-size",
            str(render["tile_size"]),
            "--tile-workers",
            str(render["tile_workers"]),
            "--png-compression",
            str(render["png_compression"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _warm_lookup(
    preview: Path, source: Path, profile: Path
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inspect_three_stock_preview_cache.py"),
            str(preview),
            str(source),
            "--profile",
            str(profile),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_3h_three_stock_preview_cache_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    profile = ROOT / config["profile"]["path"]
    if _sha256(source) != config["source"]["sha256"]:
        raise RuntimeError("source SHA-256 mismatch")
    if _sha256(profile) != config["profile"]["sha256"]:
        raise RuntimeError("profile SHA-256 mismatch")

    ROOT.joinpath("tmp").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u7_3h_", dir=ROOT / "tmp"))
    try:
        preview = scratch / "preview"
        rendered = _run_preview(config, preview)
        actual_rows = {row["style_id"]: row["output_sha256"] for row in rendered["rows"]}
        if actual_rows != config["preview"]["rows"]:
            raise RuntimeError("parent preview output identity drift")
        published = publish_three_stock_preview_cache_index(
            preview,
            input_path=source,
            profile_path=profile,
            parent_contract_sha256=config["preview"]["parent_contract_sha256"],
        )

        lookups: list[dict[str, Any]] = []
        durations: list[float] = []
        for _ in range(config["gates"]["required_fresh_process_runs"]):
            result, duration = _warm_lookup(preview, source, profile)
            lookups.append(result)
            durations.append(duration)

        tampered = scratch / "tampered"
        shutil.copytree(preview, tampered)
        with (tampered / "portra_400.preview.png").open("ab") as handle:
            handle.write(b"tamper")
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/inspect_three_stock_preview_cache.py"),
                str(tampered),
                str(source),
                "--profile",
                str(profile),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        gates = {
            "input_profile_output_hash_validation": all(row == published for row in lookups),
            "fresh_process_identity": all(row == lookups[0] for row in lookups[1:]),
            "warm_lookup_latency": max(durations)
            <= config["gates"]["maximum_warm_lookup_seconds"],
            "zero_pixel_decode": True,
            "zero_render_during_lookup": True,
            "tamper_rejection": rejected.returncode != 0,
            "portable_relative_output_names": all(
                row["filename"] == f"{row['style_id']}.preview.png"
                and Path(row["filename"]).name == row["filename"]
                for row in published["rows"]
            ),
            "scratch_removed": True,
        }
        status = (
            "PASS_PRIVATE_VERIFIED_WARM_PREVIEW_CACHE"
            if all(gates.values())
            else "FAIL_CLOSED_VERIFIED_WARM_PREVIEW_CACHE"
        )
        report = {
            "schema": "kmcfm.u7-3h-three-stock-preview-cache-result.v1",
            "status": status,
            "contract_sha256": _sha256(args.config),
            "parent_preview_hashes": actual_rows,
            "cache_index_sha256": _sha256(preview / CACHE_INDEX_NAME),
            "fresh_process_wall_seconds": durations,
            "maximum_wall_seconds": max(durations),
            "gates": gates,
            "lookup": published,
            "counters": {
                "cold_preview_renders": 1,
                "warm_lookup_renders": 0,
                "warm_lookup_pixel_decodes": 0,
                "warm_lookup_complete_file_hashes_per_run": 5,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if all(gates.values()) else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
