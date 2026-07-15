"""Freeze exact BlueNeg paths for the four stock-first metadata pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.stock_acquisition import (  # noqa: E402
    build_stock_pilot_acquisition,
    write_stock_pilot_acquisition,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "film_stock_registry.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "evidence" / "frames.jsonl")
    parser.add_argument("--rolls", type=Path, default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "evidence" / "rolls.jsonl")
    parser.add_argument("--inventory", type=Path, default=ROOT / "data" / "raw" / "blueneg" / "remote_inventory.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_film" / "stock_pilots_v1")
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest, report = build_stock_pilot_acquisition(
        json.loads(args.registry.read_text(encoding="utf-8")),
        _jsonl(args.frames),
        _jsonl(args.rolls),
        json.loads(args.inventory.read_text(encoding="utf-8")),
        software_commit=commit,
        input_sha256={
            "registry": _sha256(args.registry),
            "frames": _sha256(args.frames),
            "rolls": _sha256(args.rolls),
            "remote_inventory": _sha256(args.inventory),
        },
    )
    hashes = write_stock_pilot_acquisition(
        args.output_dir / "acquisition.json",
        args.output_dir / "report.json",
        manifest,
        report,
    )
    print(json.dumps({
        "passed": report["passed"],
        "files": report["file_count"],
        "bytes": report["bytes"],
        "stocks": report["stock_summaries"],
        **hashes,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
