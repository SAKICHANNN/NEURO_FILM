#!/usr/bin/env python3
"""Build and replay-verify the frozen SF3.A0K display stimulus pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_display_stimulus import (
    build_pack,
    load_contract,
    write_report,
)


def _copy_create(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"canonical pack already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copyfile)


def _inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/sf3_a0k_three_stock_display_stimulus_v1.json")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    formal_root = ROOT / contract["output"]["formal_run_root"]
    reports = []
    hashes = []
    for run_id in ("run_a", "run_b"):
        run_root = formal_root / run_id
        report = build_pack(contract, ROOT, run_root / "pack")
        hashes.append(write_report(report, run_root / contract["output"]["report_name"]))
        reports.append(report)
    if reports[0] != reports[1] or len(set(hashes)) != 1:
        raise RuntimeError("SF3.A0K formal builds differ")
    canonical = ROOT / contract["output"]["canonical_pack_root"]
    _copy_create(formal_root / "run_a" / "pack", canonical)
    if _inventory(canonical) != reports[0]["inventory"]:
        raise RuntimeError("SF3.A0K canonical inventory differs")
    print(json.dumps({"report_sha256": hashes[0], "report": reports[0]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
