"""Validate the stock-first registry against frozen BlueNeg roll/frame metadata."""

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

from src.real_film.stock_registry import crosscheck_blueneg, validate_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "film_stock_registry.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "evidence" / "frames.jsonl")
    parser.add_argument("--rolls", type=Path, default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "evidence" / "rolls.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "real_film" / "stock_registry_v1" / "report.json")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in args.frames.read_text(encoding="utf-8").splitlines() if line]
    rolls = [json.loads(line) for line in args.rolls.read_text(encoding="utf-8").splitlines() if line]
    validation = validate_registry(registry)
    crosscheck = crosscheck_blueneg(registry, frames, rolls)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "registry_id": registry["registry_id"],
        "software_commit": commit,
        "registry_sha256": sha(args.registry),
        "frames_sha256": sha(args.frames),
        "rolls_sha256": sha(args.rolls),
        "validation": validation,
        "blueneg_crosscheck": crosscheck,
        "coverage": registry["coverage"],
        "passed": bool(crosscheck["passed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(json.dumps({
        "passed": report["passed"],
        "stock_rows": validation["stock_rows"],
        "selected_pilots": validation["selected_pilots"],
        "named_stock_data_s1": registry["coverage"]["named_stock_data_s1_or_higher"],
        "s2_experts": registry["coverage"]["named_stock_experts_s2_or_higher"],
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
    }, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
