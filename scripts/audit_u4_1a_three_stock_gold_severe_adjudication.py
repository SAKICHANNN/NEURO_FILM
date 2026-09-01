#!/usr/bin/env python3
"""Committed-head materialization audit for U4.1A."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.three_stock_gold_severe_adjudication import (
    canonical_bytes,
    run_gold_materialization,
    sha256_bytes,
)

CONFIG = ROOT / "configs/u4_1a_three_stock_gold_severe_adjudication_v1.json"
BOUND_PATHS = (
    "configs/u4_1a_three_stock_gold_severe_adjudication_v1.json",
    "docs/planning/U4_1A_THREE_STOCK_GOLD_SEVERE_ADJUDICATION_CONTRACT.md",
    "src/eval/three_stock_gold_severe_adjudication.py",
    "scripts/audit_u4_1a_three_stock_gold_severe_adjudication.py",
    "tests/test_u4_1a_three_stock_gold_severe_adjudication.py",
)


def _run(*args: str) -> bytes:
    completed = subprocess.run(args, cwd=ROOT, check=True, capture_output=True)
    return completed.stdout


def _git_blob(path: str) -> bytes:
    return _run("git", "show", f"HEAD:{path}")


def _tracked_clean() -> bool:
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    if not _tracked_clean():
        raise RuntimeError("U4.1A formal execution requires a clean tracked tree")
    bindings = {path: sha256_bytes(_git_blob(path)) for path in BOUND_PATHS}
    config_blob = _git_blob(BOUND_PATHS[0])
    config: dict[str, Any] = json.loads(config_blob)
    material = run_gold_materialization(
        config,
        ROOT,
        args.output_dir,
        reverse=args.order == "reverse",
    )
    scientific = dict(material["scientific_payload"])
    scientific["execution"] = {
        "source_commit": commit,
        "git_blob_bindings": bindings,
        "tracked_diff_clean": True,
        "order_is_scientifically_invariant": True,
        "new_data_downloads": 0,
        "network_reads": 0,
    }
    scientific_identity = sha256_bytes(canonical_bytes(scientific))
    report = {
        "schema": "neuro-film.u4-1a-three-stock-gold-materialization-formal-report.v1",
        "scientific_payload": scientific,
        "scientific_identity": scientific_identity,
        "status": scientific["status"],
    }
    encoded = canonical_bytes(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(encoded)
    (args.output_dir / "report.json").write_bytes(encoded)
    print(
        json.dumps(
            {
                "report_bytes": len(encoded),
                "report_sha256": hashlib.sha256(encoded).hexdigest(),
                "scientific_identity": scientific_identity,
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
