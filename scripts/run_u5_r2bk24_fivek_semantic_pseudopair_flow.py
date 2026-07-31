#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_semantic_pseudopair_flow import evaluate, load_config, report_bytes
from src.eval.canonicalizer_consensus_appearance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2bk24_fivek_semantic_pseudopair_flow_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = evaluate(ROOT, config, config_sha256=sha256_file(args.config), software_commit=commit)
    payload = report_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"report_sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"decision={report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
