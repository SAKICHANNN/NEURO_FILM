"""Run frozen U5.R2BL10 colour-blind hard-case retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_semantic_hard_case import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "configs/u5_r2bl10_filmmatch_semantic_hard_case_development_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = run_experiment(root=root, config=config, config_path=config_path, software_commit=commit)
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError("BL10 output is create-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "selector_choice_gain": report["selector_choice_gain"], "checks": report["checks"]}, sort_keys=True))
    print(report["stable_evidence_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
