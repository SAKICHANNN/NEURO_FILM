from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.eval.fivek_triangular_logit_transport_development import run_development


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = run_development(
        root=ROOT,
        config=config,
        config_path=config_path,
        output_dir=args.output_dir.resolve(),
        software_commit=commit,
    )
    print(json.dumps(result["report"]["gates"], indent=2, sort_keys=True))
    print(f"automatic_pass={result['report']['automatic_pass']}")
    print(f"report_sha256={result['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
