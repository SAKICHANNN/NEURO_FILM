#!/usr/bin/env python3
"""Run the frozen U5.R2H1 safe-bank content-routing audit."""

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

from src.eval.global_frontier import sha256_file  # noqa: E402
from src.eval.safe_bank_content_routing import run_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2h1_safe_bank_content_routing_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_audit(ROOT, config)
    result.update(
        {
            "software_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "config_sha256": sha256_file(config_path),
        }
    )
    output = args.output or ROOT / str(config["output"])
    output = output if output.is_absolute() else ROOT / output
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "decision": result["decision"],
                "viable_primary_routers": result["viable_primary_routers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
