#!/usr/bin/env python3
"""Run the frozen U5.R2B fixed global operator frontier audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.global_frontier import (  # noqa: E402
    build_blind_contact_sheets,
    evaluate_manifest,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2b_global_operator_frontier_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "u5_r2b_global_frontier_v1" / "render_pass1" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "u5_r2b_global_frontier_v1" / "automatic_report.json",
    )
    parser.add_argument("--build-visual", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_manifest(root=ROOT, config=config, manifest_path=args.manifest)
    visual = {"status": "not_requested"}
    if args.build_visual and result["automatic_survivors"]:
        visual = {
            "status": "pending_blind_review",
            **build_blind_contact_sheets(
                root=ROOT,
                config=config,
                manifest_path=args.manifest,
                survivors=result["automatic_survivors"],
                output_dir=args.output.parent / "visual",
            ),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        **result,
        "visual": visual,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "automatic_survivors": result["automatic_survivors"],
                "automatic_decision": result["automatic_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
