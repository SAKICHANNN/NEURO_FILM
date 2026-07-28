#!/usr/bin/env python
"""Render and evaluate the frozen U5.R2AO3 factorized boundary frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.factorized_chart_frontier import (  # noqa: E402
    evaluate_bank,
    render_bank,
)
from src.eval.global_frontier import sha256_file  # noqa: E402


CONFIG_SHA256 = "f71240a536e9eeb7a836a2a8024ec9eb1276cd35bfc954181dba4461a36733a6"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ao3_factorized_chart_boundary_frontier_v1.json"
        ),
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.expected_config_sha256 != CONFIG_SHA256
        or sha256_file(args.config) != CONFIG_SHA256
    ):
        raise ValueError("AO3 config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / str(config["output_root"])
    if args.render_dir is not None:
        render_dir = (
            args.render_dir
            if args.render_dir.is_absolute()
            else ROOT / args.render_dir
        )
        result = render_bank(root=ROOT, config=config, output_dir=render_dir)
        print(
            json.dumps(
                {
                    "manifest": str(result["manifest_path"]),
                    "sha256": result["manifest_sha256"],
                    "records": len(result["manifest"]["records"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = args.manifest or output_root / "render_pass1" / "manifest.json"
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    output = args.output or output_root / "automatic_report.json"
    if not output.is_absolute():
        output = ROOT / output
    result = evaluate_bank(root=ROOT, config=config, manifest_path=manifest)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "manifest_sha256": sha256_file(manifest),
        **result,
        "visual": {
            "status": (
                "required_not_yet_run"
                if result["shortlist"]
                else "not_run_automatic_gate_failed"
            )
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "report": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "automatic_survivors": result["automatic_survivors"],
                "shortlist": result["shortlist"],
                "automatic_decision": result["automatic_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
