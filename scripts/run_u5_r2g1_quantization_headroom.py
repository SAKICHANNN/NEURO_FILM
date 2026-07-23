#!/usr/bin/env python3
"""Run the frozen U5.R2G1 fixed quantization-headroom frontier."""

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
from src.eval.quantization_headroom import evaluate_bank, render_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2g1_quantization_headroom_v1.json",
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / str(config["output_root"])
    if args.render_dir:
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
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    output = args.output or output_root / "automatic_report.json"
    output = output if output.is_absolute() else ROOT / output
    result = evaluate_bank(root=ROOT, config=config, manifest_path=manifest)
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
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        **result,
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
                "automatic_decision": result["automatic_decision"],
                "automatic_survivors": result["automatic_survivors"],
                "shortlist": result["shortlist"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
