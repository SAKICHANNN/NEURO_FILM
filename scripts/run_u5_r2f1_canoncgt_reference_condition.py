#!/usr/bin/env python3
"""Render and evaluate the frozen U5.R2F1 CanonCGT reference bank."""

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

from src.eval.canoncgt_reference_condition import (  # noqa: E402
    build_blind_sheets,
    evaluate_bank,
    render_bank,
)
from src.eval.global_frontier import sha256_file  # noqa: E402


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2f1_canoncgt_reference_condition_v1.json",
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--build-visual", action="store_true")
    args = parser.parse_args()
    args.config = _absolute(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / str(config["output_root"])
    if args.render_dir is not None:
        result = render_bank(
            root=ROOT,
            config=config,
            output_dir=_absolute(args.render_dir),
            device=str(args.device),
        )
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
    manifest = _absolute(
        args.manifest
        or output_root / "render_pass1" / "manifest.json"
    )
    output = _absolute(args.output or output_root / "automatic_report.json")
    result = evaluate_bank(root=ROOT, config=config, manifest_path=manifest)
    visual = {"status": "not_requested"}
    if args.build_visual and result["shortlist"]:
        visual = {
            "status": "pending_blind_review",
            **build_blind_sheets(
                root=ROOT,
                config=config,
                manifest_path=manifest,
                shortlist=result["shortlist"],
                output_dir=output.parent / "visual",
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
        "manifest_sha256": sha256_file(manifest),
        **result,
        "visual": visual,
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
