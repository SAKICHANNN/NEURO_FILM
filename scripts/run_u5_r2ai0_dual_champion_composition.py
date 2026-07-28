#!/usr/bin/env python
"""Render and evaluate the frozen U5.R2AI0 composition bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.eval.dual_champion_composition import (
    build_blind_sheets,
    evaluate_bank,
    render_bank,
)


ROOT = Path(__file__).resolve().parents[1]


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_json(path: Path, payload: dict) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ai0_dual_champion_global_composition_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["software_commit"] = _commit()
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    rendered = render_bank(root=ROOT, config=config, output_dir=output_dir)
    report = evaluate_bank(
        root=ROOT,
        config=config,
        manifest_path=rendered["manifest_path"],
    )
    report.update(
        {
            "software_commit": config["software_commit"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "render_manifest_sha256": rendered["manifest_sha256"],
        }
    )
    report_path = output_dir / "automatic_report.json"
    report_sha = _write_json(report_path, report)
    sheets = None
    if args.build_blind_sheets and report["automatic_survivors"]:
        sheets = build_blind_sheets(
            root=ROOT,
            config=config,
            manifest_path=rendered["manifest_path"],
            survivors=report["automatic_survivors"],
            output_dir=output_dir / "blind",
        )
    print(
        json.dumps(
            {
                "automatic_decision": report["automatic_decision"],
                "automatic_survivors": report["automatic_survivors"],
                "manifest_sha256": rendered["manifest_sha256"],
                "report_sha256": report_sha,
                "rounds": (
                    [str(path) for path in sheets["rounds"]] if sheets else []
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
