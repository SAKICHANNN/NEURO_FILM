#!/usr/bin/env python
"""Render and evaluate frozen AO7 fresh residual confirmation."""

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

from src.eval.b0_real_film_residual_fresh_confirmation import (
    build_blind_sheets,
    evaluate_confirmation,
    render_confirmation,
)


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
        default=(
            ROOT
            / "configs/u5_r2ao7_b0_real_film_residual_fresh_confirmation_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes.decode("utf-8"))
    software_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    rendered = render_confirmation(
        root=ROOT,
        config=config,
        output_dir=output_dir,
        software_commit=software_commit,
        config_sha256=config_sha256,
    )
    report = evaluate_confirmation(
        root=ROOT,
        config=config,
        manifest_path=rendered["manifest_path"],
        config_sha256=config_sha256,
    )
    report_sha256 = _write_json(output_dir / "automatic_report.json", report)
    sheets = None
    if args.build_blind_sheets and report["automatic_pass"]:
        sheets = build_blind_sheets(
            root=ROOT,
            config=config,
            manifest_path=rendered["manifest_path"],
            config_sha256=config_sha256,
            output_dir=output_dir / "blind",
        )
    print(
        json.dumps(
            {
                "automatic_decision": report["automatic_decision"],
                "automatic_pass": report["automatic_pass"],
                "manifest_sha256": rendered["manifest_sha256"],
                "report_sha256": report_sha256,
                "rounds": (
                    [str(path) for path in sheets["rounds"]]
                    if sheets is not None
                    else []
                ),
                "round_sha256": (
                    sheets["round_sha256"] if sheets is not None else []
                ),
                "mapping_sha256": (
                    sheets["mapping_sha256"]
                    if sheets is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
