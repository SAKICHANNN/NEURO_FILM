#!/usr/bin/env python3
"""Run fixed AO6 plus linear optical-density grain value audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_density_grain_value import (
    build_blind_sheets,
    evaluate_report,
    render_bank,
)
from src.eval.ao6_procedural_filmfx_value import sha256_file

CONFIG = ROOT / "configs" / "u5_r2bc1_ao6_density_grain_value_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    report = render_bank(
        root=ROOT,
        config=config,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        config_sha256=sha256_file(CONFIG),
    )
    report["automatic"] = evaluate_report(config, report)
    if report["automatic"]["automatic_pass"]:
        blind = build_blind_sheets(
            root=ROOT,
            config=config,
            report=report,
            output_dir=output_dir / "blind",
        )
        report["blind"] = {
            "status": "built_after_automatic_pass",
            "sample_ids": blind["sample_ids"],
            "mapping_commitment_sha256": blind["mapping_commitment_sha256"],
        }
        (output_dir / "blind" / "mapping.json").write_text(
            json.dumps(blind, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        report["blind"] = {
            "status": "forbidden_by_automatic_failure",
            "mapping_commitment_sha256": None,
        }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "automatic": report["automatic"],
                "report_sha256": sha256_file(report_path),
                "mapping_commitment_sha256": report["blind"][
                    "mapping_commitment_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
