#!/usr/bin/env python3
"""Run the frozen BN7 triangular-transport versus AO6 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_triangular_logit_transport_visual_product_value import (  # noqa: E402
    build_blind_round,
)
from src.eval.fivek_triangular_logit_transport_vs_ao6 import (  # noqa: E402
    load_contract,
    run_incumbent_comparison,
)


CONFIG = ROOT / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    config = load_contract(CONFIG)
    result = run_incumbent_comparison(
        root=ROOT,
        config=config,
        config_path=CONFIG,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    sheets = []
    if args.build_blind_sheets:
        primary = tuple(config["blind_protocol"]["primary_pair"])
        for round_index in range(1, 4):
            sheets.append(
                build_blind_round(
                    root=ROOT,
                    render_dir=output_dir,
                    report=result["report"],
                    round_index=round_index,
                    output_dir=output_dir / "blind",
                    primary_arms=primary,
                )
            )
    print(
        json.dumps(
            {
                "automatic_pass": result["report"]["automatic_pass"],
                "gates": result["report"]["gates"],
                "report_sha256": result["report_sha256"],
                "stable_evidence_id": result["report"]["stable_evidence_id"],
                "blind_sheet_sets": len(sheets),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
