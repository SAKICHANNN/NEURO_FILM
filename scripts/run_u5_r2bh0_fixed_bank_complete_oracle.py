#!/usr/bin/env python3
"""Render the frozen U5.R2BH0 five-arm bank and blind sheets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# LibRaw/X-Trans auto-bright and demosaic reductions are not byte-stable under
# unconstrained OpenMP scheduling. Freeze before any module can import rawpy.
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402,E501
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402,E501
from src.eval.fixed_bank_complete_oracle import (  # noqa: E402
    build_blind_round,
    run_render,
    validate_contract,
)
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)


CONFIG = ROOT / "configs/u5_r2bh0_fixed_bank_complete_oracle_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = run_render(
        root=ROOT,
        config=config,
        config_path=CONFIG,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        build_components=p8aq._build_components,
        patch_runtime=p8aw._patch_runtime,
        compile_artifact=compile_standalone_profile_artifact,
        create_runtime=create_opt_in_native_standard_runtime,
    )
    sheets = []
    if args.build_blind_sheets:
        if not report["blind_review_allowed"]:
            raise RuntimeError("automatic gate forbids blind sheets")
        validated = validate_contract(ROOT, config)
        for round_index in (1, 2, 3):
            sheets.append(
                build_blind_round(
                    root=ROOT,
                    render_dir=output_dir,
                    source_rows=validated["source_rows"],
                    eligible_ids=validated["eligible_ids"],
                    round_index=round_index,
                    output_dir=output_dir / "blind",
                )
            )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "source_count": report["source_count"],
                "row_count": len(report["rows"]),
                "stable_evidence_id": report["stable_evidence_id"],
                "blind_mapping_sha256": [
                    row["mapping_sha256"] for row in sheets
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
