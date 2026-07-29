#!/usr/bin/env python3
"""Render one exact U6.P8BP fixed-arm comparison run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402,E501
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402,E501
from src.eval.fresh_native_standard_confirmation import (  # noqa: E402
    build_blind_sheet,
    run_comparison,
    validate_preflight,
)


CONFIG = (
    ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads(
        (
            ROOT
            / "configs/u6_p8bb_native_standard_working_image_resources_v1.json"
        ).read_text(encoding="utf-8")
    )
    p8aw._patch_runtime()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = run_comparison(
        root=ROOT,
        config=config,
        output_dir=output_dir,
        build_components=p8aq._build_components,
        build_config=base,
    )
    if args.build_blind_sheets:
        if not report["blind_review_allowed"]:
            raise RuntimeError("automatic gate forbids blind sheets")
        manifest = validate_preflight(ROOT, config)
        for round_index in (1, 2, 3):
            build_blind_sheet(
                root=ROOT,
                render_dir=output_dir,
                manifest=manifest,
                round_index=round_index,
                output_path=output_dir / f"blind_round_{round_index}.png",
                mapping_path=(
                    output_dir / f"blind_round_{round_index}_mapping.json"
                ),
            )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "source_count": report["source_count"],
                "row_count": len(report["rows"]),
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
