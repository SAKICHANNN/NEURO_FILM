#!/usr/bin/env python3
"""Compare color-engine evaluation summaries against the safe_lab champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLOR_STYLES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare color engine summary.json files.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--label", default="challenger")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--styles", default=",".join(COLOR_STYLES))
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["styles"]


def better_or_close(challenger: float, baseline: float, tolerance: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return challenger >= baseline - tolerance
    return challenger <= baseline + tolerance


def compare_style(baseline: dict, challenger: dict) -> dict:
    chroma_gain = (challenger["after_chroma_mean"] - baseline["after_chroma_mean"]) / max(
        baseline["after_chroma_mean"], 1e-6
    )
    return {
        "new_clip_pass": challenger["new_clip_image_count"] == 0 and challenger["hard_bound_image_count"] == 0,
        "l_ssim_pass": challenger["l_ssim_min"] >= max(0.995, baseline["l_ssim_min"] - 0.0005),
        "neutral_pass": challenger["neutral_contaminated_percent_max"] <= baseline["neutral_contaminated_percent_max"] + 0.10,
        "hf_pass": better_or_close(
            challenger["high_frequency_delta_p95_mean"],
            baseline["high_frequency_delta_p95_mean"],
            baseline["high_frequency_delta_p95_mean"] * 0.05,
            higher_is_better=False,
        ),
        "chroma_gain_percent": chroma_gain * 100.0,
        "richness_pass": chroma_gain >= 0.03,
        "baseline": baseline,
        "challenger": challenger,
    }


def main() -> int:
    args = parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    baseline = load_summary(args.baseline)
    challenger = load_summary(args.challenger)
    comparisons = {}
    promote_count = 0
    for style in styles:
        row = compare_style(baseline[style], challenger[style])
        row["promote_candidate"] = all(
            [row["new_clip_pass"], row["l_ssim_pass"], row["neutral_pass"], row["hf_pass"], row["richness_pass"]]
        )
        promote_count += int(row["promote_candidate"])
        comparisons[style] = row
    result = {
        "label": args.label,
        "baseline": str(args.baseline),
        "challenger": str(args.challenger),
        "style_count": len(styles),
        "promote_count": promote_count,
        "all_promote": promote_count == len(styles),
        "styles": comparisons,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
