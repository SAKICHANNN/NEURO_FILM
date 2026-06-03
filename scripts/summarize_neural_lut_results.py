#!/usr/bin/env python3
"""Summarize Neural Film LUT V2 eval summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Neural Film LUT V2 runs.")
    parser.add_argument("--eval-root", type=Path, default=ROOT / "outputs" / "eval" / "neural_film_lut_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval" / "neural_film_lut_v2" / "candidate_summary_v1.json")
    parser.add_argument(
        "--patterns",
        default="scheme_a_distilled_v1_s*,scheme_b_nilut_distilled_v1_s*,scheme_c_context4d_distilled_v1_s*",
    )
    return parser.parse_args()


def summarize_run(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    styles = data["styles"]
    style_names = list(styles)
    lssim = [styles[name]["l_ssim_mean"] for name in style_names]
    chroma = [styles[name]["after_chroma_mean"] for name in style_names]
    gates = [styles[name].get("gate_mean", 1.0) for name in style_names]
    clips = sum(int(styles[name]["new_clipped_pixel_count_total"]) for name in style_names)
    mins = [styles[name]["after_min_min"] for name in style_names]
    maxs = [styles[name]["after_max_max"] for name in style_names]
    neutral = [styles[name]["neutral_contaminated_percent_max"] for name in style_names]
    return {
        "run": summary_path.parent.name,
        "engine": data.get("engine", ""),
        "strength": data.get("strength"),
        "style_count": len(style_names),
        "new_clipped_pixel_count_total": clips,
        "after_min_min": min(mins),
        "after_max_max": max(maxs),
        "l_ssim_mean": mean(lssim),
        "l_ssim_min_across_styles": min(lssim),
        "style_chroma_range": max(chroma) - min(chroma),
        "style_chroma_std_proxy": (sum((value - mean(chroma)) ** 2 for value in chroma) / len(chroma)) ** 0.5,
        "neutral_contaminated_percent_max": max(neutral),
        "gate_mean": mean(gates),
        "contact_sheets": {
            style: str(summary_path.parent / style / "contact_sheet.png")
            for style in style_names
        },
    }


def main() -> int:
    args = parse_args()
    rows = []
    for pattern in [item.strip() for item in args.patterns.split(",") if item.strip()]:
        for summary_path in sorted(args.eval_root.glob(f"{pattern}/summary.json")):
            rows.append(summarize_run(summary_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in rows[0] if key != "contact_sheets"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})
    print(args.output)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
