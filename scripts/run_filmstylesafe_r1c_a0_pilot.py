"""Run the FilmStyleSafe R1C A0 conventional-metric vs SCIS v0 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmstylesafe_r1c import (  # noqa: E402
    FilmStyleSafeR1CError,
    load_r1c_contract,
    run_a0_metric_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1c_contract_v1.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1b6_a0_inventory_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "filmstylesafe" / "r1c" / "a0_metric_pilot_report.json",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Optional longest-side downsample for faster development runs (0=full).",
    )
    args = parser.parse_args()
    contract = load_r1c_contract(args.contract)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = run_a0_metric_pilot(
        inventory,
        root=ROOT,
        parent_sources=contract["parent_sources"],
        scis_params=contract["scis_v0"],
        max_side=(args.max_side or None),
    )
    report = {
        "contract_id": contract["contract_id"],
        "inventory_id": inventory.get("inventory_id"),
        "node": "ULT > U5.R1 > U5.R1C",
        **result,
    }
    # Drop bulky nested arrays from stdout summary; keep full report on disk.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "n_positives": report["n_positives"],
        "n_negatives": report["n_negatives"],
        "conventional_metric_gap_at_zero_fpr": report["conventional_metric_gap_at_zero_fpr"],
        "conventional_metric_gap_vs_hardneg_external": report[
            "conventional_metric_gap_vs_hardneg_external"
        ],
        "best_conventional_zero_fpr_all_nonsevere": report[
            "best_conventional_zero_fpr_all_nonsevere"
        ],
        "best_conventional_zero_fpr_hardneg_external": report[
            "best_conventional_zero_fpr_hardneg_external"
        ],
        "scis_v0_zero_fpr_all_nonsevere": report["scis_v0_zero_fpr_all_nonsevere"],
        "scis_v0_zero_fpr_hardneg_external": report["scis_v0_zero_fpr_hardneg_external"],
        "scis_v0_1_zero_fpr_all_nonsevere": report["scis_v0_1_zero_fpr_all_nonsevere"],
        "scis_v0_1_zero_fpr_hardneg_external": report["scis_v0_1_zero_fpr_hardneg_external"],
        "scis_v0_perfect_sensitivity_at_zero_fpr": report[
            "scis_v0_perfect_sensitivity_at_zero_fpr"
        ],
        "scis_v0_perfect_vs_hardneg_external": report["scis_v0_perfect_vs_hardneg_external"],
        "scis_v0_1_perfect_sensitivity_at_zero_fpr": report[
            "scis_v0_1_perfect_sensitivity_at_zero_fpr"
        ],
        "scis_v0_1_perfect_vs_hardneg_external": report["scis_v0_1_perfect_vs_hardneg_external"],
        "hardneg_scis_below_all_positives": report["hardneg_scis_below_all_positives"],
        "hardneg_scis_v0_1_below_all_positives": report["hardneg_scis_v0_1_below_all_positives"],
        "report_path": str(args.report.as_posix()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FilmStyleSafeR1CError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
