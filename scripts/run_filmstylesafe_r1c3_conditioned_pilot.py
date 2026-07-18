"""Run the frozen FilmStyleSafe R1C3 conditioned style-control A0 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmstylesafe_r1c import load_r1c_contract, run_a0_metric_pilot  # noqa: E402
from src.eval.filmstylesafe_r1c3 import (  # noqa: E402
    FilmStyleSafeR1CError,
    load_r1c3_contract,
    run_conditioned_pilot,
)


def _assert_parent_reproduction(report: dict, decision: dict) -> dict:
    observed = {
        "conventional_best_sensitivity_all_nonsevere": report[
            "best_conventional_zero_fpr_all_nonsevere"
        ]["sensitivity_at_zero_fpr"],
        "conventional_best_sensitivity_hardneg_external": report[
            "best_conventional_zero_fpr_hardneg_external"
        ]["sensitivity_at_zero_fpr"],
        "scis_v0_sensitivity_all_nonsevere": report[
            "scis_v0_zero_fpr_all_nonsevere"
        ]["sensitivity_at_zero_fpr"],
        "scis_v0_sensitivity_hardneg_external": report[
            "scis_v0_zero_fpr_hardneg_external"
        ]["sensitivity_at_zero_fpr"],
        "scis_v0_1_sensitivity_all_nonsevere": report[
            "scis_v0_1_zero_fpr_all_nonsevere"
        ]["sensitivity_at_zero_fpr"],
        "scis_v0_1_sensitivity_hardneg_external": report[
            "scis_v0_1_zero_fpr_hardneg_external"
        ]["sensitivity_at_zero_fpr"],
    }
    expected = {
        "conventional_best_sensitivity_all_nonsevere": 1.0 / 3.0,
        "conventional_best_sensitivity_hardneg_external": decision[
            "conventional_best_sensitivity_hardneg_external"
        ],
        "scis_v0_sensitivity_all_nonsevere": 1.0 / 3.0,
        "scis_v0_sensitivity_hardneg_external": decision[
            "scis_v0_sensitivity_hardneg_external"
        ],
        "scis_v0_1_sensitivity_all_nonsevere": decision[
            "scis_v0_1_sensitivity_all_nonsevere"
        ],
        "scis_v0_1_sensitivity_hardneg_external": decision[
            "scis_v0_1_sensitivity_hardneg_external"
        ],
    }
    if observed != expected:
        raise FilmStyleSafeR1CError(
            f"R1C2 baseline reproduction failed: observed={observed}, expected={expected}"
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1c3_conditioned_style_control_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "filmstylesafe" / "r1c3" / "conditioned_pilot.json",
    )
    args = parser.parse_args()
    contract = load_r1c3_contract(args.contract)
    baseline_contract = load_r1c_contract(ROOT / contract["baseline_contract"])
    inventory = json.loads((ROOT / contract["inventory"]).read_text(encoding="utf-8"))
    parent_decision = json.loads(
        (ROOT / contract["parent_decision"]).read_text(encoding="utf-8")
    )
    baseline = run_a0_metric_pilot(
        inventory,
        root=ROOT,
        parent_sources=baseline_contract["parent_sources"],
        scis_params=baseline_contract["scis_v0"],
    )
    reproduction = _assert_parent_reproduction(baseline, parent_decision)
    conditioned = run_conditioned_pilot(
        inventory,
        root=ROOT,
        parent_sources=baseline_contract["parent_sources"],
        fit=contract["fit"],
        score=contract["score"],
    )
    report = {
        "contract_id": contract["contract_id"],
        "inventory_id": inventory["inventory_id"],
        "node": contract["node"],
        "baseline_reproduction": reproduction,
        **conditioned,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_candidate": report["selected_candidate"],
                "candidates": report["candidates"],
                "report": str(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
