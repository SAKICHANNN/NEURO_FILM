"""Run the frozen R1C3D affine spatial cross-fit absorption diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmstylesafe_r1c import load_r1c_contract  # noqa: E402
from src.eval.filmstylesafe_r1c3 import (  # noqa: E402
    load_r1c3_contract,
    run_conditioned_pilot,
    run_spatial_crossfit_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1c3d_spatial_crossfit_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "filmstylesafe" / "r1c3d" / "spatial_crossfit.json",
    )
    args = parser.parse_args()
    diagnostic = json.loads(args.contract.read_text(encoding="utf-8"))
    if diagnostic.get("contract_id") != "kmcfm.filmstylesafe-r1c3d.v1":
        raise ValueError("unexpected R1C3D contract id")
    r1c3 = load_r1c3_contract(ROOT / diagnostic["r1c3_contract"])
    baseline = load_r1c_contract(ROOT / r1c3["baseline_contract"])
    inventory = json.loads((ROOT / r1c3["inventory"]).read_text(encoding="utf-8"))
    parent = run_conditioned_pilot(
        inventory,
        root=ROOT,
        parent_sources=baseline["parent_sources"],
        fit=r1c3["fit"],
        score=r1c3["score"],
    )
    affine = next(row for row in parent["candidates"] if row["degree"] == 1)
    if affine["positives_detected"] != 2 or parent["decision"] != "weak_pass":
        raise ValueError("R1C3 affine weak-pass baseline did not reproduce")
    crossfit = run_spatial_crossfit_pilot(
        inventory,
        root=ROOT,
        parent_sources=baseline["parent_sources"],
        fit=r1c3["fit"],
        score=r1c3["score"],
        grid=diagnostic["grid"],
    )
    report = {
        "contract_id": diagnostic["contract_id"],
        "inventory_id": inventory["inventory_id"],
        "node": diagnostic["node"],
        "parent_affine_reproduction": affine,
        **crossfit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "positives_detected": report["positives_detected"],
                "zero_fpr_threshold": report["zero_fpr_threshold"],
                "hardneg_external_below_all_positives": report[
                    "hardneg_external_below_all_positives"
                ],
                "report": str(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
