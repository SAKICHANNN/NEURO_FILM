"""Run U6.P4CF twice against the frozen uniform-scan cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_uniform_higher_order_structure import (
    evaluate_real_uniform_higher_order_structure,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p4cf_real_uniform_higher_order_structure_v1.json"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / args.contract).read_text(encoding="utf-8"))
    first = evaluate_real_uniform_higher_order_structure(contract, root)
    first_sha = write_report(first, root / contract["outputs"]["report_a"])
    second = evaluate_real_uniform_higher_order_structure(contract, root)
    second_sha = write_report(second, root / contract["outputs"]["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P4CF repeat drift")
    print(
        json.dumps(
            {
                "automatic_pass": first["automatic_pass"],
                "decision": first["decision"],
                "report_sha256": first_sha,
                "stable_evidence_id": first["stable_evidence_id"],
                "development_summary": first["development_summary"],
                "confirmation_summary": first["confirmation_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
