from __future__ import annotations

import json
from pathlib import Path

from src.eval.finite_support_thomas_filter import (
    evaluate_finite_support_thomas_filter,
    write_report,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/u6_p4bu_finite_support_thomas_filter_v1.json").read_text(
            encoding="utf-8"
        )
    )
    first = evaluate_finite_support_thomas_filter(contract, root)
    first_sha = write_report(first, root / contract["outputs"]["report_a"])
    second = evaluate_finite_support_thomas_filter(contract, root)
    second_sha = write_report(second, root / contract["outputs"]["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P4BU repeat drift")
    print(
        json.dumps(
            {
                "automatic_pass": first["automatic_pass"],
                "decision": first["decision"],
                "report_sha256": first_sha,
                "stable_evidence_id": first["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
