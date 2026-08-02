from __future__ import annotations

import json
from pathlib import Path

from src.eval.clustered_nps_canonical_field import (
    evaluate_clustered_nps_canonical_field,
    write_report,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "configs/u6_p4bt_clustered_nps_canonical_field_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    first = evaluate_clustered_nps_canonical_field(contract, root)
    first_sha = write_report(first, root / contract["outputs"]["report_a"])
    second = evaluate_clustered_nps_canonical_field(contract, root)
    second_sha = write_report(second, root / contract["outputs"]["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P4BT repeat drift")
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
