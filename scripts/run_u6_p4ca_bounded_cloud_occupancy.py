from __future__ import annotations

import json
from pathlib import Path

from src.eval.bounded_cloud_occupancy import (
    evaluate_bounded_cloud_occupancy,
    write_report,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/u6_p4ca_bounded_cloud_occupancy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    first = evaluate_bounded_cloud_occupancy(contract, root)
    first_sha = write_report(first, root / contract["outputs"]["report_a"])
    second = evaluate_bounded_cloud_occupancy(contract, root)
    second_sha = write_report(second, root / contract["outputs"]["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P4CA repeat drift")
    print(json.dumps({"report_sha256": first_sha, **first}, indent=2))


if __name__ == "__main__":
    main()
