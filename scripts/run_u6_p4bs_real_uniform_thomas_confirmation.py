"""Run U6.P4BS twice against exact local uniform-scan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.real_uniform_thomas_confirmation import (
    evaluate_real_uniform_thomas_confirmation,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p4bs_real_uniform_thomas_confirmation_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / args.contract).read_text(encoding="utf-8"))
    outputs = contract["outputs"]
    first = evaluate_real_uniform_thomas_confirmation(contract, root)
    first_sha = write_report(first, root / outputs["report_a"])
    second = evaluate_real_uniform_thomas_confirmation(contract, root)
    second_sha = write_report(second, root / outputs["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P4BS repeat drift")
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
