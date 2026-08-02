"""Run U6.P8BR source conformance twice before any scan read."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.vfgs_frequency_shaping_approximation import (
    evaluate_vfgs_frequency_shaping_approximation,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p8br_vfgs_frequency_shaping_approximation_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / args.contract).read_text(encoding="utf-8"))
    first = evaluate_vfgs_frequency_shaping_approximation(contract, root)
    first_sha = write_report(first, root / contract["outputs"]["report_a"])
    second = evaluate_vfgs_frequency_shaping_approximation(contract, root)
    second_sha = write_report(second, root / contract["outputs"]["report_b"])
    if first != second or first_sha != second_sha:
        raise SystemExit("P8BR repeat drift")
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
