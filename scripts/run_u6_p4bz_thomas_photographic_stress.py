"""Run the frozen U6.P4BZ photographic Thomas stress."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.thomas_photographic_stress import (
    evaluate_thomas_photographic_stress,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4bz_thomas_photographic_stress_v1.json",
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    outputs = contract["outputs"]
    report_a = evaluate_thomas_photographic_stress(
        contract, ROOT, contact_sheet_path=ROOT / outputs["contact_sheet"]
    )
    report_b = evaluate_thomas_photographic_stress(contract, ROOT)
    sha_a = write_report(report_a, ROOT / outputs["report_a"])
    sha_b = write_report(report_b, ROOT / outputs["report_b"])
    if report_a != report_b or sha_a != sha_b:
        raise RuntimeError("P4BZ reports are not byte-identical")
    print(json.dumps({"report_sha256": sha_a, **report_a}, sort_keys=True))
    return 0 if report_a["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
