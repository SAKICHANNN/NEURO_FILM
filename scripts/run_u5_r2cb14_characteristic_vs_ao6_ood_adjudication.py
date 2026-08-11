#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.characteristic_vs_ao6_ood_adjudication import adjudicate_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudicator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / "outputs/experiments/u5_r2cb14_characteristic_vs_ao6_ood_v1"
    result = adjudicate_files(
        config_path=ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json",
        observations_path=ROOT
        / "configs/u5_r2cb14_characteristic_vs_ao6_ood_observations_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2cb14_characteristic_vs_ao6_ood_mapping_receipt_v1.json",
        report_paths=[base / "report_b.json", base / "report_c.json"],
        render_dir=base / "run_b",
        adjudicator_software_commit=args.adjudicator_commit,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"pass={result['pass']}")
    print(f"status={result['status']}")
    print(f"stable_evidence_id={result['stable_evidence_id']}")


if __name__ == "__main__":
    main()
