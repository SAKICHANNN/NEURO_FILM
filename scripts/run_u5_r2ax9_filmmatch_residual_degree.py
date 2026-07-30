"""Run degree-3 versus degree-4 safe residual FilmMatch capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_domain_balanced_capacity import (  # noqa: E402
    evaluate_domain_balanced_capacity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ax9_filmmatch_residual_degree_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "residual_degree_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_raw = (ROOT / config["parent"]["report"]).read_bytes()
    if (
        hashlib.sha256(parent_raw).hexdigest()
        != config["parent"]["report_sha256"]
        or json.loads(parent_raw)["stable_evidence_id"]
        != config["parent"]["stable_evidence_id"]
    ):
        raise ValueError("AX8 parent identity drift")
    sample_config = json.loads(
        (ROOT / config["parent"]["sample_config"]).read_text(encoding="utf-8")
    )
    report = evaluate_domain_balanced_capacity(
        extract_pair_datasets(ROOT, sample_config), config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "development_champion": report["development_champion"],
                "development_readout_passed": report[
                    "development_readout_passed"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
