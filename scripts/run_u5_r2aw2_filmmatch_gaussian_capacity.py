"""Run the frozen bounded local FilmMatch Gaussian capacity audit."""

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
from src.eval.filmmatch_gaussian_residual_capacity import (  # noqa: E402
    evaluate_gaussian_capacity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aw2_filmmatch_gaussian_capacity_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "gaussian_capacity_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_config_path = ROOT / config["parent"]["config"]
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_result_path = ROOT / config["parent_capacity_result"]["path"]
    parent_result_raw = parent_result_path.read_bytes()
    if (
        hashlib.sha256(parent_result_raw).hexdigest()
        != config["parent_capacity_result"]["sha256"]
        or json.loads(parent_result_raw)["stable_evidence_id"]
        != config["parent_capacity_result"]["stable_evidence_id"]
    ):
        raise ValueError("AW1 capacity parent identity drift")
    report = evaluate_gaussian_capacity(
        extract_pair_datasets(ROOT, parent_config), config
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
                "selected": report["selection"]["selected"],
                "development_champion": report["development_champion"],
                "validation_opened": report["validation_opened"],
                "promotion_opened": report["promotion_opened"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
