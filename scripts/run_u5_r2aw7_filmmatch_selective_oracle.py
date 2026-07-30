"""Run the post-AW6 sparse high-exposure Oracle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_condition_oracle import (  # noqa: E402
    evaluate_selective_high_exposure_oracle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aw7_filmmatch_selective_oracle_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "selective_oracle_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent"]["report"]
    parent_raw = parent_path.read_bytes()
    if (
        hashlib.sha256(parent_raw).hexdigest()
        != config["parent"]["report_sha256"]
        or json.loads(parent_raw)["stable_evidence_id"]
        != config["parent"]["stable_evidence_id"]
    ):
        raise ValueError("AW6 parent identity drift")
    report = evaluate_selective_high_exposure_oracle(
        json.loads(parent_raw), config
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
                "selective_oracle_gate_passed": report[
                    "selective_oracle_gate_passed"
                ],
                "luma_proxy_router_pilot_opened": report[
                    "luma_proxy_router_pilot_opened"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
