"""Render the fixed AX13 cap-0.70 operator on the validation scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_domain_balanced_validation import (  # noqa: E402
    evaluate_domain_balanced_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ax15_filmmatch_cap070_validation_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "validation_scene_ax15",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    decision_raw = (
        ROOT / config["parent"]["dense_safety_decision"]
    ).read_bytes()
    if (
        hashlib.sha256(decision_raw).hexdigest()
        != config["parent"]["dense_safety_decision_sha256"]
    ):
        raise ValueError("AX14 dense-safety decision identity drift")
    if json.loads(decision_raw)["automatic_gate_passed"] is not True:
        raise ValueError("AX14 dense-safety gate must pass")
    parent_raw = (ROOT / config["parent"]["operator_report"]).read_bytes()
    if (
        hashlib.sha256(parent_raw).hexdigest()
        != config["parent"]["operator_report_sha256"]
    ):
        raise ValueError("AX13 operator report identity drift")
    parent = json.loads(parent_raw)
    if (
        parent["stable_evidence_id"]
        != config["parent"]["operator_stable_evidence_id"]
    ):
        raise ValueError("AX13 operator stable identity drift")
    report = evaluate_domain_balanced_validation(
        parent,
        config,
        root=ROOT / config["data_root"],
        output_dir=args.output_dir,
    )
    report_path = args.output_dir / "report.json"
    temporary = report_path.with_name("report.tmp.json")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "automatic_gate_passed": report["automatic_gate_passed"],
                "visual_review_opened": report["visual_review_opened"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
