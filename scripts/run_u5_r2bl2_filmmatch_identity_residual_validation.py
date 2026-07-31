"""Render the fixed BL1 operator on the fit-forbidden real scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_identity_residual_validation import (  # noqa: E402
    evaluate_identity_residual_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bl2_filmmatch_identity_residual_validation_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "identity_residual_validation_bl2",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    parent_config_raw = (ROOT / config["parent"]["config"]).read_bytes()
    if hashlib.sha256(parent_config_raw).hexdigest() != config["parent"][
        "config_sha256"
    ]:
        raise ValueError("BL1 config identity drift")
    parent_raw = (ROOT / config["parent"]["report"]).read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != config["parent"][
        "report_sha256"
    ]:
        raise ValueError("BL1 report identity drift")
    parent = json.loads(parent_raw)
    if parent["stable_evidence_id"] != config["parent"]["stable_evidence_id"]:
        raise ValueError("BL1 stable evidence identity drift")
    if parent["automatic_gate_passed"] is not True:
        raise ValueError("BL1 automatic gate must pass")

    report = evaluate_identity_residual_validation(
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
