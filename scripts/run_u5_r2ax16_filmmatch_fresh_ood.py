"""Run fresh nine-camera OOD validation for the fixed AX13 operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_fresh_ood import evaluate_fresh_ood  # noqa: E402


def _read_exact(path: Path, expected_sha256: str, label: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} identity drift")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ax16_filmmatch_fresh_ood_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "fresh_ood_ax16",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    decision = json.loads(
        _read_exact(
            ROOT / config["parent"]["validation_decision"],
            config["parent"]["validation_decision_sha256"],
            "AX15 decision",
        )
    )
    if decision["automatic_gate_passed"] is not True:
        raise ValueError("AX15 validation must pass")
    operator_raw = _read_exact(
        ROOT / config["parent"]["operator_report"],
        config["parent"]["operator_report_sha256"],
        "AX13 operator report",
    )
    operator_report = json.loads(operator_raw)
    if (
        operator_report["stable_evidence_id"]
        != config["parent"]["operator_stable_evidence_id"]
    ):
        raise ValueError("AX13 stable operator identity drift")
    source_decision = json.loads(
        _read_exact(
            ROOT / config["population"]["source_decision"],
            config["population"]["source_decision_sha256"],
            "P8BP source decision",
        )
    )
    if source_decision["result"]["decision"] != (
        "pass_fresh_population_and_open_fixed_arm_comparison"
    ):
        raise ValueError("P8BP source population is not eligible")
    manifest = json.loads(
        _read_exact(
            ROOT / config["population"]["manifest"],
            config["population"]["manifest_sha256"],
            "P8BP manifest",
        )
    )
    report = evaluate_fresh_ood(
        operator_report, manifest, config, root=ROOT, output_dir=args.output_dir
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
                "source_count": len(report["rows"]),
                "stable_evidence_id": report["stable_evidence_id"],
                "visual_review_opened": report["visual_review_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
