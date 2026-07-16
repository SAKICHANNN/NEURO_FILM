"""Run the frozen RF1.4B1 Gold100 paired-transform consistency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.gold_transform_consistency import (  # noqa: E402
    GoldTransformConsistencyError,
    load_paired_frame_samples,
    run_whole_roll_evaluation,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_gold_transform_consistency_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf1_4b1" / "report.json",
    )
    args = parser.parse_args()
    config = _load(args.config)
    decision_path = ROOT / config["alignment_decision"]
    report_path = ROOT / config["alignment_report"]
    for path, expected in (
        (decision_path, config["alignment_decision_sha256"]),
        (report_path, config["alignment_report_sha256"]),
    ):
        if _sha(path) != expected:
            raise GoldTransformConsistencyError(f"pinned evidence hash mismatch: {path}")

    alignment_decision = _load(decision_path)
    alignment_report = _load(report_path)
    if alignment_decision["decision"] != "alignment_and_support_pass" or not alignment_report["passed"]:
        raise GoldTransformConsistencyError("RF1.4B0 alignment evidence did not pass")
    if alignment_decision["paired_frames"] != alignment_report["paired_frames"]:
        raise GoldTransformConsistencyError("RF1.4B0 paired-frame counts disagree")
    if alignment_decision["pairs_by_roll"] != alignment_report["pairs_by_roll"]:
        raise GoldTransformConsistencyError("RF1.4B0 roll support disagrees")

    frames = load_paired_frame_samples(
        download_root=ROOT / config["download_root"],
        pair_records=alignment_report["pair_records"],
        maximum_pixels_per_frame=int(config["sampling"]["maximum_pixels_per_frame"]),
    )
    if len(frames) != int(alignment_decision["paired_frames"]):
        raise GoldTransformConsistencyError("loaded frame count differs from frozen alignment evidence")
    evaluation = run_whole_roll_evaluation(frames, config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": _sha(args.config),
        "input_hashes_verified_before_image_decode": True,
        "working_space": config["working_space"],
        "fit_scope": "training_rolls_only",
        "test_target_fitting_forbidden_and_absent": True,
        **evaluation,
        "visual_status": (
            "pending_required_visual_adjudication"
            if evaluation["metric_passed"]
            else "not_run_metric_gate_failed"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(json.dumps({
        "report": str(args.output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decision": report["decision"],
        "metric_passed": report["metric_passed"],
        "rolls_improved_over_simple_affine": report["rolls_improved_over_simple_affine"],
        "rolls_beating_median_wrong_roll": report["rolls_beating_median_wrong_roll"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
