"""Render the fit-forbidden FilmMatch real validation scene."""

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
from src.eval.filmmatch_validation_scene import (  # noqa: E402
    evaluate_validation_scene,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aw9_filmmatch_validation_scene_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/validation_scene",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_config = json.loads(
        (ROOT / config["parent"]["sample_config"]).read_text(encoding="utf-8")
    )
    parent_raw = (ROOT / config["parent"]["router_report"]).read_bytes()
    if (
        hashlib.sha256(parent_raw).hexdigest()
        != config["parent"]["router_report_sha256"]
        or json.loads(parent_raw)["stable_evidence_id"]
        != config["parent"]["router_stable_evidence_id"]
    ):
        raise ValueError("AW8 router parent identity drift")
    report = evaluate_validation_scene(
        extract_pair_datasets(ROOT, sample_config),
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
                "route": report["router"]["route"],
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
