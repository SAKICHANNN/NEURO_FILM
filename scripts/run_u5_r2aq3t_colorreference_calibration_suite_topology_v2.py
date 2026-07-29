#!/usr/bin/env python
"""Run the separately versioned AQ3T source-only topology repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from scripts.run_u5_r2aq3t_colorreference_calibration_suite_topology import (  # noqa: E402
    _require_clean_tracked_worktree,
    run_audit,
)


CONFIG_SHA256 = "9af6185d76467757323ed8cc900dc900ad17ba4d20747535d981a96070647855"
EXPERIMENT_ID = "u5.r2aq3t-colorreference-calibration-suite-topology-v2"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3T-v2 config hash mismatch")
    config = json.loads(raw)
    if (
        config["schema_version"] != 2
        or config["experiment_id"] != EXPERIMENT_ID
        or config["fit_allowed"]
        or config["target_fields_allowed"]
        or config["aq2_reopened"]
        or config["operator_promotion_allowed"]
    ):
        raise ValueError("AQ3T-v2 frozen contract mismatch")
    repair = config["supersedes_source_design_only"]
    if (
        repair["report_sha256"]
        != "f07c9bb6f09d6b5357e86cb1c28de04488c925866d48f6127c5f7d00bad993fa"
    ):
        raise ValueError("AQ3T-v2 predecessor binding mismatch")
    return config


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report = run_audit(config, config_sha256=CONFIG_SHA256)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "fold_counts": report["fold_counts"],
                "fold_metrics": report["fold_metrics"],
                "report_sha256": _sha256(report_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq3t_colorreference_calibration_suite_topology_v2.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3t_colorreference_calibration_suite_topology_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
