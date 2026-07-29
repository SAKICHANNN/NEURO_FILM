#!/usr/bin/env python
"""Run the frozen U6.P4N bounded explicit LOD policy audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.eval.physical_bounded_lod_policy_fit import (  # noqa: E402
    evaluate_bounded_lod_policy,
    load_contract,
)


CONFIG_SHA256 = "dfad5f3cde53c70043ae9d9c64e80d7ec8fe4c73bca17709ed08a4d096640d1e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4n_bounded_lod_policy_fit_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4n_bounded_lod_policy_fit_v1/report.json",
    )
    args = parser.parse_args()
    contract, parent, manifest, dataset_root = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    report = evaluate_bounded_lod_policy(
        contract, parent, manifest, dataset_root
    )
    report["config_sha256"] = CONFIG_SHA256
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(
            {
                "schema": report["schema"],
                "node": report["node"],
                "selected_threshold": report["selected_threshold"],
                "development_grid": report["development_grid"],
                "global_p4h_development": report[
                    "global_p4h_development"
                ],
                "selected_policy": report["selected_policy"],
                "checks": report["checks"],
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "config_sha256": CONFIG_SHA256,
            }
        )
    ).hexdigest()
    _atomic_write(args.output, _canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
