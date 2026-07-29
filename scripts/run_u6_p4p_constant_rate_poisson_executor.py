#!/usr/bin/env python
"""Run the frozen U6.P4P constant-rate Poisson executor audit."""

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
from src.eval.physical_constant_rate_poisson_executor import (  # noqa: E402
    evaluate_constant_rate_executor,
    load_contract,
)


CONFIG_SHA256 = "bdfc4423afd226250672f901f85e6f7a484c8a386e5c863f078e6e8d1d1d07d1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p4p_constant_rate_poisson_executor_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4p_constant_rate_poisson_executor_v1/report.json",
    )
    args = parser.parse_args()
    contract = load_contract(ROOT, args.config, args.expected_config_sha256)
    report = evaluate_constant_rate_executor(contract)
    report["config_sha256"] = CONFIG_SHA256
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(
            {
                "schema": report["schema"],
                "node": report["node"],
                "exactness_rows": report["exactness_rows"],
                "failure_parity": report["failure_parity"],
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
