#!/usr/bin/env python
"""Run the frozen U6.P4E density-conditioned physical-LOD audit."""

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
from src.eval.physical_density_conditioned_lod import (  # noqa: E402
    evaluate_density_conditioned_lod,
    load_contract,
)


CONFIG_SHA256 = "93f1ea71bf9a6ee9db873b76dff3f7ab99d0f75b0698c6f47af5c7540ba02488"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4e_density_conditioned_lod_audit_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4e_density_conditioned_lod_audit_v1/report.json",
    )
    args = parser.parse_args()
    contract, parent = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    report = evaluate_density_conditioned_lod(contract, parent)
    report["config_sha256"] = CONFIG_SHA256
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(
            {
                key: value
                for key, value in report.items()
                if key not in {"software_commit", "stable_evidence_id"}
            }
        )
    ).hexdigest()
    _atomic_write(args.output, _canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
