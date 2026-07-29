#!/usr/bin/env python
"""Run the frozen U6.P4J adaptive exact-area fallback audit."""

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
from src.eval.physical_adaptive_exact_area_fallback import (  # noqa: E402
    evaluate_adaptive_exact_area_fallback,
    load_contract,
)


CONFIG_SHA256 = "fe49b43076caf17fb7cb6d01d4cc5d0d708122c886aa384609eb017ad711f140"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p4j_adaptive_exact_area_fallback_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4j_adaptive_exact_area_fallback_v1/report.json",
    )
    args = parser.parse_args()
    contract, parent, p4i = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    report = evaluate_adaptive_exact_area_fallback(
        contract, parent, p4i
    )
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
