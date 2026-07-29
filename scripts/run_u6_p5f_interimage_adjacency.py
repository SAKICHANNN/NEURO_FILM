#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_interimage_adjacency import (  # noqa: E402
    evaluate_interimage_adjacency,
    load_contract,
    write_report,
)


def _verify_parent(path: Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} hash does not match frozen parent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5f_interimage_adjacency_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    parent_paths = {
        "p5c_contract_sha256": ROOT
        / "configs"
        / "u6_p5c_bounded_adjacency_v1.json",
        "p5c_decision_sha256": ROOT
        / "configs"
        / "u6_p5c_bounded_adjacency_decision_v1.json",
        "p5d_contract_sha256": ROOT
        / "configs"
        / "u6_p5d_spatial_photographic_stress_v1.json",
        "p5d_decision_sha256": ROOT
        / "configs"
        / "u6_p5d_spatial_photographic_stress_decision_v1.json",
    }
    for key, path in parent_paths.items():
        _verify_parent(path, contract["parents"][key], key)
    report = evaluate_interimage_adjacency(contract)
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    minimum_gain = min(
        row["gain_fraction"]
        for row in report["metrics"]["single_channel_edges"]
    )
    print(f"minimum_opponent_gain={minimum_gain}")


if __name__ == "__main__":
    main()
