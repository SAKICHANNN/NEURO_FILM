#!/usr/bin/env python
"""Generate and audit the frozen U6.P4M synthetic reference dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.eval.physical_reference_profile_dataset import (  # noqa: E402
    evaluate_reference_datasets,
    generate_reference_dataset,
    load_contract,
)


CONFIG_SHA256 = "4f106f552fd1b6b0e760e76ebc4ab712df6901d2687fab9852b3b70d5db1db27"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4m_reference_profile_dataset_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4m_reference_profile_dataset_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4m_reference_profile_dataset_v1/report.json",
    )
    args = parser.parse_args()
    contract, parent = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    token = f"{os.getpid()}"
    first_root = args.output_root / f".owned_generation_a_{token}"
    second_root = args.output_root / f".owned_generation_b_{token}"
    retained = args.output_root / "dataset"
    for path in (first_root, second_root, retained):
        if path.exists():
            raise FileExistsError(f"refusing to replace U6.P4M output: {path}")
    generations = []
    residue = -1
    try:
        generations.append(
            generate_reference_dataset(contract, parent, first_root)
        )
        generations.append(
            generate_reference_dataset(contract, parent, second_root)
        )
        preliminary = evaluate_reference_datasets(
            contract,
            generations,
            owned_temp_residue_count=0,
        )
        if preliminary["automatic_pass"]:
            first_root.replace(retained)
            shutil.rmtree(second_root)
        else:
            shutil.rmtree(first_root)
            shutil.rmtree(second_root)
        residue = int(first_root.exists()) + int(second_root.exists())
        report = evaluate_reference_datasets(
            contract,
            generations,
            owned_temp_residue_count=residue,
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
                    "record_counts": report["record_counts"],
                    "inventory_sha256": report["inventory_sha256"],
                    "checks": report["checks"],
                    "automatic_pass": report["automatic_pass"],
                    "decision": report["decision"],
                    "config_sha256": CONFIG_SHA256,
                }
            )
        ).hexdigest()
        report["owned_temp_residue_count"] = residue
        _atomic_write(args.report, _canonical_json(report))
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        for path in (first_root, second_root):
            if path.exists():
                shutil.rmtree(path)


if __name__ == "__main__":
    main()
