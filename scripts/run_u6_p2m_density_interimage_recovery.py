#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_interimage_recovery import (  # noqa: E402
    evaluate_interimage_recovery,
    load_contract,
    write_report,
)


def _verify(path: Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} hash does not match frozen parent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p2m_density_interimage_recovery_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    parents = {
        "p2a_contract_sha256": ROOT
        / "configs"
        / "u6_p2a_exposure_development_interpretation_contract_v1.json",
        "p2a_decision_sha256": ROOT
        / "configs"
        / "u6_p2a_exposure_development_interpretation_decision_v1.json",
    }
    for key, path in parents.items():
        _verify(path, contract["parents"][key], key)
    report = evaluate_interimage_recovery(contract)
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    for witness in report["witnesses"]:
        metrics = witness["metrics"]
        print(
            witness["id"],
            f"rmse={metrics['candidate_confirmation_rmse_density']:.8f}",
            f"gain_independent={metrics['gain_over_independent_fraction']:.6f}",
            f"gain_affine={metrics['gain_over_affine_fraction']:.6f}",
        )


if __name__ == "__main__":
    main()

