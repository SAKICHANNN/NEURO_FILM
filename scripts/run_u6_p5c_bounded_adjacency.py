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

from src.eval.physical_bounded_adjacency import (  # noqa: E402
    evaluate_bounded_adjacency,
    load_contract,
    write_report,
)


def _exact_json(path: Path, expected_sha256: str, name: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} hash does not match frozen parent")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5c_bounded_adjacency_v1.json",
    )
    parser.add_argument(
        "--p5b-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5b_spatial_halo_audit_v1.json",
    )
    parser.add_argument(
        "--p5b-decision",
        type=Path,
        default=ROOT / "configs" / "u6_p5b_spatial_halo_decision_v1.json",
    )
    parser.add_argument(
        "--p5a-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5a_spatial_response_primitives_v1.json",
    )
    parser.add_argument(
        "--sensitometry",
        type=Path,
        default=ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    p5b_contract = _exact_json(
        args.p5b_contract,
        contract["parents"]["p5b_contract_sha256"],
        "P5B contract",
    )
    _exact_json(
        args.p5b_decision,
        contract["parents"]["p5b_decision_sha256"],
        "P5B decision",
    )
    p5a_contract = _exact_json(
        args.p5a_contract,
        p5b_contract["parents"]["p5a_contract_sha256"],
        "P5A contract",
    )
    sensitometry = _exact_json(
        args.sensitometry,
        p5a_contract["parents"]["sensitometry_config_sha256"],
        "sensitometry",
    )
    report = evaluate_bounded_adjacency(
        contract,
        p5b_contract,
        p5a_contract,
        sensitometry,
        diagnostic_path=args.diagnostic,
    )
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    print(
        "diagnostic_sha256="
        f"{report['p5b_halo_metrics']['diagnostic_sha256']}"
    )


if __name__ == "__main__":
    main()
