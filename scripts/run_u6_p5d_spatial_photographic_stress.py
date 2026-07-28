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

from src.eval.physical_spatial_photographic_stress import (  # noqa: E402
    evaluate_photographic_stress,
    load_contract,
    write_report,
)


def _exact_json(path: Path, expected_sha256: str, name: str):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} hash does not match frozen parent")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5d_spatial_photographic_stress_v1.json",
    )
    parser.add_argument(
        "--p5c-decision",
        type=Path,
        default=ROOT / "configs" / "u6_p5c_bounded_adjacency_decision_v1.json",
    )
    parser.add_argument(
        "--p5c-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5c_bounded_adjacency_v1.json",
    )
    parser.add_argument(
        "--p5b-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5b_spatial_halo_audit_v1.json",
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
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    input_spec = contract["input"]
    manifest = _exact_json(
        ROOT / input_spec["manifest"],
        input_spec["manifest_sha256"],
        "photographic manifest",
    )
    _exact_json(
        ROOT / input_spec["preflight_report"],
        input_spec["preflight_report_sha256"],
        "photographic preflight",
    )
    p5c_decision = _exact_json(
        args.p5c_decision,
        contract["parents"]["p5c_decision_sha256"],
        "P5C decision",
    )
    p5c_contract = _exact_json(
        args.p5c_contract,
        p5c_decision["contract_sha256"],
        "P5C contract",
    )
    p5b_contract = _exact_json(
        args.p5b_contract,
        p5c_contract["parents"]["p5b_contract_sha256"],
        "P5B contract",
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
    report = evaluate_photographic_stress(
        contract,
        manifest,
        p5c_contract,
        p5a_contract,
        sensitometry,
        root=ROOT,
        contact_sheet_path=args.contact_sheet,
    )
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    print(f"contact_sheet_sha256={report['contact_sheet_sha256']}")


if __name__ == "__main__":
    main()
