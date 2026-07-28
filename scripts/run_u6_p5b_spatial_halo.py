from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_spatial_halo import (
    evaluate_spatial_halo,
    load_contract,
    write_report,
)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
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
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    p5a_raw = args.p5a_contract.read_bytes()
    p5a_sha = hashlib.sha256(p5a_raw).hexdigest()
    if p5a_sha != contract["parents"]["p5a_contract_sha256"]:
        raise ValueError("P5A contract hash does not match frozen P5B parent")
    p5a_contract = json.loads(p5a_raw.decode("utf-8"))
    sensitometry_raw = args.sensitometry.read_bytes()
    sensitometry_sha = hashlib.sha256(sensitometry_raw).hexdigest()
    if sensitometry_sha != p5a_contract["parents"]["sensitometry_config_sha256"]:
        raise ValueError("sensitometry hash does not match frozen P5A parent")
    sensitometry = json.loads(sensitometry_raw.decode("utf-8"))
    report = evaluate_spatial_halo(
        contract,
        p5a_contract,
        sensitometry,
        diagnostic_path=args.diagnostic,
    )
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    print(f"diagnostic_sha256={report['diagnostic_sha256']}")


if __name__ == "__main__":
    main()
