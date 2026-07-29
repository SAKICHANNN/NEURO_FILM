from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_scanner_characterization_bundle import (  # noqa: E402
    compile_measured_scanner_characterization_bundle,
    write_bundle,
    write_report,
)

CONTRACT_SHA256 = "98ddd7ab0c543614720210af61a11bcda028e5525947cca851e4061536a7ac3c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6j_measured_scanner_characterization_bundle_v1.json"
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    contract_path = root / args.contract
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise SystemExit("U6.P6J contract hash mismatch")
    bundle, report = compile_measured_scanner_characterization_bundle(
        root, contract_path
    )
    bundle_sha = write_bundle(bundle, root / args.bundle)
    report_sha = write_report(report, root / args.report)
    print(f"bundle_sha256={bundle_sha}")
    print(f"bundle_id={bundle.bundle_id}")
    print(f"report_sha256={report_sha}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={report['automatic_pass']}")


if __name__ == "__main__":
    main()
