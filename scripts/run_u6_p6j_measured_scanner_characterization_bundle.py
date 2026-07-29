from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.physical_measured_scanner_characterization_bundle import (
    compile_measured_scanner_characterization_bundle,
    write_bundle,
    write_report,
)


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
    root = Path(__file__).resolve().parents[1]
    bundle, report = compile_measured_scanner_characterization_bundle(
        root, root / args.contract
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
