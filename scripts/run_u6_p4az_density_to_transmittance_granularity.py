from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.density_to_transmittance_granularity import (
    compile_and_evaluate,
    load_contract,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/u6_p4az_density_to_transmittance_granularity_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/experiments/u6_p4az_density_to_transmittance_granularity_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    bundle, report = compile_and_evaluate(contract, ROOT)
    print(f"bundle_sha256={write_json(bundle, args.output_dir / 'bundle.json')}")
    print(f"report_sha256={write_json(report, args.output_dir / 'report.json')}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"decision={report['decision']}")


if __name__ == "__main__":
    main()
