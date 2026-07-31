"""Run the frozen BL5 strict-interior sigmoid experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_strict_interior_sigmoid import (  # noqa: E402
    evaluate_strict_interior_sigmoid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bl5_filmmatch_strict_interior_sigmoid_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = config["parent"]
    bl1_raw = (ROOT / parent["bl1_report"]).read_bytes()
    if hashlib.sha256(bl1_raw).hexdigest() != parent["bl1_report_sha256"]:
        raise ValueError("BL1 report identity drift")
    bl1 = json.loads(bl1_raw)
    if bl1["stable_evidence_id"] != parent["bl1_stable_evidence_id"]:
        raise ValueError("BL1 stable evidence drift")
    bl4_raw = (ROOT / parent["bl4_decision"]).read_bytes()
    if hashlib.sha256(bl4_raw).hexdigest() != parent["bl4_decision_sha256"]:
        raise ValueError("BL4 decision identity drift")
    if json.loads(bl4_raw)["decision"]["status"] != parent["required_bl4_status"]:
        raise ValueError("BL4 decision status drift")
    sample_config_raw = (ROOT / parent["sample_config"]).read_bytes()
    if (
        hashlib.sha256(sample_config_raw).hexdigest()
        != parent["sample_config_sha256"]
    ):
        raise ValueError("sample config identity drift")
    sample_config = json.loads(sample_config_raw)
    datasets = extract_pair_datasets(ROOT, sample_config)
    report = evaluate_strict_interior_sigmoid(datasets, config)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "automatic_gate_passed": report["automatic_gate_passed"],
                "branch": report["branch"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
