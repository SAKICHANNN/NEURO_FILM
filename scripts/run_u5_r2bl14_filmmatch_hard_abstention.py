"""Run the preregistered BL14 hard-abstention experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_hard_abstention import evaluate_hard_abstention  # noqa: E402


def _load_bound_json(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"bound parent hash drift: {path}")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2bl14_filmmatch_hard_abstention_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = config["parent"]
    parent_config = _load_bound_json(ROOT / parent["contract"], parent["contract_sha256"])
    decision = _load_bound_json(ROOT / parent["decision"], parent["decision_sha256"])
    if decision.get("decision") != parent["required_decision"] or decision.get("automatic_result", {}).get("branch") != parent["required_branch"]:
        raise ValueError("BL0 parent decision drift")
    sample_parent = parent_config["parent"]
    sample_config = _load_bound_json(ROOT / sample_parent["sample_config"], sample_parent["sample_config_sha256"])
    report = evaluate_hard_abstention(extract_pair_datasets(ROOT, sample_config), config, parent_config)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"automatic_gate_passed": report["automatic_gate_passed"], "branch": report["branch"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
