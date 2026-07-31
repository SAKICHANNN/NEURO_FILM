"""Run the preregistered FilmMatch source-conditioned operator experiment."""

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
from src.eval.filmmatch_source_conditioned_operator import (  # noqa: E402
    evaluate_source_conditioned_operator,
)


def _load_bound_json(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"bound parent hash drift: {path}")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bl0_filmmatch_source_conditioned_operator_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "source_conditioned_bl0/report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = config["parent"]
    sample_config = _load_bound_json(
        ROOT / parent["sample_config"], parent["sample_config_sha256"]
    )
    _load_bound_json(
        ROOT / parent["capacity_report"], parent["capacity_report_sha256"]
    )
    _load_bound_json(
        ROOT / parent["validation_decision"],
        parent["validation_decision_sha256"],
    )
    report = evaluate_source_conditioned_operator(
        extract_pair_datasets(ROOT, sample_config), config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
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
