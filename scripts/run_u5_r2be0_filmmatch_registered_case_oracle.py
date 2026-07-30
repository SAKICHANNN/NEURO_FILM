"""Run the frozen FilmMatch registered validation case-bank Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_registered_case_oracle import (  # noqa: E402
    evaluate_registered_case_oracle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2be0_filmmatch_registered_case_oracle_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "registered_case_oracle_be0/report.json",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    sample_config = json.loads(
        (ROOT / config["parents"]["sample_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = evaluate_registered_case_oracle(
        extract_pair_datasets(ROOT, sample_config),
        config,
        root=ROOT,
        software_commit=commit,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "automatic_passed": report["automatic_passed"],
                "retrieval_contract_opened": report[
                    "retrieval_contract_opened"
                ],
                "best_regime": report["aggregates"]["best_regime"],
                "best_exact_ev": report["aggregates"]["best_exact_ev"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
