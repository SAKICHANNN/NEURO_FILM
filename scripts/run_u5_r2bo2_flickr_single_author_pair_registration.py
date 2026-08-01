"""Run frozen BO2 same-scene registration feasibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_single_author_pair_acquisition import atomic_json, sha256_file  # noqa: E402
from src.eval.flickr_single_author_pair_registration import evaluate, render_contact_sheets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bo2_flickr_single_author_pair_registration_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(ROOT, config)
    report["config_sha256"] = sha256_file(args.config)
    if report["automatic_pass"]:
        report["contact_sheets"] = render_contact_sheets(ROOT, report, config)
    digest = atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "metrics": report["metrics"],
                "report_sha256": digest,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
