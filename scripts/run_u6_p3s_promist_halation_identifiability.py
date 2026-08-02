"""Run the frozen U6.P3S lens-diffusion/film-halation discriminator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.promist_halation_identifiability import evaluate, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p3s_promist_halation_identifiability_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(ROOT, args.contract)
    digest = write_report(report, args.output)
    print(json.dumps({"report_sha256": digest, "passed": report["passed"], "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
