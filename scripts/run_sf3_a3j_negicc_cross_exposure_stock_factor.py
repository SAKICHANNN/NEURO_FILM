from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.negicc_cross_exposure_factor import run_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_config = json.loads(
        (ROOT / config["source_lock"]["config"]).read_text(encoding="utf-8")
    )
    report = run_diagnostic(config, source_config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scientific_identity": report["scientific_identity"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_RETROSPECTIVE_STOCK_FACTOR_STABILITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
