from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.ntire_night_metadata_preflight import run_preflight

DEFAULT_CONFIG = ROOT / "configs" / "sf3_a0v_ntire_night_metadata_preflight_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-member-order", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        args.config,
        reverse_member_order=args.reverse_member_order,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
