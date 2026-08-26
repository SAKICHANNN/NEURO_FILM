"""Run the frozen metadata-only YFCC Ektar digital-companion audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_ektar_companion import audit, canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a3f_yfcc_ektar_digital_companion_metadata_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = audit(ROOT, args.contract.resolve(), reverse=args.reverse)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("decision", "target_rows", "target_uids", "eligible_pair_count", "eligible_uid_count", "stable_evidence_id")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
