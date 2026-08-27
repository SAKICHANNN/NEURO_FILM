"""Run SF3.A3M metadata lock, bounded acquisition or formal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.internet_archive_fixed_camera_stock import (
    acquire,
    build_source_lock,
    canonical_json,
    evaluate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf3_a3m_ia_ektar_fixed_camera_identifiability_v1.json",
    )
    parser.add_argument("--mode", choices=("lock", "acquire", "formal"), required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "lock":
        result = build_source_lock(config, reverse=args.reverse)
        destination = ROOT / str(config["source"]["source_lock"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json(result))
    elif args.mode == "acquire":
        result = acquire(config, ROOT, reverse=args.reverse)
    else:
        result = evaluate(config, ROOT, reverse=args.reverse)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_json(result))
    print(json.dumps({key: result.get(key) for key in ("experiment_id", "decision", "metadata_requests", "download_bytes", "rows", "item_groups") if key in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
