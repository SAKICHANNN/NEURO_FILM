"""Run the prospective SF1.0A YFCC same-source support scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_stock_source import atomic_json, scan_stock_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_matched_stock_metadata_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = ROOT / config["metadata_freeze"]["download_root"]
    paths = [root / name for name in config["metadata_freeze"]["expected_filenames"]]
    report = scan_stock_candidates(paths, config)
    digest = atomic_json(ROOT / config["metadata_freeze"]["audit_report"], report)
    print(json.dumps({"passing_stocks": report["passing_stocks"], "matches": len(report["matches"]), "report_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
