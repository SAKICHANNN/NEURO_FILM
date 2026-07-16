"""Filter the full YFCC100M index for exact stock and shared-author support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_full_index import scan_full_index  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_full_index_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = scan_full_index(ROOT / config["download"]["destination"], config)
    digest = atomic_json(ROOT / config["scan"]["report"], report)
    print(json.dumps({"matches": len(report["matches"]), "any_gate_passed": report["any_shared_author_gate_passed"], "report_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
