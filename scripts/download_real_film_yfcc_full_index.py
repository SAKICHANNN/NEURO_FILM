"""Resume and verify the frozen SF1.1 full YFCC100M metadata index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_full_index import download_full_index  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_full_index_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    destination = ROOT / config["download"]["destination"]
    manifest = download_full_index(config, destination, progress_path=ROOT / config["download"]["progress"])
    digest = atomic_json(ROOT / config["download"]["manifest"], manifest)
    print(json.dumps({"bytes": manifest["bytes"], "sha256": manifest["sha256"], "manifest_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
