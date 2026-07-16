"""Download the frozen SF0.8 YFCC15M Parquet metadata subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_stock_source import (  # noqa: E402
    atomic_json,
    download_parquet_shards,
    fetch_parquet_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_stock_metadata_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    index_payload, rows = fetch_parquet_index(config)
    index_sha = atomic_json(ROOT / config["metadata_freeze"]["index_snapshot"], index_payload)
    manifest = download_parquet_shards(rows, config, ROOT / config["metadata_freeze"]["download_root"])
    manifest["index_snapshot_sha256"] = index_sha
    manifest_sha = atomic_json(ROOT / config["metadata_freeze"]["download_manifest"], manifest)
    print(json.dumps({"files": len(manifest["files"]), "bytes": manifest["total_bytes"], "manifest_sha256": manifest_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
