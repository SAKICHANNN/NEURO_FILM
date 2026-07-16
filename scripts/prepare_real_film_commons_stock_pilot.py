"""Freeze the deterministic SF0.5 Commons pixel selection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import atomic_json, build_selection_manifest, merge_metadata_snapshots, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_film_commons_stock_pixel_pilot_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if "metadata_snapshots" in config:
        snapshots = []
        for item in config["metadata_snapshots"]:
            path = ROOT / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"metadata snapshot hash mismatch: {path}")
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        snapshot = merge_metadata_snapshots(
            snapshots, allowed_stock_ids=config["allowed_stock_ids"]
        )
        merged_path = ROOT / config["merged_metadata_snapshot"]
        merged_hash = atomic_json(merged_path, snapshot)
        config = {**config, "metadata_snapshot_sha256": merged_hash}
    else:
        snapshot_path = ROOT / config["metadata_snapshot"]
        if sha256_file(snapshot_path) != config["metadata_snapshot_sha256"]:
            raise ValueError("metadata snapshot hash mismatch")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    manifest = build_selection_manifest(snapshot, config)
    output = ROOT / config["selection_manifest"]
    digest = atomic_json(output, manifest)
    print(json.dumps({"manifest": str(output), "sha256": digest, "metadata_snapshot_sha256": manifest["metadata_snapshot_sha256"], "selected_files": manifest["selected_files"], "selected_by_stock": manifest["selected_by_stock"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
