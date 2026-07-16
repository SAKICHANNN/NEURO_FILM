"""Verify live rights and download the bounded SF0.8B Velvia50 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_stock_pilot import (  # noqa: E402
    balanced_candidate_order,
    download_live_pixels,
    eligible_rows,
)
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_velvia50_pixel_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report_path = ROOT / config["metadata_report"]
    payload = report_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config["metadata_report_sha256"]:
        raise SystemExit("metadata report hash drift")
    report = json.loads(payload)
    candidates = balanced_candidate_order(eligible_rows(report, config), config)
    manifest_path = ROOT / config["download_manifest"]
    manifest = download_live_pixels(candidates, config, root=ROOT / config["download_root"], checkpoint_path=manifest_path)
    digest = atomic_json(manifest_path, manifest)
    print(json.dumps({"attempts": len(manifest["attempts"]), "files": manifest["files"], "bytes": manifest["bytes"], "manifest_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
