"""Download the frozen SF0.5 Commons derivative selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import atomic_json, download_selected_rows, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_film_commons_stock_pixel_pilot_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection_path = ROOT / config["selection_manifest"]
    expected_selection_hash = config.get("selection_manifest_sha256")
    if not expected_selection_hash:
        raise ValueError("selection_manifest_sha256 must be frozen before download")
    if sha256_file(selection_path) != expected_selection_hash:
        raise ValueError("selection manifest hash mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output = ROOT / config["download_manifest"]
    prior = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    manifest = download_selected_rows(
        selection["rows"],
        root=ROOT / config["download_root"],
        config=config,
        prior_manifest=prior,
        retries=int(config["download_limits"]["request_retries"]),
        request_interval_seconds=float(config["download_limits"]["request_interval_seconds"]),
        checkpoint_path=output,
    )
    digest = atomic_json(output, manifest)
    print(json.dumps({"manifest": str(output), "sha256": digest, "files": manifest["files"], "bytes": manifest["bytes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
