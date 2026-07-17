"""Fetch the frozen SF2.1A Openverse metadata snapshot without pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.openverse_stock_source import fetch_snapshot  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "real_film_openverse_shared_creator_v1.json"
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    snapshot = fetch_snapshot(config)
    snapshot["provenance"] = {
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
    }
    digest = atomic_json(ROOT / config["snapshot"], snapshot)
    print(json.dumps({"snapshot": config["snapshot"], "sha256": digest, "requests": snapshot["request_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
