"""Download only the frozen SF1.3A shared-author derivative candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_shared_author_pixels import (  # noqa: E402
    evaluate_shared_author_pixel_download,
    prepare_shared_author_pixel_candidates,
)
from src.real_film.yfcc_stock_pilot import download_live_pixels  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def _load_hashed(path: Path, expected: str) -> dict:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"input hash drift: {path}")
    return json.loads(payload)


def _stock_config(config: dict, stock: str, candidate_count: int, remaining_bytes: int) -> dict:
    return {
        "pilot_id": f"{config['pilot_id']}-{stock}",
        "target_stock_id": stock,
        "user_agent": config["user_agent"],
        "selection": {"maximum_retained_files": candidate_count},
        "live_rights": config["live_rights"],
        "download_limits": {
            "maximum_bytes_total": remaining_bytes,
            "maximum_bytes_per_file": config["download_limits"]["maximum_bytes_per_file"],
            "timeout_seconds": config["download_limits"]["timeout_seconds"],
            "request_interval_seconds": config["download_limits"]["request_interval_seconds"],
            "original_download_forbidden": True,
        },
        "pixel_audit": {"minimum_short_dimension": config["pixel_gate"]["minimum_short_dimension"]},
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_yfcc_shared_author_pixel_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rights_config = _load_hashed(
        ROOT / config["source_rights_config"], str(config["source_rights_config_sha256"])
    )
    metadata = _load_hashed(
        ROOT / config["input_metadata_report"], str(config["input_metadata_report_sha256"])
    )
    rights = _load_hashed(ROOT / config["input_rights_report"], str(config["input_rights_report_sha256"]))
    sf11_decision = _load_hashed(
        ROOT / rights_config["input_decision"], str(rights_config["input_decision_sha256"])
    )
    candidates = prepare_shared_author_pixel_candidates(
        metadata, sf11_decision, rights, rights_config, config
    )
    results: dict[str, dict] = {}
    total_limit = int(config["download_limits"]["maximum_bytes_total"])
    used_bytes = 0
    root = ROOT / config["download_root"]
    for stock in config["stock_ids"]:
        remaining = max(total_limit - used_bytes, 0)
        stock_config = _stock_config(config, str(stock), len(candidates[str(stock)]), remaining)
        result = download_live_pixels(candidates[str(stock)], stock_config, root=root)
        results[str(stock)] = result
        used_bytes += int(result["bytes"])
    combined = evaluate_shared_author_pixel_download(results, config)
    digest = atomic_json(ROOT / config["download_manifest"], combined)
    print(
        json.dumps(
            {
                "files": combined["files"],
                "bytes": combined["bytes"],
                "stock_counts": combined["stock_counts"],
                "usable_shared_authors": combined["usable_shared_author_count"],
                "acquisition_gate_passed": combined["acquisition_gate_passed"],
                "manifest_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
