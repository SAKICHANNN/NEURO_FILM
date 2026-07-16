"""Filter the full YFCC100M index for exact stock and shared-author support."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_full_index import scan_full_index, validate_download_manifest  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_full_index_v1.json")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = ROOT / config["download"]["destination"]
    manifest_payload = (ROOT / config["download"]["manifest"]).read_bytes()
    manifest = json.loads(manifest_payload)
    source_evidence = validate_download_manifest(manifest, config, source_path)
    report = scan_full_index(source_path, config)
    report["source_evidence"] = {
        **source_evidence,
        "download_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    report_path = args.report or (ROOT / config["scan"]["report_a"])
    digest = atomic_json(report_path, report)
    print(json.dumps({"matches": len(report["matches"]), "any_gate_passed": report["any_shared_author_gate_passed"], "report_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
