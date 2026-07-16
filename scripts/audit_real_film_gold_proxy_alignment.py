"""Run the frozen RF1.4B0 Gold100 official-alignment preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.gold_proxy_alignment import (  # noqa: E402
    audit_gold_proxy_alignment,
    safe_load_transformations,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_gold_proxy_alignment_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf1_4b0" / "report.json",
    )
    args = parser.parse_args()
    config = _load(args.config)
    for path_key, hash_key in (
        ("preview_decision", "preview_decision_sha256"),
        ("integrity_report", "integrity_report_sha256"),
        ("transformations", "transformations_sha256"),
    ):
        path = ROOT / config[path_key]
        if _sha(path) != config[hash_key]:
            raise ValueError(f"pinned evidence hash mismatch: {path}")
    transformations = safe_load_transformations(ROOT / config["transformations"])
    integrity = _load(ROOT / config["integrity_report"])
    audit = audit_gold_proxy_alignment(
        download_root=ROOT / config["download_root"],
        file_records=integrity["file_records"],
        transformations=transformations,
        film_stock_id=config["film_stock_id"],
        gates=config["gates"],
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "software_commit": commit,
        "config_sha256": _sha(args.config),
        "input_hashes_verified_before_pickle_or_image_decode": True,
        "restricted_pickle_loader_used": True,
        **audit,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(json.dumps({
        "report": str(args.output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "passed": report["passed"],
        "paired_frames": report["paired_frames"],
        "paired_rolls": report["paired_rolls"],
        "pairs_by_roll": report["pairs_by_roll"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
