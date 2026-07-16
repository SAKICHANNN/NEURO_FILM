"""Audit a frozen SF0.4 Commons metadata snapshot without network or pixels."""

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

from src.real_film.commons_stock_source import audit_snapshot  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "real_film_commons_stock_source_audit_v1.json",
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _load(args.config)
    snapshot_path = args.snapshot or ROOT / config["snapshot_output"]
    output = args.output or ROOT / config["report_output"]
    result = audit_snapshot(_load(snapshot_path), config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "software_commit": commit,
        "config_sha256": _sha(args.config),
        "snapshot_sha256": _sha(snapshot_path),
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    print(json.dumps({
        "report": str(output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decision": report["decision"],
        "exact_stock_metadata_passes": report["exact_stock_metadata_passes"],
        "conditional_pixel_pilot_allowed": report["conditional_pixel_pilot_allowed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
