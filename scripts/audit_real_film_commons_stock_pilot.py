"""Run the offline SF0.5 pixel/source audit and contact-sheet renderer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import atomic_json, audit_download_manifest, render_contact_sheets, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_film_commons_stock_pixel_pilot_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["download_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = ROOT / config["download_root"]
    report = audit_download_manifest(manifest, root=root, config=config)
    report["software_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    report["config_sha256"] = sha256_file(args.config)
    report["selection_manifest_sha256"] = config["selection_manifest_sha256"]
    report["download_manifest_sha256"] = sha256_file(manifest_path)
    report["contact_sheets"] = render_contact_sheets(report["file_records"], root=root, output_dir=ROOT / config["contact_sheet_root"])
    output = ROOT / config["audit_report"]
    digest = atomic_json(output, report)
    print(json.dumps({"report": str(output), "sha256": digest, "files": report["files"], "bytes": report["bytes"], "integrity_passed": report["integrity_passed"], "stock_source_gates": report["stock_source_gates"], "contact_sheets": len(report["contact_sheets"])}, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
