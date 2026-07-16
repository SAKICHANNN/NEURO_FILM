"""Audit pixels and render contact sheets for the SF0.8B Velvia50 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import audit_download_manifest, render_contact_sheets  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_velvia50_pixel_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["download_manifest"]).read_text(encoding="utf-8"))
    report = audit_download_manifest(manifest, root=ROOT / config["download_root"], config=config)
    report["contact_sheets"] = render_contact_sheets(report["file_records"], root=ROOT / config["download_root"], output_dir=ROOT / config["contact_sheet_root"])
    digest = atomic_json(ROOT / config["audit_report"], report)
    print(json.dumps({"files": report["files"], "integrity_passed": report["integrity_passed"], "report_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
