"""Download and hash-verify only the frozen four-stock BlueNeg manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.blueneg_download import (  # noqa: E402
    BlueNegDownloadConfig,
    download_blueneg_acquisition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        type=Path,
        default=ROOT / "configs" / "real_film_stock_pilot_acquisition_decision.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    root = (ROOT / decision["download_root"]).resolve()
    manifest = (ROOT / decision["acquisition_manifest"]).resolve()
    metadata_report = (ROOT / decision["metadata_report"]).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()

    def fetch(remote_path: str, force: bool) -> Path:
        return Path(hf_hub_download(
            repo_id=decision["repo_id"],
            repo_type="dataset",
            filename=remote_path,
            revision=decision["revision"],
            local_dir=root,
            force_download=force,
        )).resolve()

    report = download_blueneg_acquisition(
        BlueNegDownloadConfig(
            root=root,
            acquisition_manifest=manifest,
            acquisition_manifest_sha256=decision["acquisition_manifest_sha256"],
            metadata_report=metadata_report,
            metadata_report_sha256=decision["metadata_report_sha256"],
            output_report=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "download_report.json",
            repo_id=decision["repo_id"],
            revision=decision["revision"],
            expected_files=int(decision["expected_files"]),
            expected_bytes=int(decision["expected_bytes"]),
            workers=args.workers,
            software_commit=commit,
        ),
        fetch,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "file_inventory"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
