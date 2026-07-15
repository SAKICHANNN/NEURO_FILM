"""Download only the frozen BlueNeg pilot files and verify every payload."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_pilot.json",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_acquisition_decision.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    pilot = json.loads(args.pilot_config.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if pilot["repo_id"] != decision["repo_id"] or pilot["revision"] != decision["revision"]:
        raise ValueError("BlueNeg pilot/decision source mismatch")
    root = (ROOT / pilot["root"]).resolve()
    evidence = (ROOT / pilot["evidence_dir"]).resolve()

    def fetch(remote_path: str, force: bool) -> Path:
        return Path(
            hf_hub_download(
                repo_id=pilot["repo_id"],
                repo_type="dataset",
                filename=remote_path,
                revision=pilot["revision"],
                local_dir=root,
                force_download=force,
            )
        ).resolve()

    report = download_blueneg_acquisition(
        BlueNegDownloadConfig(
            root=root,
            acquisition_manifest=evidence / "acquisition.json",
            acquisition_manifest_sha256=decision["acquisition_manifest_sha256"],
            metadata_report=evidence / "report.json",
            metadata_report_sha256=decision["metadata_report_sha256"],
            output_report=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "download_report.json",
            repo_id=pilot["repo_id"],
            revision=pilot["revision"],
            expected_files=decision["acquisition_files"],
            expected_bytes=decision["acquisition_bytes"],
            workers=args.workers,
            software_commit=_commit(),
        ),
        fetch,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "file_inventory"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
