"""Snapshot BlueNeg metadata/inventory and freeze a whole-roll pilot contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.blueneg import (  # noqa: E402
    BlueNegEvidenceConfig,
    build_blueneg_metadata_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_pilot.json",
    )
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
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    root = (ROOT / payload["root"]).resolve()
    output = (ROOT / payload["evidence_dir"]).resolve()
    snapshot_download(
        repo_id=payload["repo_id"],
        repo_type="dataset",
        revision=payload["revision"],
        allow_patterns=["LICENSE", "README.md", "meta.json", "transformations.pkl"],
        local_dir=root,
    )
    info = HfApi().dataset_info(
        payload["repo_id"],
        revision=payload["revision"],
        files_metadata=True,
    )
    if info.sha != payload["revision"]:
        raise ValueError("BlueNeg resolved revision mismatch")
    inventory = {
        "schema_version": 1,
        "repo_id": payload["repo_id"],
        "revision": info.sha,
        "files": [
            {
                "path": row.rfilename,
                "size": row.size,
                "sha256": row.lfs.sha256 if row.lfs else None,
                "blob_id": row.blob_id,
            }
            for row in info.siblings
            if row.size is not None
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "remote_inventory.json").write_bytes(
        (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    )
    split = payload["split"]
    result = build_blueneg_metadata_evidence(
        BlueNegEvidenceConfig(
            root=root,
            output_dir=output,
            repo_id=payload["repo_id"],
            revision=payload["revision"],
            metadata_sha256=payload["metadata_sha256"],
            expected=payload["expected"],
            split_seed=split["seed"],
            minimum_paired_frames=split["minimum_paired_frames_per_operator_roll"],
            confirm_if_at_least_four=split["confirmatory_rolls_per_film_if_at_least_four"],
            confirm_if_at_least_two=split["confirmatory_rolls_per_film_if_at_least_two"],
            required_credit=payload["acquisition"]["required_credit"],
            software_commit=_commit(),
        )
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
