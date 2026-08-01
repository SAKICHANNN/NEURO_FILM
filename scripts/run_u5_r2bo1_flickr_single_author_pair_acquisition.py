"""Acquire and integrity-audit the frozen BO0 Flickr weak-pair derivatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_single_author_pair_acquisition import (  # noqa: E402
    acquire,
    atomic_json,
    audit,
    selected_rows,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bo1_flickr_single_author_pair_acquisition_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent"]["report"]
    if sha256_file(parent_path) != config["parent"]["report_sha256"]:
        raise SystemExit("parent BO0 report hash drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    rows = selected_rows(parent, config)
    root = ROOT / config["acquisition"]["download_root"]
    manifest_path = ROOT / config["acquisition"]["manifest"]
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    manifest = acquire(
        rows,
        root=root,
        config=config,
        prior_manifest=prior,
        checkpoint_path=manifest_path,
    )
    manifest_sha = atomic_json(manifest_path, manifest)
    report = audit(manifest, root=root, config=config)
    report["manifest_sha256"] = manifest_sha
    report["config_sha256"] = sha256_file(args.config)
    report_sha = atomic_json(ROOT / config["acquisition"]["report"], report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "files": report["metrics"]["files"],
                "bytes": report["metrics"]["bytes"],
                "report_sha256": report_sha,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
