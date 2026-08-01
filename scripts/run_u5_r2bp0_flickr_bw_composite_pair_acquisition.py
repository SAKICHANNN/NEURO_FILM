#!/usr/bin/env python
"""Acquire and split the seven frozen BO9 B&W composite weak pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_bw_composite_pair_acquisition import acquire
from src.eval.flickr_single_author_pair_acquisition import atomic_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bp0_flickr_bw_composite_pair_acquisition_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent"]["report"]
    if sha256_file(parent_path) != config["parent"]["report_sha256"]:
        raise SystemExit("BO9 parent report hash drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest, report = acquire(parent, config, ROOT)
    manifest_sha = atomic_json(ROOT / config["acquisition"]["manifest"], manifest)
    report["manifest_sha256"] = manifest_sha
    report["config_sha256"] = sha256_file(args.config)
    report_sha = atomic_json(ROOT / config["acquisition"]["report"], report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "branch": report["branch"],
                "composites": report["metrics"]["composites"],
                "derived_pairs": report["metrics"]["derived_pairs"],
                "downloaded_bytes": report["metrics"]["downloaded_bytes"],
                "manifest_sha256": manifest_sha,
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
