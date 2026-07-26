#!/usr/bin/env python
"""Build integrity-checked U5.R2L0 visual-review pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_strength_oracle_review import (  # noqa: E402
    build_review_records,
    evidence_sha256,
    write_review_pages,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2l0_density_strength_oracle_v1.json",
    )
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2l0_density_strength_oracle_v1"
            / "selected_manifest_pass1.json"
        ),
    )
    parser.add_argument(
        "--e1-config",
        type=Path,
        default=ROOT / "configs/u5_r2e1_density_witness_frontier_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2l0_density_strength_oracle_v1"
            / "visual"
        ),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    selected_path = (
        args.selected_manifest
        if args.selected_manifest.is_absolute()
        else ROOT / args.selected_manifest
    )
    e1_path = (
        args.e1_config if args.e1_config.is_absolute() else ROOT / args.e1_config
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = build_review_records(
        root=ROOT,
        oracle_config=config,
        selected_manifest_path=selected_path,
        e1_config_path=e1_path,
    )
    evidence = write_review_pages(records, output_dir)
    encoded = (
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    evidence_path = output_dir / "evidence.json"
    temporary = evidence_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(evidence_path)
    print(
        json.dumps(
            {
                "evidence": str(evidence_path),
                "sha256": evidence_sha256(evidence),
                "overview_pages": len(evidence["overview_pages"]),
                "one_to_one_crop_pages": len(evidence["one_to_one_crop_pages"]),
                "reviewed_sample_count": evidence["reviewed_sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
