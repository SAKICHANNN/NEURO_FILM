#!/usr/bin/env python3
"""List configured evaluation source buckets without touching image assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation source bucket configuration.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "eval_buckets.yaml")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--require-existing", action="store_true", help="Fail if referenced local manifests are missing.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not config.get("ignored_image_policy"):
        errors.append("ignored_image_policy is required")
    if not isinstance(config.get("benchmark_sets"), list) or not config["benchmark_sets"]:
        errors.append("benchmark_sets must be a non-empty list")
    if not isinstance(config.get("buckets"), dict) or not config["buckets"]:
        errors.append("buckets must be a non-empty mapping")

    known_buckets = set(config.get("buckets", {}))
    for index, item in enumerate(config.get("benchmark_sets", [])):
        for field in ("id", "status", "source_type", "manifest_path", "images_root", "buckets"):
            if field not in item:
                errors.append(f"benchmark_sets[{index}] missing {field}")
        for bucket in item.get("buckets", []):
            if bucket not in known_buckets:
                errors.append(f"benchmark_sets[{index}] references unknown bucket {bucket!r}")
    return errors


def summarize(config: dict[str, Any], require_existing: bool) -> dict[str, Any]:
    sets = []
    missing_required: list[str] = []
    for item in config["benchmark_sets"]:
        manifest_path = resolve_repo_path(item["manifest_path"])
        images_root = resolve_repo_path(item["images_root"])
        manifest_exists = manifest_path.exists()
        images_root_exists = images_root.exists()
        if require_existing and item.get("status") != "manual" and not manifest_exists:
            missing_required.append(str(manifest_path))
        sets.append(
            {
                "id": item["id"],
                "status": item["status"],
                "source_type": item["source_type"],
                "manifest_path": str(manifest_path),
                "manifest_exists": manifest_exists,
                "images_root": str(images_root),
                "images_root_exists": images_root_exists,
                "bucket_count": len(item.get("buckets", [])),
                "buckets": item.get("buckets", []),
            }
        )

    return {
        "schema_version": config["schema_version"],
        "benchmark_set_count": len(sets),
        "bucket_count": len(config["buckets"]),
        "sets": sets,
        "missing_required_manifests": missing_required,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.manifest)
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = summarize(config, args.require_existing)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"schema_version={summary['schema_version']}")
        print(f"benchmark_sets={summary['benchmark_set_count']} buckets={summary['bucket_count']}")
        for item in summary["sets"]:
            exists = "yes" if item["manifest_exists"] else "no"
            print(f"- {item['id']} [{item['status']}] manifest_exists={exists} buckets={','.join(item['buckets'])}")
        if summary["missing_required_manifests"]:
            print("missing_required_manifests:")
            for path in summary["missing_required_manifests"]:
                print(f"- {path}")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
