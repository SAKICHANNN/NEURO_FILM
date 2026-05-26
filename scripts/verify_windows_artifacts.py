#!/usr/bin/env python3
"""Verify files transferred from the Windows training machine.

Run from the repository root after copying artifacts listed in
docs/WINDOWS_ARTIFACT_MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Windows artifact transfer manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs" / "WINDOWS_ARTIFACT_MANIFEST.json",
        help="Manifest generated on the Windows training machine.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root containing transferred files.")
    parser.add_argument("--hash-all", action="store_true", help="Hash entries that were recorded size-only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root.resolve()
    failures: list[str] = []

    for item in manifest["items"]:
        path = root / item["path"]
        if not path.exists():
            failures.append(f"missing: {item['path']}")
            continue
        size = path.stat().st_size
        if size != item["size_bytes"]:
            failures.append(f"size mismatch: {item['path']} expected {item['size_bytes']} got {size}")
            continue
        expected_hash = item.get("sha256")
        if expected_hash or args.hash_all:
            actual_hash = sha256(path)
            if expected_hash and actual_hash != expected_hash:
                failures.append(f"sha256 mismatch: {item['path']}")

    if failures:
        print("Artifact verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Artifact verification OK: {len(manifest['items'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
