#!/usr/bin/env python3
"""Build deterministic CycloneDX/SPDX SBOMs for an installed product runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_runtime_sbom import (
    build_sbom_documents,
    collect_installed_inventory,
    inventory_digest,
    publish_sbom_documents,
)

DEFAULT_CONFIG = ROOT / "configs/u8_2a_private_product_runtime_sbom_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    config = json.loads(arguments.config.read_text("utf-8"))
    receipt_path = ROOT / config["input"]["receipt"]
    receipt, receipt_sha256, first_inventory = collect_installed_inventory(
        receipt_path,
        expected_sha256=config["input"]["receipt_sha256"],
    )
    _, second_sha256, second_inventory = collect_installed_inventory(
        receipt_path,
        expected_sha256=config["input"]["receipt_sha256"],
    )
    if receipt_sha256 != second_sha256 or first_inventory != second_inventory:
        raise RuntimeError("fresh installed-runtime inventories differ")
    if receipt["source_commit"] != config["input"]["source_commit"]:
        raise RuntimeError("installed source commit mismatch")
    if receipt["requirements"]["sha256"] != config["input"]["requirements_sha256"]:
        raise RuntimeError("installed requirements identity mismatch")
    cdx, spdx = build_sbom_documents(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        inventory=first_inventory,
        creation_time=config["canonical_creation_time"],
    )
    output_directory = arguments.output_directory or ROOT / config["output"][
        "directory"
    ]
    paths = publish_sbom_documents(
        output_directory=output_directory,
        cyclonedx_name=config["output"]["cyclonedx"],
        spdx_name=config["output"]["spdx"],
        cyclonedx_bytes=cdx,
        spdx_bytes=spdx,
    )
    summary = {
        "schema": "kmcfm.u8-2a-private-product-runtime-sbom-result.v1",
        "receipt_sha256": receipt_sha256,
        "inventory_count": len(first_inventory),
        "inventory_sha256": inventory_digest(first_inventory),
        "cyclonedx": {
            "path": str(paths[0]),
            "bytes": paths[0].stat().st_size,
            "sha256": _sha256(paths[0]),
        },
        "spdx": {
            "path": str(paths[1]),
            "bytes": paths[1].stat().st_size,
            "sha256": _sha256(paths[1]),
        },
        "claim": config["claim_ceiling"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
