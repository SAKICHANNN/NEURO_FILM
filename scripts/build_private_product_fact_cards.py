from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_fact_cards import (
    build_product_fact_cards,
    canonical_json,
    publish_product_fact_cards,
    sha256_file,
)
from src.inference.product_look_catalog import list_product_looks

DEFAULT_CONFIG = ROOT / "configs/u8_2b_private_look_approximation_fact_cards_v1.json"


def build_from_config(config_path: Path) -> tuple[dict[str, object], bytes]:
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text("utf-8"))
    receipt_path = (ROOT / config["input"]["runtime_receipt"]).resolve(strict=True)
    profile_path = (ROOT / config["input"]["profile"]).resolve(strict=True)
    evidence_sha256 = {
        role: sha256_file((ROOT / row["path"]).resolve(strict=True))
        for role, row in config["evidence"].items()
    }
    bundle = build_product_fact_cards(
        config=config,
        receipt=json.loads(receipt_path.read_text("utf-8")),
        receipt_sha256=sha256_file(receipt_path),
        profile=json.loads(profile_path.read_text("utf-8")),
        profile_sha256=sha256_file(profile_path),
        catalog=list_product_looks(),
        evidence_sha256=evidence_sha256,
        root_license_present=(ROOT / config["input"]["root_license"]).exists(),
    )
    return bundle, canonical_json(bundle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact private K-MCFM Look Approximation fact cards."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text("utf-8"))
    output = args.output
    if output is None:
        output = (
            ROOT
            / config["output"]["directory"]
            / config["output"]["filename"]
        )
    output = output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    _, payload = build_from_config(config_path)
    publish_product_fact_cards(output, payload)
    print(
        json.dumps(
            {
                "bytes": len(payload),
                "output": str(output),
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
