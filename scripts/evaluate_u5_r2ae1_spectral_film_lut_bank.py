from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.spectral_film_lut_bank import (  # noqa: E402
    canonical_sha256,
    evaluate_manifests,
    load_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_manifests(
        load_manifest(args.manifest_a),
        load_manifest(args.manifest_b),
        config,
        root=ROOT,
    )
    report["lineage"] = {
        "config_sha256": canonical_sha256(config),
        "manifest_a_sha256": canonical_sha256(load_manifest(args.manifest_a)),
        "manifest_b_sha256": canonical_sha256(load_manifest(args.manifest_b)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()

