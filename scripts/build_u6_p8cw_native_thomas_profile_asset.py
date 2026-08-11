#!/usr/bin/env python3
"""Build the exact P8CR canonical Thomas profile as a tracked runtime asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_thomas_atomic_publication import _compile_profile_payload, _json
from src.film_physics.native_thomas_export_profile import canonical_profile_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8cv_canonical_profile_atomic_scale_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json",
    )
    args = parser.parse_args()
    contract = _json(args.contract)
    _prior, profile = _compile_profile_payload(ROOT, contract)
    if profile["profile_sha256"] != contract["profile_sha256"]:
        raise RuntimeError("P8CW profile identity drift")
    encoded = canonical_profile_bytes(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "bytes": len(encoded),
                "path": str(args.output.resolve()),
                "profile_sha256": profile["profile_sha256"],
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
