#!/usr/bin/env python
"""Run the frozen U6.4B photographic creative-diffusion stress."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.creative_diffusion_photographic_stress import (  # noqa: E402
    canonical_bytes,
    evaluate,
    load_contract,
)


def exact_json(path: Path, expected_sha256: str, name: str):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} hash drift")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_4b_creative_diffusion_photographic_stress_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    parents = contract["parents"]
    parent = exact_json(
        ROOT / parents["u6_4a_decision"],
        parents["u6_4a_decision_sha256"],
        "U6.4A decision",
    )
    if parent["decision"] != parents["required_decision"]:
        raise ValueError("U6.4A parent decision drift")
    inputs = contract["input"]
    manifest = exact_json(
        ROOT / inputs["manifest"],
        inputs["manifest_sha256"],
        "photographic manifest",
    )
    exact_json(
        ROOT / inputs["preflight_report"],
        inputs["preflight_report_sha256"],
        "photographic preflight",
    )
    report = evaluate(contract, manifest, root=ROOT, output_root=args.render_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(report)
    args.output.write_bytes(raw)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": hashlib.sha256(raw).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
