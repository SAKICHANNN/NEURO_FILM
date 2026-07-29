#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_interimage_visible_value import (  # noqa: E402
    evaluate_interimage_visible_value,
    load_contract,
    write_json,
)


def _exact_json(path: Path, expected_sha256: str, name: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} hash does not match frozen parent")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p5h_interimage_visible_value_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-directory", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    p5g_decision = _exact_json(
        ROOT
        / "configs"
        / "u6_p5g_interimage_photographic_stress_decision_v1.json",
        contract["parents"]["p5g_decision_sha256"],
        "P5G decision",
    )
    p5g_contract = _exact_json(
        ROOT / contract["input"]["reuse_p5g_contract"],
        contract["input"]["reuse_p5g_contract_sha256"],
        "P5G contract",
    )
    p5f_decision = _exact_json(
        ROOT / "configs" / "u6_p5f_interimage_adjacency_decision_v1.json",
        contract["parents"]["p5f_decision_sha256"],
        "P5F decision",
    )
    p5f_contract = _exact_json(
        ROOT / "configs" / "u6_p5f_interimage_adjacency_v1.json",
        p5f_decision["contract_sha256"],
        "P5F contract",
    )
    p5d_contract = _exact_json(
        ROOT / p5g_contract["input"]["reuse_p5d_contract"],
        p5g_contract["input"]["reuse_p5d_contract_sha256"],
        "P5D contract",
    )
    p5c_decision = _exact_json(
        ROOT / "configs" / "u6_p5c_bounded_adjacency_decision_v1.json",
        p5d_contract["parents"]["p5c_decision_sha256"],
        "P5C decision",
    )
    p5c_contract = _exact_json(
        ROOT / "configs" / "u6_p5c_bounded_adjacency_v1.json",
        p5c_decision["contract_sha256"],
        "P5C contract",
    )
    p5b_contract = _exact_json(
        ROOT / "configs" / "u6_p5b_spatial_halo_audit_v1.json",
        p5c_contract["parents"]["p5b_contract_sha256"],
        "P5B contract",
    )
    p5a_contract = _exact_json(
        ROOT / "configs" / "u6_p5a_spatial_response_primitives_v1.json",
        p5b_contract["parents"]["p5a_contract_sha256"],
        "P5A contract",
    )
    sensitometry = _exact_json(
        ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json",
        p5a_contract["parents"]["sensitometry_config_sha256"],
        "sensitometry",
    )
    input_spec = p5d_contract["input"]
    manifest = _exact_json(
        ROOT / input_spec["manifest"],
        input_spec["manifest_sha256"],
        "photographic manifest",
    )
    _exact_json(
        ROOT / input_spec["preflight_report"],
        input_spec["preflight_report_sha256"],
        "photographic preflight",
    )
    report, key = evaluate_interimage_visible_value(
        contract,
        manifest,
        p5g_decision,
        p5c_contract,
        p5f_contract,
        p5a_contract,
        sensitometry,
        root=ROOT,
        blind_directory=args.blind_directory,
    )
    report_sha = write_json(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    if key is not None:
        key_sha = write_json(key, args.blind_key)
        print(f"blind_key_sha256={key_sha}")


if __name__ == "__main__":
    main()
