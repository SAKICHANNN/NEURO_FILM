#!/usr/bin/env python3
"""Lock true-PNG SPCP pairs before assigning a fresh experimental role."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_u5_r2spcp2_preference_roles import load_eligible_scene_rows
from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    ROOT,
    _canonical_bytes,
    _fetch,
    _sha256,
    _stable_id,
)

DEFAULT_CONTRACT = ROOT / "configs/u5_r2spcp3_png_eligible_preference_roles_v1.json"
DEFAULT_OUTPUT = ROOT / "manifests/u5_r2spcp3_png_eligible_preference_roles_v1.json"


def _sorted_ids_sha256(scene_ids: list[str]) -> str:
    return _sha256((("\n".join(sorted(scene_ids))) + "\n").encode())


def _role_sort_key(scene_id: str) -> str:
    return hashlib.sha256(f"u5-r2spcp3-v1|{scene_id}".encode()).hexdigest()


def _probe_member(
    url: str,
    member: dict[str, Any],
    *,
    header_bytes: int,
    compressed_prefix_bytes: int,
    required_signature: bytes,
) -> dict[str, Any]:
    start = int(member["local_offset"])
    header, _ = _fetch(url, (start, start + header_bytes - 1))
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"local header missing: {member['name']}")
    values = struct.unpack_from("<IHHHHHIIIHH", header, 0)
    method, name_len, extra_len = values[3], values[9], values[10]
    if int(member["compressed_size"]) <= compressed_prefix_bytes:
        raise ValueError(f"member too small for a non-complete prefix: {member['name']}")
    compressed_start = start + header_bytes + name_len + extra_len
    compressed, _ = _fetch(
        url,
        (compressed_start, compressed_start + compressed_prefix_bytes - 1),
    )
    decoder = zlib.decompressobj(-15)
    signature = decoder.decompress(compressed, len(required_signature))
    return {
        "name": member["name"],
        "method": method,
        "local_header_range_start": start,
        "compressed_prefix_range_start": compressed_start,
        "range_bytes": len(header) + len(compressed),
        "decompressed_signature_hex": signature.hex(),
        "required_signature_exact": signature == required_signature,
    }


def build(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    parent_payloads: dict[str, Any] = {}
    parent_gates: dict[str, bool] = {}
    for name, binding in contract["parents"].items():
        data = (ROOT / binding["path"]).read_bytes()
        payload = json.loads(data)
        parent_payloads[name] = payload
        parent_gates[f"{name}_exact"] = _sha256(data) == binding["sha256"]
        if "required_decision" in binding:
            parent_gates[f"{name}_decision_exact"] = (
                payload.get("decision") == binding["required_decision"]
            )

    source = load_eligible_scene_rows(contract)
    old_rows = parent_payloads["spcp2_manifest"]["selected_rows"]
    excluded_ids = sorted({str(row["scene_id"]) for row in old_rows})
    excluded = set(excluded_ids)
    remaining = [
        row for row in source["eligible"] if str(row["scene_id"]) not in excluded
    ]
    remaining.sort(key=lambda row: str(row["scene_id"]))

    required_signature = bytes.fromhex(
        contract["eligibility"]["required_decompressed_signature_hex"]
    )
    header_bytes = int(contract["eligibility"]["local_header_bytes_per_member"])
    compressed_prefix_bytes = int(
        contract["eligibility"]["compressed_prefix_bytes_per_member"]
    )
    jobs = [
        (str(row["scene_id"]), endpoint, row[f"{endpoint}_member"])
        for row in remaining
        for endpoint in ("loser", "winner")
    ]

    def run_job(job: tuple[str, str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        scene_id, endpoint, member = job
        fact = _probe_member(
            contract["source"]["zip_url"],
            member,
            header_bytes=header_bytes,
            compressed_prefix_bytes=compressed_prefix_bytes,
            required_signature=required_signature,
        )
        return scene_id, endpoint, fact

    facts: dict[str, dict[str, dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(contract["execution"]["maximum_parallel_range_requests"])
    ) as executor:
        for scene_id, endpoint, fact in executor.map(run_job, jobs):
            facts.setdefault(scene_id, {})[endpoint] = fact

    signature_rows: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for row in remaining:
        scene_id = str(row["scene_id"])
        pair_facts = facts[scene_id]
        exact = all(
            pair_facts[endpoint]["required_signature_exact"]
            and pair_facts[endpoint]["method"]
            == int(contract["eligibility"]["required_member_method"])
            for endpoint in ("loser", "winner")
        )
        signature_rows.append(
            {
                "scene_id": scene_id,
                "loser": pair_facts["loser"],
                "winner": pair_facts["winner"],
                "pair_signature_eligible": exact,
            }
        )
        if exact:
            selected_row = dict(row)
            selected_row["scene_sort_sha256"] = _role_sort_key(scene_id)
            qualified.append(selected_row)

    qualified.sort(key=lambda row: row["scene_sort_sha256"])
    selection = contract["selection"]
    fit_end = int(selection["fit_scenes"])
    cal_end = fit_end + int(selection["calibration_scenes"])
    sealed_end = cal_end + int(selection["sealed_scenes"])
    selected = qualified[:sealed_end]
    for index, row in enumerate(selected):
        row["role"] = (
            "fit" if index < fit_end else "calibration" if index < cal_end else "sealed"
        )
    role_counts = dict(sorted(Counter(row["role"] for row in selected).items()))
    total_range_bytes = sum(
        fact[endpoint]["range_bytes"]
        for fact in signature_rows
        for endpoint in ("loser", "winner")
    )
    eligibility = contract["eligibility"]
    gates = {
        **parent_gates,
        "spcp2_exclusion_count_exact": len(excluded_ids)
        == int(contract["exclusion"]["selected_scene_count_exact"]),
        "spcp2_exclusion_identity_exact": _sorted_ids_sha256(excluded_ids)
        == contract["exclusion"]["sorted_scene_ids_lf_sha256"],
        "central_directory_exact": _sha256(source["central"])
        == contract["source"]["central_directory_sha256"],
        "pair_annotation_exact": _sha256(source["pair_bytes"])
        == contract["source"]["pair_annotation_sha256"],
        "score_annotation_exact": _sha256(source["score_bytes"])
        == contract["source"]["score_annotation_sha256"],
        "cross_table_concordance_pass": source["cross_table_concordance"]
        >= float(eligibility["minimum_cross_table_concordance"]),
        "metadata_eligible_count_exact": len(source["eligible"])
        == int(eligibility["expected_metadata_eligible_scene_count"]),
        "unassigned_scene_count_exact": len(remaining)
        == int(eligibility["expected_unassigned_scene_count"]),
        "png_pair_support": len(qualified) >= sealed_end,
        "role_counts_exact": role_counts
        == {
            "calibration": int(selection["calibration_scenes"]),
            "fit": int(selection["fit_scenes"]),
            "sealed": int(selection["sealed_scenes"]),
        },
        "selected_scenes_disjoint_from_spcp2": not any(
            row["scene_id"] in excluded for row in selected
        ),
        "range_byte_budget_pass": total_range_bytes
        <= int(eligibility["maximum_total_range_bytes"]),
        "full_member_payload_reads_zero": True,
        "image_decodes_zero": True,
        "operator_fits_zero": True,
    }
    failed = sorted(key for key, value in gates.items() if not value)
    payload = {
        "schema": "neuro-film.u5-r2spcp3-png-eligible-preference-roles-manifest.v1",
        "experiment_id": contract["experiment_id"],
        "contract_path": contract_path.relative_to(ROOT).as_posix(),
        "contract_sha256": _sha256(contract_bytes),
        "metadata_eligible_scene_count": len(source["eligible"]),
        "excluded_spcp2_scene_count": len(excluded_ids),
        "unassigned_scene_count": len(remaining),
        "png_pair_eligible_scene_count": len(qualified),
        "signature_member_count": len(jobs),
        "total_range_bytes": total_range_bytes,
        "full_member_payload_reads": 0,
        "image_decodes": 0,
        "operator_fits": 0,
        "role_counts": role_counts,
        "signature_rows": signature_rows,
        "selected_rows": selected,
        "gates": gates,
        "failed_gates": failed,
        "status": "PASS_PNG_ELIGIBLE_ROLE_LOCK" if not failed else "FAIL_CLOSED_PNG_ELIGIBILITY",
        "decision": contract["decision_if_pass"] if not failed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    payload["stable_manifest_id"] = _stable_id(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.contract.resolve(), args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["failed_gates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
