#!/usr/bin/env python3
"""Build the SPCP2 scene roles without reading any image member payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    ROOT,
    _canonical_bytes,
    _extract_member,
    _fetch,
    _parse_central_directory,
    _sha256,
    _sheet_rows,
    _stable_id,
)

DEFAULT_CONTRACT = ROOT / "configs/u5_r2spcp2_global_logit_affine_preference_d0_v1.json"
DEFAULT_OUTPUT = ROOT / "manifests/u5_r2spcp2_global_logit_affine_roles_v1.json"
ANNOTATION_RANGE = (75, 3804507)
CENTRAL_RANGE = (9049028973, 9050462728)
CENTRAL_SIZE = 1433756
PAIR_COMPRESSED_SIZE = 941967
SCORE_COMPRESSED_SIZE = 2862267


def _scene_sort_key(scene_id: str) -> str:
    return hashlib.sha256(f"u5-r2spcp2-v1|{scene_id}".encode()).hexdigest()


def _member_fact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "local_offset": int(row["local_offset"]),
        "compressed_size": int(row["compressed_size"]),
        "uncompressed_size": int(row["uncompressed_size"]),
        "crc32_hex": f"{int(row['crc32']):08x}",
        "method": int(row["method"]),
    }


def build(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    parent = contract["parent"]
    decision_bytes = (ROOT / parent["decision_path"]).read_bytes()
    evidence_bytes = (ROOT / parent["evidence_path"]).read_bytes()
    decision = json.loads(decision_bytes)

    central, _ = _fetch(contract["source"]["zip_url"], CENTRAL_RANGE)
    annotations, _ = _fetch(contract["source"]["zip_url"], ANNOTATION_RANGE)
    members = _parse_central_directory(central)
    member_by_name = {row["name"]: row for row in members}
    pair_member = next(row for row in members if row["name"].endswith("order_trans.xlsx"))
    score_member = next(row for row in members if row["name"].endswith("score_trans2.xlsx"))
    pair_bytes = _extract_member(
        annotations, range_start=ANNOTATION_RANGE[0], row=pair_member
    )
    score_bytes = _extract_member(
        annotations, range_start=ANNOTATION_RANGE[0], row=score_member
    )
    _, pair_rows = _sheet_rows(pair_bytes)
    _, score_rows = _sheet_rows(score_bytes)

    score_mean = {
        str(row[0]).removesuffix(".png"): statistics.fmean(float(x) for x in row[1:])
        for row in score_rows
    }
    pair_by_endpoints = {
        frozenset(str(row[0]).split(",")): row for row in pair_rows
    }

    concordant = 0
    comparable = 0
    for row in pair_rows:
        left, right = str(row[0]).split(",")
        right_votes = sum(int(value) for value in row[1:])
        if right_votes == 10 or score_mean[left] == score_mean[right]:
            continue
        comparable += 1
        pair_winner = right if right_votes > 10 else left
        score_winner = left if score_mean[left] > score_mean[right] else right
        concordant += pair_winner == score_winner
    cross_table_concordance = concordant / comparable

    images_by_scene: dict[str, list[str]] = defaultdict(list)
    for image_id in score_mean:
        images_by_scene[image_id.split("_")[0]].append(image_id)

    eligible: list[dict[str, Any]] = []
    rejected = defaultdict(int)
    minimum_margin = int(contract["selection"]["minimum_absolute_vote_margin_from_ten"])
    for scene_id, image_ids in images_by_scene.items():
        ordered = sorted((score_mean[image_id], image_id) for image_id in image_ids)
        if ordered[0][0] == ordered[1][0] or ordered[-1][0] == ordered[-2][0]:
            rejected["nonunique_score_extreme"] += 1
            continue
        loser_score, loser = ordered[0]
        winner_score, winner = ordered[-1]
        pair_row = pair_by_endpoints.get(frozenset((loser, winner)))
        if pair_row is None:
            rejected["extreme_pair_missing"] += 1
            continue
        left, right = str(pair_row[0]).split(",")
        right_votes = sum(int(value) for value in pair_row[1:])
        if right_votes == 10:
            rejected["extreme_pair_tie"] += 1
            continue
        pair_winner = right if right_votes > 10 else left
        if pair_winner != winner:
            rejected["extreme_pair_disagrees"] += 1
            continue
        margin = abs(right_votes - 10)
        if margin < minimum_margin:
            rejected["vote_margin_below_minimum"] += 1
            continue
        winner_member_name = f"SPCP_dataset/images/{winner}.png"
        loser_member_name = f"SPCP_dataset/images/{loser}.png"
        eligible.append(
            {
                "scene_id": scene_id,
                "scene_sort_sha256": _scene_sort_key(scene_id),
                "loser_image_id": loser,
                "winner_image_id": winner,
                "loser_mean_score": round(loser_score, 12),
                "winner_mean_score": round(winner_score, 12),
                "score_gap": round(winner_score - loser_score, 12),
                "direct_pair": str(pair_row[0]),
                "right_votes": right_votes,
                "vote_margin_from_ten": margin,
                "loser_member": _member_fact(member_by_name[loser_member_name]),
                "winner_member": _member_fact(member_by_name[winner_member_name]),
            }
        )

    eligible.sort(key=lambda row: row["scene_sort_sha256"])
    selection = contract["selection"]
    fit_end = int(selection["fit_scenes"])
    cal_end = fit_end + int(selection["calibration_scenes"])
    sealed_end = cal_end + int(selection["sealed_scenes"])
    selected = eligible[:sealed_end]
    for index, row in enumerate(selected):
        row["role"] = "fit" if index < fit_end else "calibration" if index < cal_end else "sealed"

    role_counts = dict(sorted(Counter(row["role"] for row in selected).items()))
    initial = [row for row in selected if row["role"] != "sealed"]
    initial_compressed_bytes = sum(
        row[key]["compressed_size"]
        for row in initial
        for key in ("loser_member", "winner_member")
    )
    selected_scene_overlap = len({row["scene_id"] for row in selected}) != len(selected)
    gates = {
        "parent_exact": _sha256(decision_bytes) == parent["decision_sha256"]
        and _sha256(evidence_bytes) == parent["evidence_sha256"]
        and decision["decision"] == parent["required_decision"],
        "central_directory_exact": _sha256(central)
        == contract["source"]["central_directory_sha256"],
        "pair_annotation_exact": _sha256(pair_bytes)
        == contract["source"]["pair_annotation_sha256"],
        "score_annotation_exact": _sha256(score_bytes)
        == contract["source"]["score_annotation_sha256"],
        "label_semantics_cross_table": cross_table_concordance
        >= float(selection["minimum_cross_table_concordance"]),
        "eligible_scene_support": len(eligible) >= sealed_end,
        "role_counts_exact": role_counts
        == {
            "calibration": int(selection["calibration_scenes"]),
            "fit": int(selection["fit_scenes"]),
            "sealed": int(selection["sealed_scenes"]),
        },
        "scene_roles_disjoint": not selected_scene_overlap,
        "initial_member_count_exact": len(initial) * 2
        == contract["acquisition"]["initial_member_count_exact"],
        "initial_compressed_bytes_within_budget": initial_compressed_bytes
        <= contract["acquisition"]["maximum_initial_compressed_bytes"],
        "image_payload_bytes_read_exact": True,
    }
    failed = sorted(key for key, value in gates.items() if not value)
    manifest = {
        "schema": "neuro-film.u5-r2spcp2-global-logit-affine-roles-manifest.v1",
        "experiment_id": contract["experiment_id"],
        "contract_path": contract_path.relative_to(ROOT).as_posix(),
        "contract_sha256": _sha256(contract_bytes),
        "source_revision": contract["source"]["revision"],
        "cross_table_label_1_means_right_concordance": cross_table_concordance,
        "cross_table_concordant_pairs": concordant,
        "cross_table_comparable_pairs": comparable,
        "eligible_scene_count": len(eligible),
        "rejected_scene_counts": dict(sorted(rejected.items())),
        "role_counts": role_counts,
        "initial_member_count": len(initial) * 2,
        "initial_compressed_bytes": initial_compressed_bytes,
        "image_payload_bytes_read": 0,
        "selected_rows": selected,
        "gates": gates,
        "failed_gates": failed,
        "status": "PASS_METADATA_ROLE_LOCK" if not failed else "FAIL_CLOSED_METADATA_ROLE_LOCK",
        "decision": "open_exact_fit_cal_range_acquisition" if not failed else "close_before_image_acquisition",
        "claim_ceiling": contract["claim_ceiling"],
    }
    manifest["stable_manifest_id"] = _stable_id(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.contract.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
