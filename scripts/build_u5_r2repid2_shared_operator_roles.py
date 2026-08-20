"""Freeze strong geometry-consistent REPID roles before image acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "K-MCFM-U5-R2REPID2/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("bounded metadata response exceeded")
    return payload


def _post_paths(url: str, paths: list[str]) -> list[dict[str, Any]]:
    payload = json.dumps({"paths": paths, "expand": True}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "K-MCFM-U5-R2REPID2/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read(4 * 1024 * 1024))


def _canonical_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r" \(\d+\)$", "", stem)


def _prior_scenes(root: Path) -> tuple[set[str], str]:
    names = sorted(path.stem for path in root.iterdir() if path.is_file())
    identity = _sha256((("\n".join(names)) + "\n").encode())
    return {_canonical_stem(name) for name in names}, identity


def eligible_rows(
    rows: list[dict[str, str]],
    *,
    geometry_fields: list[str],
    minimum_margin: float,
    excluded_stems: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    counters: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for scene, scene_rows in grouped.items():
        if len(scene_rows) != 15:
            counters["incomplete_graph"] += 1
            continue
        scores: Counter[str] = Counter()
        for row in scene_rows:
            signed = float(row["mos"]) - 0.5
            scores[row["left"]] += signed
            scores[row["right"]] -= signed
        ordered = sorted(scores, key=lambda role: (scores[role], role))
        if (
            scores[ordered[0]] == scores[ordered[1]]
            or scores[ordered[-1]] == scores[ordered[-2]]
        ):
            counters["nonunique_extreme"] += 1
            continue
        loser, winner = ordered[0], ordered[-1]
        direct = next(
            row for row in scene_rows if {row["left"], row["right"]} == {loser, winner}
        )
        margin = (float(direct["mos"]) - 0.5) * (
            1.0 if direct["left"] == winner else -1.0
        )
        if margin < minimum_margin:
            counters["weak_or_disagreeing_direct_edge"] += 1
            continue
        if not all(
            direct[f"{field}_left"] == direct[f"{field}_right"]
            for field in geometry_fields
        ):
            counters["geometry_mismatch"] += 1
            continue
        if _canonical_stem(scene) in excluded_stems:
            counters["prior_project_scene_overlap"] += 1
            continue
        eligible.append(
            {
                "scene_id": scene,
                "loser": loser,
                "winner": winner,
                "direct_margin": margin,
                "score_gap": scores[winner] - scores[loser],
            }
        )
    return eligible, counters


def evaluate(
    contract: dict[str, Any], *, fetch=_fetch, post_paths=_post_paths
) -> dict[str, Any]:
    parent_path = ROOT / contract["parent"]["evidence_path"]
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    source = contract["source"]
    markup = fetch(source["processed_markup_url"], source["processed_markup_size"])
    markup_exact = (
        len(markup) == source["processed_markup_size"]
        and _sha256(markup) == source["processed_markup_sha256"]
    )
    prior, prior_identity = _prior_scenes(ROOT / source["prior_scene_root"])
    prior_exact = (
        len(list((ROOT / source["prior_scene_root"]).iterdir()))
        == source["prior_scene_count"]
        and prior_identity == source["prior_scene_identity_sha256"]
    )
    rows = list(csv.DictReader(io.StringIO(markup.decode("utf-8-sig"))))
    selection = contract["selection"]
    eligible, exclusions = eligible_rows(
        rows,
        geometry_fields=selection["geometry_fields"],
        minimum_margin=float(selection["minimum_direct_preference_margin"]),
        excluded_stems=prior,
    )
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"u5-r2repid2-v1|{row['scene_id']}".encode()
        ).digest()
    )
    counts = {
        "fit": int(selection["fit_scenes"]),
        "calibration": int(selection["calibration_scenes"]),
        "sealed": int(selection["sealed_scenes"]),
    }
    required = sum(counts.values())
    selected = eligible[:required]
    cursor = 0
    for role, count in counts.items():
        for row in selected[cursor : cursor + count]:
            row["role"] = role
        cursor += count
    paths = [
        f"images/{endpoint}/{row['scene_id']}"
        for row in selected
        for endpoint in (row["loser"], row["winner"])
    ]
    path_rows = (
        post_paths(source["paths_info_url"], paths) if len(selected) == required else []
    )
    observed = {row["path"]: row for row in path_rows}
    members = []
    for path in paths:
        row = observed.get(path, {})
        lfs = row.get("lfs") or {}
        members.append(
            {"path": path, "size": row.get("size"), "sha256": lfs.get("oid")}
        )
    selected_identity = _sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    )
    member_identity = _sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    )
    pair_counts = Counter(f"{row['loser']}->{row['winner']}" for row in selected)
    initial_scenes = {row["scene_id"] for row in selected if row["role"] != "sealed"}
    initial_bytes = sum(
        int(row["size"] or 0)
        for row in members
        if Path(row["path"]).name in initial_scenes
    )
    total_bytes = sum(int(row["size"] or 0) for row in members)
    gates_cfg = contract["metadata_gates"]
    gates = {
        "parent_stable_evidence_exact": parent["stable_evidence_id"]
        == contract["parent"]["stable_evidence_id"],
        "parent_formal_report_exact": parent["formal_report_sha256"]
        == contract["parent"]["formal_report_sha256"],
        "parent_decision_exact": parent["decision"]
        == contract["parent"]["required_decision"],
        "processed_markup_exact": markup_exact,
        "prior_scene_inventory_exact": prior_exact,
        "eligible_scene_count": len(eligible) >= gates_cfg["eligible_scene_count_min"],
        "selected_scene_count_exact": len(selected) == required,
        "selected_prior_scene_overlap_zero": all(
            _canonical_stem(row["scene_id"]) not in prior for row in selected
        ),
        "selected_unique_directed_role_pairs": len(pair_counts)
        >= gates_cfg["selected_unique_directed_role_pairs_min"],
        "selected_largest_directed_role_pair_share": (
            max(pair_counts.values(), default=required) / max(required, 1)
            <= gates_cfg["selected_largest_directed_role_pair_share_max"]
        ),
        "all_selected_paths_resolved": len(observed) == len(paths),
        "all_paths_lfs_sha256_present": all(row["sha256"] for row in members),
        "all_paths_jpeg_extension": all(
            Path(row["path"]).suffix.lower() in {".jpg", ".jpeg"} for row in members
        ),
        "initial_fit_calibration_bytes": initial_bytes
        <= gates_cfg["initial_fit_calibration_bytes_max"],
        "all_selected_bytes": total_bytes <= gates_cfg["all_selected_bytes_max"],
        "image_member_reads_zero": True,
        "image_decodes_zero": True,
        "operator_fits_zero": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro_film.u5_r2repid2_shared_operator_roles_report.v1",
        "experiment_id": contract["experiment_id"],
        "aggregate": {
            "row_count": len(rows),
            "eligible_scene_count": len(eligible),
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "selected": {
            "role_counts": counts,
            "scene_count": len(selected),
            "identity_sha256": selected_identity,
            "directed_role_pair_counts": dict(sorted(pair_counts.items())),
            "direct_margin_minimum": min(
                (row["direct_margin"] for row in selected), default=None
            ),
        },
        "members": {
            "count": len(members),
            "identity_sha256": member_identity,
            "initial_fit_calibration_bytes": initial_bytes,
            "all_selected_bytes": total_bytes,
            "image_member_reads": 0,
        },
        "gates": gates,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid2_shared_operator_roles_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
