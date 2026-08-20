"""Evaluate scene-stable professional-render role utility in REPID metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "K-MCFM-U5-R2REPID5/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("bounded metadata response exceeded")
    return payload


def _load_complete_scenes(
    payload: bytes, roles: tuple[str, ...], rows_per_scene: int
) -> dict[str, list[tuple[str, str, float]]]:
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or not {"name", "left", "right", "mos"}.issubset(
        reader.fieldnames
    ):
        raise ValueError("REPID aggregate columns are incomplete")
    role_set = set(roles)
    for row in reader:
        left, right = row["left"], row["right"]
        mos = float(row["mos"])
        if left not in role_set or right not in role_set or left == right:
            raise ValueError("unexpected REPID role identity")
        if not math.isfinite(mos) or not 0.0 <= mos <= 1.0:
            raise ValueError("invalid REPID MOS")
        grouped[row["name"]].append((left, right, mos))
    return {
        scene: rows
        for scene, rows in grouped.items()
        if len(rows) == rows_per_scene
        and len({tuple(sorted((left, right))) for left, right, _ in rows})
        == rows_per_scene
    }


def _fit_utilities(
    scenes: list[list[tuple[str, str, float]]], roles: tuple[str, ...]
) -> dict[str, float]:
    totals = {role: 0.0 for role in roles}
    for rows in scenes:
        for left, right, mos in rows:
            signed = mos - 0.5
            totals[left] += signed
            totals[right] -= signed
    scale = max(len(scenes), 1)
    return {role: totals[role] / scale for role in roles}


def _score(
    scenes: list[list[tuple[str, str, float]]], utilities: dict[str, float]
) -> dict[str, Any]:
    correct = 0
    total = 0
    by_pair: dict[str, list[bool]] = defaultdict(list)
    top = max(utilities, key=lambda role: (utilities[role], role))
    top_rows: dict[str, list[bool]] = defaultdict(list)
    for rows in scenes:
        for left, right, mos in rows:
            actual = 1 if mos > 0.5 else -1 if mos < 0.5 else 0
            predicted = 1 if utilities[left] > utilities[right] else -1
            if actual == 0:
                continue
            hit = actual == predicted
            correct += int(hit)
            total += 1
            pair = "->".join(sorted((left, right)))
            by_pair[pair].append(hit)
            if top in {left, right}:
                opponent = right if left == top else left
                top_preferred = (actual == 1 and left == top) or (
                    actual == -1 and right == top
                )
                top_rows[opponent].append(top_preferred)
    return {
        "accuracy": correct / max(total, 1),
        "pair_accuracies": {
            pair: sum(values) / len(values) for pair, values in sorted(by_pair.items())
        },
        "top_role": top,
        "top_role_opponent_win_rates": {
            role: sum(values) / len(values)
            for role, values in sorted(top_rows.items())
        },
    }


def evaluate(
    contract: dict[str, Any], *, fetch: Callable[[str, int], bytes] = _fetch
) -> dict[str, Any]:
    parent_cfg = contract["parent"]
    parent_path = ROOT / parent_cfg["evidence_path"]
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    selected_path = ROOT / parent_cfg["selected_roles_report_path"]
    selected_bytes = selected_path.read_bytes()
    selected = json.loads(selected_bytes)
    excluded = {row["scene_id"] for row in selected["selected"]["rows"]}

    source = contract["source"]
    payload = fetch(source["processed_markup_url"], source["processed_markup_size"])
    source_exact = (
        len(payload) == source["processed_markup_size"]
        and _sha256(payload) == source["processed_markup_sha256"]
    )
    roles = tuple(source["roles"])
    protocol = contract["protocol"]
    complete = _load_complete_scenes(
        payload, roles, int(protocol["complete_pair_rows_per_scene"])
    )
    available = sorted(set(complete) - excluded)
    domain = protocol["split_hash_domain"]
    cut = int(protocol["development_hash_byte_upper_exclusive"])
    development_ids = [
        scene for scene in available if hashlib.sha256(f"{domain}{scene}".encode()).digest()[0] < cut
    ]
    confirmation_ids = [scene for scene in available if scene not in set(development_ids)]
    development = [complete[scene] for scene in development_ids]
    confirmation = [complete[scene] for scene in confirmation_ids]
    utilities = _fit_utilities(development, roles)
    development_score = _score(development, utilities)
    confirmation_score = _score(confirmation, utilities)

    fold_count = int(protocol["leave_one_hash_fold_count"])
    fold_tops = []
    for fold in range(fold_count):
        kept = [
            complete[scene]
            for scene in development_ids
            if hashlib.sha256(f"{domain}fold|{scene}".encode()).digest()[0]
            % fold_count
            != fold
        ]
        fold_utility = _fit_utilities(kept, roles)
        fold_tops.append(max(fold_utility, key=lambda role: (fold_utility[role], role)))
    top = confirmation_score["top_role"]
    top_stability = sum(value == top for value in fold_tops) / fold_count

    rotated = {role: utilities[roles[(index + 1) % len(roles)]] for index, role in enumerate(roles)}
    permuted_score = _score(confirmation, rotated)
    pair_majority_agreement = sum(
        value >= 0.5
        for value in confirmation_score["pair_accuracies"].values()
    ) / len(confirmation_score["pair_accuracies"])

    gates_cfg = contract["gates"]
    metrics = {
        "complete_scene_count": len(complete),
        "excluded_selected_scene_count": len(excluded),
        "development_scene_count": len(development),
        "confirmation_scene_count": len(confirmation),
        "development_pairwise_accuracy": development_score["accuracy"],
        "confirmation_pairwise_accuracy": confirmation_score["accuracy"],
        "worst_role_pair_accuracy": min(confirmation_score["pair_accuracies"].values()),
        "top_role": top,
        "minimum_top_role_opponent_win_rate": min(
            confirmation_score["top_role_opponent_win_rates"].values()
        ),
        "leave_one_fold_top_role_stability": top_stability,
        "development_confirmation_accuracy_gap": abs(
            development_score["accuracy"] - confirmation_score["accuracy"]
        ),
        "pair_majority_agreement_rate": pair_majority_agreement,
        "label_permuted_accuracy": permuted_score["accuracy"],
    }
    gates = {
        "parent_evidence_exact": _sha256(parent_bytes)
        == parent_cfg["evidence_sha256"],
        "parent_decision_exact": parent["decision"]
        == parent_cfg["required_decision"],
        "selected_roles_report_exact": _sha256(selected_bytes)
        == parent_cfg["selected_roles_report_sha256"],
        "selected_roles_stable_exact": selected["stable_evidence_id"]
        == parent_cfg["selected_roles_stable_evidence_id"],
        "processed_markup_exact": source_exact,
        "complete_scene_count": metrics["complete_scene_count"]
        >= gates_cfg["minimum_complete_scene_count"],
        "excluded_selected_scene_count": metrics["excluded_selected_scene_count"]
        == gates_cfg["excluded_selected_scene_count"],
        "development_scene_count": metrics["development_scene_count"]
        >= gates_cfg["minimum_development_scene_count"],
        "confirmation_scene_count": metrics["confirmation_scene_count"]
        >= gates_cfg["minimum_confirmation_scene_count"],
        "confirmation_pairwise_accuracy": metrics["confirmation_pairwise_accuracy"]
        >= gates_cfg["minimum_confirmation_pairwise_accuracy"],
        "worst_role_pair_accuracy": metrics["worst_role_pair_accuracy"]
        >= gates_cfg["minimum_worst_role_pair_accuracy"],
        "top_role_opponent_win_rate": metrics["minimum_top_role_opponent_win_rate"]
        >= gates_cfg["minimum_top_role_opponent_win_rate"],
        "leave_one_fold_top_role_stability": metrics["leave_one_fold_top_role_stability"]
        >= gates_cfg["minimum_leave_one_fold_top_role_stability"],
        "development_confirmation_accuracy_gap": metrics["development_confirmation_accuracy_gap"]
        <= gates_cfg["maximum_development_confirmation_accuracy_gap"],
        "pair_majority_agreement_rate": metrics["pair_majority_agreement_rate"]
        >= gates_cfg["minimum_pair_majority_agreement_rate"],
        "label_permuted_accuracy": metrics["label_permuted_accuracy"]
        <= gates_cfg["maximum_label_permuted_accuracy"],
        "image_member_reads_zero": True,
        "image_decodes_zero": True,
        "operator_fits_zero": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro_film.u5_r2repid5_global_role_utility_d0_report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ),
        "split_identity_sha256": _sha256(
            json.dumps(
                {"development": development_ids, "confirmation": confirmation_ids},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "utilities": utilities,
        "development_pair_accuracies": development_score["pair_accuracies"],
        "confirmation_pair_accuracies": confirmation_score["pair_accuracies"],
        "top_role_opponent_win_rates": confirmation_score[
            "top_role_opponent_win_rates"
        ],
        "leave_one_fold_top_roles": fold_tops,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": sorted(name for name, value in gates.items() if not value),
        "image_member_reads": 0,
        "image_decodes": 0,
        "operator_fits": 0,
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
        default=ROOT / "configs/u5_r2repid5_global_role_utility_d0_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
