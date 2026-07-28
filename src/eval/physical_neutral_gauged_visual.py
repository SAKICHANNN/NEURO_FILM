"""U6.P7F1 frozen visual confirmation for the neutral-gauged chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw

from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_chain import validate_contract
from src.eval.physical_virtual_scan_visual import _load_exact_json, _tile


SCHEMA = (
    "neuro_film.u6_p7f1_neutral_gauged_visual_confirmation_contract.v1"
)
ADJUDICATION_SCHEMA = (
    "neuro_film.u6_p7f1_neutral_gauged_visual_adjudication_contract.v1"
)


def validate_visual_contract(
    root: Path, config: dict[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config.get("post_result_retuning_allowed")
        or config["blind"].get("score_before_mapping_reveal") is not True
        or config["severe_review"].get(
            "confirmed_severe_blocks_mapping_reveal"
        )
        is not True
    ):
        raise ValueError("unsupported U6.P7F1 contract")
    parent_contract = _load_exact_json(
        root,
        config["parent_contract"],
        config["parent_contract_sha256"],
    )
    runtime, _ = validate_contract(root, parent_contract)
    first = _load_exact_json(
        root,
        config["parent_report_run_a"],
        config["parent_report_sha256"],
    )
    second = _load_exact_json(
        root,
        config["parent_report_run_b"],
        config["parent_report_sha256"],
    )
    colour = _load_exact_json(
        root,
        config["colour_baseline_report"],
        config["colour_baseline_report_sha256"],
    )
    selected = config["selected_candidate_id"]
    if (
        first != second
        or first["selected_candidate_arm_id"] != selected
        or selected not in first["automatic_eligible_arm_ids"]
        or not all(first["decisions"][selected].values())
        or colour["node"] != "U6.P7B"
        or not colour["automatic_pass"]
    ):
        raise ValueError("U6.P7F1 parent evidence drift")
    visual_ids = config["visual_ids"]
    if (
        len(visual_ids) != 9
        or len(set(visual_ids)) != len(visual_ids)
        or any(sample_id not in runtime.eligible_ids for sample_id in visual_ids)
        or int(config["blind"]["rounds"]) != 3
        or config["blind"]["arms"] != ["colour_only", selected]
    ):
        raise ValueError("U6.P7F1 visual protocol drift")
    return runtime, first, colour


def _visual_paths(
    *,
    root: Path,
    config: dict[str, Any],
    runtime: Any,
    parent: dict[str, Any],
    colour: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    selected = config["selected_candidate_id"]
    candidate_hashes = {
        row["sample_id"]: row["output_sha256"]
        for row in parent["rows"]
        if row["arm_id"] == selected
    }
    colour_rows = {
        row["sample_id"]: row
        for row in colour["photographic_ablation"]["rows"]
    }
    paths: dict[str, dict[str, Path]] = {}
    for sample_id in config["visual_ids"]:
        source = root / runtime.source_rows[sample_id]["decoded_path"]
        colour_only = (
            root
            / "outputs"
            / "u6_p7b_source_context_joint_ablation_v1"
            / "run_a"
            / "renders"
            / "colour_only"
            / f"{sample_id}.png"
        )
        candidate = (
            root
            / "outputs"
            / "u6_p7f_neutral_gauged_physical_chain_v1"
            / "run_a"
            / "renders"
            / selected
            / f"{sample_id}.png"
        )
        expected = {
            source: runtime.source_rows[sample_id]["decoded_sha256"],
            colour_only: colour_rows[sample_id]["output_sha256"]["colour_only"],
            candidate: candidate_hashes[sample_id],
        }
        for path, digest in expected.items():
            if sha256_file(path) != digest:
                raise ValueError(f"visual input hash drift: {sample_id}")
        paths[sample_id] = {
            "source": source,
            "colour_only": colour_only,
            selected: candidate,
        }
    return paths


def build_visual_evidence(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime, parent, colour = validate_visual_contract(root, config)
    paths = _visual_paths(
        root=root,
        config=config,
        runtime=runtime,
        parent=parent,
        colour=colour,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = (320, 210)
    header = 24
    mappings: dict[str, Any] = {}
    sheet_hashes: list[str] = []
    for round_index in range(1, int(config["blind"]["rounds"]) + 1):
        canvas = Image.new(
            "RGB",
            (
                3 * tile_size[0],
                len(config["visual_ids"]) * (tile_size[1] + header),
            ),
            (24, 24, 24),
        )
        draw = ImageDraw.Draw(canvas)
        round_mapping: dict[str, dict[str, str]] = {}
        for row_index, sample_id in enumerate(config["visual_ids"]):
            order = list(config["blind"]["arms"])
            random.Random(
                int(config["blind"]["seed"])
                + round_index * 1009
                + row_index
            ).shuffle(order)
            round_mapping[sample_id] = {"A": order[0], "B": order[1]}
            row_paths = (
                paths[sample_id]["source"],
                paths[sample_id][order[0]],
                paths[sample_id][order[1]],
            )
            y = row_index * (tile_size[1] + header)
            for column, (label, path) in enumerate(
                zip(("SOURCE", "A", "B"), row_paths, strict=True)
            ):
                canvas.paste(
                    _tile(path, tile_size),
                    (column * tile_size[0], y + header),
                )
                draw.text(
                    (column * tile_size[0] + 4, y + 4),
                    f"SOURCE {sample_id}" if column == 0 else label,
                    fill=(235, 235, 235),
                )
        mappings[f"round_{round_index}"] = round_mapping
        path = output_dir / f"blind_round_{round_index}.png"
        canvas.save(path, format="PNG", compress_level=6)
        sheet_hashes.append(sha256_file(path))
    if len(
        {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in mappings.values()
        }
    ) != int(config["blind"]["rounds"]):
        raise RuntimeError("blind round mappings must differ")
    mapping_raw = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output_dir / "private_mapping.json").write_bytes(mapping_raw)
    core = {
        "schema": "neuro_film.u6_p7f1_neutral_gauged_visual_evidence.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "selected_candidate_id": config["selected_candidate_id"],
        "visual_ids": config["visual_ids"],
        "blind_sheet_sha256": sheet_hashes,
        "private_mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "status": "awaiting_severe_and_blind_scoring",
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def adjudicate_visual_evidence(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    if config.get("schema") != ADJUDICATION_SCHEMA:
        raise ValueError("unsupported U6.P7F1 adjudication contract")
    visual_contract = _load_exact_json(
        root,
        config["visual_contract"],
        config["visual_contract_sha256"],
    )
    report = _load_exact_json(
        root,
        config["visual_report"],
        config["visual_report_sha256"],
    )
    scoring = _load_exact_json(
        root,
        config["blind_scoring"],
        config["blind_scoring_sha256"],
    )
    mapping = _load_exact_json(
        root,
        config["private_mapping"],
        config["private_mapping_sha256"],
    )
    if (
        scoring.get("status") != "frozen_before_mapping_reveal"
        or report["private_mapping_sha256"]
        != config["private_mapping_sha256"]
        or report["selected_candidate_id"]
        != visual_contract["selected_candidate_id"]
        or set(scoring["blind_choices"]) != set(mapping)
    ):
        raise ValueError("U6.P7F1 scoring or mapping drift")
    arms = set(visual_contract["blind"]["arms"])
    candidate = visual_contract["selected_candidate_id"]
    per_round: dict[str, Any] = {}
    total = {arm: 0 for arm in arms}
    for round_id, choices in scoring["blind_choices"].items():
        if set(choices) != set(visual_contract["visual_ids"]):
            raise ValueError("blind scoring population drift")
        counts = {arm: 0 for arm in arms}
        for sample_id, label in choices.items():
            if label not in {"A", "B"}:
                raise ValueError("invalid blind label")
            arm = mapping[round_id][sample_id][label]
            if arm not in arms:
                raise ValueError("invalid mapped arm")
            counts[arm] += 1
            total[arm] += 1
        winner = max(counts, key=counts.get)
        if list(counts.values()).count(counts[winner]) != 1:
            winner = "tie"
        per_round[round_id] = {"counts": counts, "winner": winner}
    candidate_round_wins = sum(
        row["winner"] == candidate for row in per_round.values()
    )
    severe_count = int(
        scoring["severe_review"]["confirmed_severe_candidate_artifacts"]
    )
    if severe_count > 0:
        decision = "severe_fail"
    elif candidate_round_wins < int(config["minimum_candidate_round_wins"]):
        decision = "preference_fail"
    else:
        decision = "complete_pass"
    core = {
        "schema": (
            "neuro_film.u6_p7f1_neutral_gauged_visual_adjudication.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "parent_stable_evidence_id": report["stable_evidence_id"],
        "severe_confirmed_count": severe_count,
        "per_round": per_round,
        "total_choices": total,
        "candidate_round_wins": candidate_round_wins,
        "decision": decision,
        "branch": config["branch_rule"][decision],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


__all__ = [
    "adjudicate_visual_evidence",
    "build_visual_evidence",
    "validate_visual_contract",
    "write_report",
]
