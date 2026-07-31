"""Blinded autonomous comparison of fixed BK10 and three controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file


ARMS = (
    "fixed_bk10_safe_base_orthogonal_residual",
    "fixed_bk7_smooth_perceptual_hue_density",
    "fixed_ao6_colour_only_t15_c35",
    "safe_rich_velvia_50",
)
LABELS = ("A", "B", "C", "D")


def _load_exact(root: Path, binding: dict[str, str]) -> Any:
    path = root / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"identity drift: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk15_orthogonal_residual_blind_preference.v1"
        or config.get("status") != "contract_frozen_build_ready"
        or tuple(config["arms"]) != ARMS
    ):
        raise ValueError("BK15 contract drift")
    decision = _load_exact(root, config["parents"]["decision"])
    if decision.get("decision") != config["parents"]["decision"][
        "required_decision"
    ]:
        raise ValueError("BK14 decision is not eligible")
    report = _load_exact(root, config["parents"]["render_report"])
    manifest = _load_exact(root, config["population"]["manifest"])
    if (
        not report.get("automatic_pass")
        or not report.get("visual_review_allowed")
        or tuple(report["arms"]) != ARMS
        or len(manifest) != int(config["population"]["expected_rows"])
        or len({row["id"] for row in manifest}) != len(manifest)
    ):
        raise ValueError("BK15 render population drift")
    return manifest, report


def build_blind_sheets(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    manifest, report = _validate_inputs(root, config)
    render_root = root / config["parents"]["render_root"]
    records = {
        (row["source_id"], row["arm_id"]): row for row in report["records"]
    }
    expected_keys = {
        (str(source["id"]), arm) for source in manifest for arm in ARMS
    }
    if set(records) != expected_keys:
        raise ValueError("BK15 render inventory mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, Any]] = []
    for round_index in (1, 2, 3):
        mapping: list[dict[str, str]] = []
        tiles: list[Image.Image] = []
        for source_row in manifest:
            source_id = str(source_row["id"])
            order = list(ARMS)
            random.Random(
                hashlib.sha256(
                    (
                        f"{config['experiment_id']}:{round_index}:"
                        f"{source_id}"
                    ).encode()
                ).digest()
            ).shuffle(order)
            mapping.append(
                {
                    "source_id": source_id,
                    **dict(zip(LABELS, order, strict=True)),
                }
            )
            source_path = root / str(source_row["decoded_path"])
            if sha256_file(source_path) != source_row["decoded_sha256"]:
                raise ValueError("BK15 source identity drift")
            arm_paths: dict[str, Path] = {}
            for arm in ARMS:
                record = records[(source_id, arm)]
                path = render_root / record["output"]
                if sha256_file(path) != record["output_sha256"]:
                    raise ValueError("BK15 render identity drift")
                arm_paths[arm] = path
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
                source.thumbnail((310, 216), Image.Resampling.LANCZOS)
            candidates = []
            for arm in order:
                with Image.open(arm_paths[arm]) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((310, 216), Image.Resampling.LANCZOS)
                candidates.append(image)
            tile = Image.new("RGB", (1600, 254), "white")
            draw = ImageDraw.Draw(tile)
            font = ImageFont.load_default()
            draw.text((6, 3), source_id, fill="black", font=font)
            for index, (label, image) in enumerate(
                zip(("Source", *LABELS), (source, *candidates), strict=True)
            ):
                x = 5 + index * 319
                draw.text((x, 20), label, fill="black", font=font)
                tile.paste(image, (x, 36))
            tiles.append(tile)
        mapping_path = output_dir / f"round_{round_index}_mapping.json"
        mapping_path.write_text(
            json.dumps(mapping, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for part, start in enumerate(range(0, len(tiles), 4), start=1):
            selected = tiles[start : start + 4]
            sheet = Image.new(
                "RGB", (1600, len(selected) * 254 + 28), "white"
            )
            ImageDraw.Draw(sheet).text(
                (6, 6),
                f"BK15 blind round {round_index} part {part}",
                fill="black",
                font=ImageFont.load_default(),
            )
            for index, tile in enumerate(selected):
                sheet.paste(tile, (0, 28 + index * 254))
            sheet_path = (
                output_dir / f"blind_round_{round_index}_part_{part}.png"
            )
            sheet.save(sheet_path, "PNG", compress_level=6)
            artifacts.append(
                {
                    "round": round_index,
                    "part": part,
                    "sheet": sheet_path.name,
                    "sheet_sha256": sha256_file(sheet_path),
                }
            )
        artifacts.append(
            {
                "round": round_index,
                "mapping": mapping_path.name,
                "mapping_sha256": sha256_file(mapping_path),
            }
        )
    receipt = {
        "schema": "neuro_film.u5_r2bk15_blind_build.v1",
        "experiment_id": config["experiment_id"],
        "source_ids": [str(row["id"]) for row in manifest],
        "arms": list(ARMS),
        "artifacts": artifacts,
    }
    receipt["stable_evidence_id"] = canonical_sha256(receipt)
    receipt_path = output_dir / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def adjudicate_blind(
    *,
    root: Path,
    config: dict[str, Any],
    build_dir: Path,
    observations_path: Path,
) -> dict[str, Any]:
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("mapping_unread_when_recorded") is not True:
        raise ValueError("observations were not recorded blind")
    receipt_path = build_dir / "build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("experiment_id") != config["experiment_id"]:
        raise ValueError("BK15 build receipt mismatch")
    expected_mappings = {
        int(row["round"]): row
        for row in receipt["artifacts"]
        if "mapping" in row
    }
    if set(expected_mappings) != {1, 2, 3}:
        raise ValueError("BK15 mapping inventory mismatch")
    counts = {arm: 0 for arm in (*ARMS, "tie")}
    round_wins = {arm: 0 for arm in (*ARMS, "tie")}
    pairwise_round_wins = {
        control: {ARMS[0]: 0, control: 0, "tie": 0}
        for control in ARMS[1:]
    }
    decoded_rounds = []
    mapping_hashes = []
    for round_index in (1, 2, 3):
        mapping_path = build_dir / f"round_{round_index}_mapping.json"
        mapping_hash = sha256_file(mapping_path)
        if mapping_hash != expected_mappings[round_index]["mapping_sha256"]:
            raise ValueError("BK15 mapping identity drift")
        mapping_hashes.append(mapping_hash)
        by_id = {
            row["source_id"]: row
            for row in json.loads(mapping_path.read_text(encoding="utf-8"))
        }
        observation = observations["rounds"][round_index - 1]
        if int(observation["round"]) != round_index:
            raise ValueError("BK15 observation round mismatch")
        votes = observation["votes"]
        if set(votes) != set(by_id):
            raise ValueError("BK15 vote population mismatch")
        round_counts = {arm: 0 for arm in (*ARMS, "tie")}
        decoded = {}
        for source_id, label in votes.items():
            arm = "tie" if label == "tie" else by_id[source_id][label]
            if arm not in round_counts:
                raise ValueError("invalid BK15 blind vote")
            decoded[source_id] = arm
            round_counts[arm] += 1
            counts[arm] += 1
        maximum = max(round_counts.values())
        winners = [
            arm for arm, count in round_counts.items() if count == maximum
        ]
        winner = winners[0] if len(winners) == 1 else "tie"
        round_wins[winner] += 1
        pairwise_winners: dict[str, str] = {}
        for control in ARMS[1:]:
            candidate_count = round_counts[ARMS[0]]
            control_count = round_counts[control]
            pairwise = (
                ARMS[0]
                if candidate_count > control_count
                else control
                if control_count > candidate_count
                else "tie"
            )
            pairwise_round_wins[control][pairwise] += 1
            pairwise_winners[control] = pairwise
        decoded_rounds.append(
            {
                "round": round_index,
                "decoded_votes": decoded,
                "counts": round_counts,
                "winner": winner,
                "bk10_pairwise_winners": pairwise_winners,
            }
        )
    gate = config["blind_gate"]
    total_votes = sum(counts.values())
    candidate_share = counts[ARMS[0]] / total_votes
    pairwise_shares = {}
    for control in ARMS[1:]:
        pairwise_total = counts[ARMS[0]] + counts[control]
        pairwise_shares[control] = (
            counts[ARMS[0]] / pairwise_total if pairwise_total else 0.0
        )
    severe_count = int(observations["confirmed_severe_artifact_count"])
    passed = bool(
        severe_count == 0
        and round_wins[ARMS[0]]
        >= int(gate["minimum_bk10_overall_round_wins"])
        and candidate_share >= float(gate["minimum_bk10_total_choice_share"])
        and all(
            pairwise_round_wins[control][ARMS[0]]
            >= int(gate["minimum_bk10_round_wins_vs_each_control"])
            for control in ARMS[1:]
        )
        and all(
            pairwise_shares[control]
            >= float(gate["minimum_bk10_choice_share_vs_each_control"])
            for control in ARMS[1:]
        )
    )
    core = {
        "schema": "neuro_film.u5_r2bk15_blind_result.v1",
        "experiment_id": config["experiment_id"],
        "build_receipt_sha256": sha256_file(receipt_path),
        "blind_observations_sha256": sha256_file(observations_path),
        "blind_mapping_sha256": mapping_hashes,
        "rounds": decoded_rounds,
        "counts": counts,
        "round_wins": round_wins,
        "bk10_pairwise_round_wins": pairwise_round_wins,
        "bk10_total_choice_share": candidate_share,
        "bk10_choice_share_vs_controls": pairwise_shares,
        "confirmed_severe_artifact_count": severe_count,
        "blind_gate_passed": passed,
        "development_challenger_retained": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    core["stable_evidence_id"] = canonical_sha256(core)
    return core


__all__ = [
    "ARMS",
    "adjudicate_blind",
    "build_blind_sheets",
]
