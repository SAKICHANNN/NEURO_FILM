"""Blind comparison of fixed FilmMatch, B0, and AO6 render inventories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file


ARMS = ("fixed_b0", "fixed_ao6_t15_c35", "filmmatch_cap070")


def _load_exact(root: Path, binding: dict[str, str]) -> Any:
    path = root / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"identity drift: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_filmmatch_blind_sheets(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    decision = _load_exact(root, config["parents"]["filmmatch_decision"])
    if decision["decision"] != (
        "pass_fresh_nine_camera_ood_safety_and_style_"
        "development_confirmation"
    ):
        raise ValueError("FilmMatch fresh OOD decision is not eligible")
    manifest = _load_exact(root, config["population"]["manifest"])
    baseline = _load_exact(root, config["parents"]["baseline_report"])
    filmmatch = _load_exact(root, config["parents"]["filmmatch_report"])
    baseline_by_key = {
        (row["source_id"], row["arm_id"]): row for row in baseline["rows"]
    }
    filmmatch_by_id = {row["id"]: row for row in filmmatch["rows"]}
    source_ids = [str(row["id"]) for row in manifest]
    if (
        len(source_ids) != 9
        or len(set(source_ids)) != 9
        or set(source_ids) != set(filmmatch_by_id)
    ):
        raise ValueError("blind population identity mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = ("A", "B", "C")
    artifacts = []
    for round_index in (1, 2, 3):
        mapping = []
        tiles = []
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
                    **dict(zip(labels, order, strict=True)),
                }
            )
            source_path = root / str(source_row["decoded_path"])
            if sha256_file(source_path) != source_row["decoded_sha256"]:
                raise ValueError("blind source identity drift")
            arm_paths = {
                "fixed_b0": (
                    root
                    / config["parents"]["baseline_render_root"]
                    / baseline_by_key[(source_id, "fixed_b0")][
                        "output_path"
                    ]
                ),
                "fixed_ao6_t15_c35": (
                    root
                    / config["parents"]["baseline_render_root"]
                    / baseline_by_key[
                        (source_id, "fixed_ao6_colour_only_t15_c35")
                    ]["output_path"]
                ),
                "filmmatch_cap070": (
                    root
                    / config["parents"]["filmmatch_render_root"]
                    / filmmatch_by_id[source_id]["candidate_path"]
                ),
            }
            expected = {
                "fixed_b0": baseline_by_key[(source_id, "fixed_b0")][
                    "output_sha256"
                ],
                "fixed_ao6_t15_c35": baseline_by_key[
                    (source_id, "fixed_ao6_colour_only_t15_c35")
                ]["output_sha256"],
                "filmmatch_cap070": filmmatch_by_id[source_id][
                    "candidate_output_sha256"
                ],
            }
            for arm_id, path in arm_paths.items():
                if sha256_file(path) != expected[arm_id]:
                    raise ValueError("blind render identity drift")
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
                source.thumbnail((430, 280), Image.Resampling.LANCZOS)
            candidates = []
            for arm_id in order:
                with Image.open(arm_paths[arm_id]) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((430, 280), Image.Resampling.LANCZOS)
                candidates.append(image)
            tile = Image.new("RGB", (1760, 330), "white")
            draw = ImageDraw.Draw(tile)
            font = ImageFont.load_default()
            draw.text((8, 5), source_id, fill="black", font=font)
            for index, (label, image) in enumerate(
                zip(("Source", *labels), (source, *candidates), strict=True)
            ):
                x = 8 + index * 438
                draw.text((x, 24), label, fill="black", font=font)
                tile.paste(image, (x, 44))
            tiles.append(tile)
        sheet = Image.new(
            "RGB", (1760, len(tiles) * 330 + 32), "white"
        )
        ImageDraw.Draw(sheet).text(
            (8, 8),
            f"FilmMatch fixed baseline blind round {round_index}",
            fill="black",
            font=ImageFont.load_default(),
        )
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (0, 32 + index * 330))
        sheet_path = output_dir / f"blind_round_{round_index}.png"
        mapping_path = (
            output_dir / f"blind_round_{round_index}_mapping.json"
        )
        sheet.save(sheet_path, "PNG", compress_level=6)
        mapping_path.write_text(
            json.dumps(mapping, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "round": round_index,
                "sheet": sheet_path.name,
                "sheet_sha256": sha256_file(sheet_path),
                "mapping": mapping_path.name,
                "mapping_sha256": sha256_file(mapping_path),
            }
        )
    receipt = {
        "schema": "neuro_film.u5_r2ax17_filmmatch_blind_build.v1",
        "experiment_id": config["experiment_id"],
        "source_ids": source_ids,
        "arms": list(ARMS),
        "artifacts": artifacts,
    }
    receipt["stable_evidence_id"] = canonical_sha256(receipt)
    path = output_dir / "build_receipt.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def adjudicate_filmmatch_blind(
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
        raise ValueError("blind build receipt experiment mismatch")
    expected_artifacts = {
        int(row["round"]): row for row in receipt.get("artifacts", [])
    }
    if set(expected_artifacts) != {1, 2, 3}:
        raise ValueError("blind build receipt round mismatch")
    counts = {arm: 0 for arm in (*ARMS, "tie")}
    round_wins = {arm: 0 for arm in (*ARMS, "tie")}
    pairwise_round_wins = {
        "fixed_ao6_t15_c35": 0,
        "filmmatch_cap070": 0,
        "tie": 0,
    }
    decoded_rounds = []
    mapping_sha256 = []
    for round_index in (1, 2, 3):
        mapping_path = build_dir / f"blind_round_{round_index}_mapping.json"
        mapping_hash = sha256_file(mapping_path)
        if (
            mapping_hash
            != expected_artifacts[round_index]["mapping_sha256"]
        ):
            raise ValueError("blind mapping identity drift")
        mapping_sha256.append(mapping_hash)
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        by_id = {row["source_id"]: row for row in mapping}
        observation_round = observations["rounds"][round_index - 1]
        if int(observation_round["round"]) != round_index:
            raise ValueError("blind observation round mismatch")
        votes = observation_round["votes"]
        if set(votes) != set(by_id):
            raise ValueError("blind vote population mismatch")
        round_counts = {arm: 0 for arm in (*ARMS, "tie")}
        decoded = {}
        for source_id, label in votes.items():
            arm = "tie" if label == "tie" else by_id[source_id][label]
            if arm not in round_counts:
                raise ValueError("invalid blind vote")
            decoded[source_id] = arm
            round_counts[arm] += 1
            counts[arm] += 1
        maximum = max(round_counts.values())
        winners = [
            arm for arm, count in round_counts.items() if count == maximum
        ]
        winner = winners[0] if len(winners) == 1 else "tie"
        round_wins[winner] += 1
        fm = round_counts["filmmatch_cap070"]
        ao6 = round_counts["fixed_ao6_t15_c35"]
        pairwise = (
            "filmmatch_cap070"
            if fm > ao6
            else "fixed_ao6_t15_c35"
            if ao6 > fm
            else "tie"
        )
        pairwise_round_wins[pairwise] += 1
        decoded_rounds.append(
            {
                "round": round_index,
                "decoded_votes": decoded,
                "counts": round_counts,
                "winner": winner,
                "filmmatch_vs_ao6_winner": pairwise,
            }
        )
    gates = config["blind_gate"]
    total_votes = sum(counts.values())
    fm_share = counts["filmmatch_cap070"] / total_votes
    pairwise_total = (
        counts["filmmatch_cap070"] + counts["fixed_ao6_t15_c35"]
    )
    pairwise_share = (
        counts["filmmatch_cap070"] / pairwise_total
        if pairwise_total
        else 0.0
    )
    severe_count = int(observations["confirmed_severe_artifact_count"])
    passed = bool(
        severe_count == 0
        and round_wins["filmmatch_cap070"]
        >= int(gates["minimum_overall_round_wins"])
        and fm_share >= float(gates["minimum_total_choice_share"])
        and pairwise_round_wins["filmmatch_cap070"]
        >= int(gates["minimum_round_wins_vs_ao6"])
        and pairwise_share
        >= float(gates["minimum_choice_share_vs_ao6"])
    )
    core = {
        "schema": "neuro_film.u5_r2ax17_filmmatch_blind_result.v1",
        "experiment_id": config["experiment_id"],
        "build_receipt_sha256": sha256_file(receipt_path),
        "blind_observations_sha256": sha256_file(observations_path),
        "blind_mapping_sha256": mapping_sha256,
        "rounds": decoded_rounds,
        "counts": counts,
        "round_wins": round_wins,
        "filmmatch_vs_ao6_round_wins": pairwise_round_wins,
        "filmmatch_total_choice_share": fm_share,
        "filmmatch_choice_share_vs_ao6": pairwise_share,
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
    "adjudicate_filmmatch_blind",
    "build_filmmatch_blind_sheets",
]
