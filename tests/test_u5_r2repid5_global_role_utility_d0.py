from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import scripts.run_u5_r2repid5_global_role_utility_d0 as module

ROOT = Path(__file__).resolve().parents[1]


def _payload() -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["name", "left", "right", "mos"])
    writer.writeheader()
    roles = ["tiff16_a", "tiff16_b", "tiff16_c", "tiff16_d", "tiff16_e", "tiff16_f"]
    for scene_index in range(20):
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1 :]:
                writer.writerow(
                    {
                        "name": f"scene-{scene_index}",
                        "left": left,
                        "right": right,
                        "mos": 0.75,
                    }
                )
    return stream.getvalue().encode()


def test_complete_scene_parser_and_role_utility_are_transitive() -> None:
    roles = ("tiff16_a", "tiff16_b", "tiff16_c", "tiff16_d", "tiff16_e", "tiff16_f")
    scenes = module._load_complete_scenes(_payload(), roles, 15)
    utilities = module._fit_utilities(list(scenes.values()), roles)
    score = module._score(list(scenes.values()), utilities)
    assert len(scenes) == 20
    assert score["accuracy"] == 1.0
    assert score["top_role"] == "tiff16_a"
    assert min(score["top_role_opponent_win_rates"].values()) == 1.0


def test_duplicate_pair_makes_scene_incomplete() -> None:
    roles = ("tiff16_a", "tiff16_b", "tiff16_c", "tiff16_d", "tiff16_e", "tiff16_f")
    rows = list(csv.DictReader(io.StringIO(_payload().decode())))
    rows[0]["right"] = rows[1]["right"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["name", "left", "right", "mos"])
    writer.writeheader()
    writer.writerows(rows)
    scenes = module._load_complete_scenes(stream.getvalue().encode(), roles, 15)
    assert "scene-0" not in scenes


def test_contract_is_metadata_only_and_parent_bound() -> None:
    contract = json.loads(
        (ROOT / "configs/u5_r2repid5_global_role_utility_d0_v1.json").read_text()
    )
    assert contract["protocol"]["image_member_reads"] == 0
    assert contract["protocol"]["operator_fits"] == 0
    assert contract["parent"]["required_decision"] == (
        "close_exact_repid_shared_logit_affine_population_prior"
    )


def test_corrected_contract_binds_exact_source_audited_roles() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u5_r2repid5a_corrected_global_role_utility_d0_v1.json"
        ).read_text()
    )
    assert contract["source"]["roles"] == [
        "original",
        "tiff16_a",
        "tiff16_b",
        "tiff16_c",
        "tiff16_d",
        "tiff16_e",
    ]
    assert contract["correction"]["only_change"] == (
        "replace nonexistent tiff16_f with source-audited original role"
    )
