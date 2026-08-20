from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_u5_r2repid2_shared_operator_roles import (
    _canonical_stem,
    eligible_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_binds_exact_parent_evidence_file() -> None:
    contract = json.loads(
        (ROOT / "configs/u5_r2repid2_shared_operator_roles_v1.json").read_text()
    )
    payload = (ROOT / contract["parent"]["evidence_path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == contract["parent"]["evidence_sha256"]


def _scene(name: str, geometry: str = "same") -> list[dict[str, str]]:
    roles = ["original", "tiff16_a", "tiff16_b", "tiff16_c", "tiff16_d", "tiff16_e"]
    rows = []
    for left_index, left in enumerate(roles):
        for right_index, right in enumerate(roles[left_index + 1 :], left_index + 1):
            rows.append(
                {
                    "name": name,
                    "left": left,
                    "right": right,
                    "mos": "0.0" if right_index > left_index else "1.0",
                    "CropTop_left": geometry,
                    "CropTop_right": geometry,
                }
            )
    return rows


def test_eligible_rows_select_unique_strong_geometry_matched_extremes() -> None:
    eligible, counters = eligible_rows(
        _scene("scene.jpeg"),
        geometry_fields=["CropTop"],
        minimum_margin=0.2,
        excluded_stems=set(),
    )
    assert len(eligible) == 1
    assert eligible[0]["loser"] == "original"
    assert eligible[0]["winner"] == "tiff16_e"
    assert not counters


def test_prior_scene_and_geometry_mismatch_fail_before_selection() -> None:
    rows = _scene("scene (1).jpeg")
    direct = next(
        row for row in rows if {row["left"], row["right"]} == {"original", "tiff16_e"}
    )
    direct["CropTop_right"] = "different"
    eligible, counters = eligible_rows(
        rows,
        geometry_fields=["CropTop"],
        minimum_margin=0.2,
        excluded_stems={"scene"},
    )
    assert not eligible
    assert counters["geometry_mismatch"] == 1
    assert _canonical_stem("scene (1).jpg") == "scene"
