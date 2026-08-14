from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.analytic_y_chromaticity_third_confirmation import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u5_r2cb74_analytic_y_chromaticity_third_confirmation_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cb74_keeps_cb51_operator_and_automatic_gates_exact() -> None:
    current = load_contract(CONTRACT)
    previous = json.loads(
        (
            ROOT
            / "configs/u5_r2cb51_analytic_y_chromaticity_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "dose_grid",
        "lstar_order_epsilon",
        "minimum_valid_fraction",
        "fraction_knots",
        "maximum_fraction_slope",
    ):
        assert current["operator"][key] == previous["operator"][key]
    assert current["automatic_gates"] == previous["automatic_gates"]
    assert current["parents"]["cb51_required_decision"].startswith("pass_cb51_")


def test_cb74_binds_exact_p7h_nine_camera_population() -> None:
    current = load_contract(CONTRACT)
    population = current["population"]
    source_decision_path = ROOT / population["decision_path"]
    source_decision = json.loads(source_decision_path.read_text(encoding="utf-8"))
    assert _sha(source_decision_path) == population["decision_sha256"]
    assert source_decision["status"] == population["required_status"]
    source_parent_path = ROOT / source_decision["source_parent_path"]
    source_parent = json.loads(source_parent_path.read_text(encoding="utf-8"))
    assert _sha(source_parent_path) == source_decision["source_parent_sha256"]
    assert source_parent["status"] == source_decision["source_parent_required_status"]
    manifest_path = ROOT / population["manifest_path"]
    assert _sha(manifest_path) == population["manifest_sha256"]
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_by_id = {row["id"]: row for row in rows}
    selected = [row_by_id[source_id] for source_id in population["included_source_ids"]]
    assert len(selected) == population["source_count_exact"] == 9
    assert len({row["make"] for row in selected}) == population["camera_make_count_exact"] == 9
    assert all((ROOT / row["decoded_path"]).is_file() for row in selected)


def test_cb74_blind_gate_requires_two_thirds_of_all_choices() -> None:
    current = load_contract(CONTRACT)
    blind = current["blind_protocol"]
    assert blind["rounds"] == 3
    assert blind["minimum_candidate_round_wins"] == 2
    assert blind["minimum_candidate_aggregate_choices"] == 18
    assert blind["minimum_candidate_source_majorities"] == 5
