from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_10C_INSTALLED_DESKTOP_EXPORT_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaf_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_leaf_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                _leaf_differences(left_item, right_item, f"{path}/{index}")
            )
        return differences
    return [] if left == right else [path]


def test_u7_10c_evidence_preserves_the_frozen_cross_run_failure() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert evidence["automatic_pass"] is False
    assert evidence["status"] == (
        "FAIL_CLOSED_U7_10C_REPORT_COORDINATE_NONDETERMINISM"
    )
    formal = evidence["formal_execution"]
    forward_path = ROOT / formal["forward_report_path"]
    reverse_path = ROOT / formal["reverse_report_path"]
    assert forward_path.stat().st_size == reverse_path.stat().st_size == 13443
    assert _sha256(forward_path) == formal["forward_report_sha256"]
    assert _sha256(reverse_path) == formal["reverse_report_sha256"]
    assert forward_path.read_bytes() != reverse_path.read_bytes()
    forward = json.loads(forward_path.read_text("utf-8"))
    reverse = json.loads(reverse_path.read_text("utf-8"))
    assert forward["source_commit"] == reverse["source_commit"] == (
        "3149db8c70f0a16d3d2b920d8992bc599db64d18"
    )
    assert forward["automatic_pass"] is reverse["automatic_pass"] is True
    assert all(forward["gates"].values())
    assert all(reverse["gates"].values())
    assert _leaf_differences(forward, reverse) == [
        "/interaction/save_dialog/accept/click_screen/0",
        "/interaction/save_dialog/accept/click_screen/1",
        "/stable_identity",
    ]
    assert forward["interaction"]["save_dialog"]["accept"]["click_screen"] == [
        796,
        630,
    ]
    assert reverse["interaction"]["save_dialog"]["accept"]["click_screen"] == [
        874,
        708,
    ]
    assert evidence["cross_run_failure"]["post_result_field_removal_accepted"] is False
    assert evidence["cross_run_failure"]["third_run_executed"] is False


def test_u7_10c_media_visual_and_sources_remain_exact_below_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    formal = evidence["formal_execution"]
    forward_visual = ROOT / formal["forward_visual_path"]
    reverse_visual = ROOT / formal["reverse_visual_path"]
    assert forward_visual.read_bytes() == reverse_visual.read_bytes()
    assert forward_visual.stat().st_size == formal["visual_bytes_each"] == 36202
    assert _sha256(forward_visual) == formal["visual_sha256"]
    signal = evidence["mechanical_signal_below_promotion"]
    assert signal["installed_direct_gui_png_byte_exact_per_run"] is True
    assert signal["installed_direct_gui_recipe_byte_exact_per_run"] is True
    assert signal["gui_strict_replay_png_byte_exact_per_run"] is True
    assert signal["native_save_dialog_pid_and_owner_bound"] is True
    assert signal["owned_process_window_and_formal_root_residue_zero"] is True
    assert not (ROOT / "tmp/u7_10c_formal").exists()
    for row in evidence["source_bindings"].values():
        assert _sha256(ROOT / row["path"]) == row["sha256"]
    assert evidence["manual_visual_review"] == {
        "absolute_path_visible": False,
        "final_status_exact": (
            "Export complete: installed-desktop-velvia.png + "
            "installed-desktop-velvia.recipe.json"
        ),
        "look_approximation_claim_visible": True,
        "not_calibrated_disclaimer_visible": True,
        "overlap_or_crop_observed": False,
        "three_preview_cards_visible": True,
        "velvia_radio_selected_visible": True,
    }
    assert evidence["claim"]["evidence_grade"] == "look-approximation"
    assert evidence["claim"]["calibrated_stock_response"] is False
