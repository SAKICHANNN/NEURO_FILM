from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.filmstylesafe_r1b import (
    FilmStyleSafeR1BError,
    audit_suite_leakage,
    load_r1b_contract,
    validate_suite_member,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "filmstylesafe_r1b_contract_v1.json"


def _member(**overrides):
    base = {
        "member_id": "m1",
        "role": "real_failure",
        "split": "A0",
        "parent_scene_id": "scene-1",
        "source_id": "src-1",
        "transform_family": "safe_lab_global",
        "failure_family": "neon_chroma_island_or_highlight_speckle",
        "origin": "u41_provisional_gold",
        "exact_hash": "a" * 64,
        "perceptual_hash": "b" * 64,
    }
    base.update(overrides)
    return base


def test_load_r1b_contract() -> None:
    contract = load_r1b_contract(CONTRACT)
    assert contract["contract_id"] == "kmcfm.filmstylesafe-r1b.v1"
    assert contract["authorization"]["hidden_split_population"] is False


def test_validate_real_failure_member() -> None:
    contract = load_r1b_contract(CONTRACT)
    out = validate_suite_member(_member(), contract)
    assert out["passed"] is True


def test_a0_only_origin_cannot_enter_a1() -> None:
    contract = load_r1b_contract(CONTRACT)
    with pytest.raises(FilmStyleSafeR1BError, match="A0-only"):
        validate_suite_member(_member(split="A1"), contract)


def test_synthetic_requires_operator_card() -> None:
    contract = load_r1b_contract(CONTRACT)
    with pytest.raises(FilmStyleSafeR1BError, match="operator_card"):
        validate_suite_member(_member(role="synthetic_failure", origin="synthetic_lab"), contract)
    card = {
        "operator_id": "op-speckle-v1",
        "parameter_hash": "c" * 64,
        "transform_family": "safe_lab_global",
        "failure_family": "neon_chroma_island_or_highlight_speckle",
        "seed": 7,
        "input_hash": "d" * 64,
    }
    out = validate_suite_member(
        _member(role="synthetic_failure", origin="synthetic_lab", operator_card=card),
        contract,
    )
    assert out["role"] == "synthetic_failure"


def test_strength_control_and_id11_regression() -> None:
    contract = load_r1b_contract(CONTRACT)
    strength = validate_suite_member(
        _member(
            member_id="s53",
            role="strength_control",
            transform_family="owner_anchor_53_55_56_strength_path",
            failure_family="none",
            origin="u42_owner_anchor_replays",
            owner_scheme_id="53",
        ),
        contract,
    )
    assert strength["passed"] is True
    reg = validate_suite_member(
        _member(
            member_id="id11",
            role="regression_case",
            origin="known_id11_red_speckle",
            failure_family="id11_red_speckle_regression",
            transform_family="owner_anchor_53_55_56_strength_path",
        ),
        contract,
    )
    assert reg["passed"] is True


def test_suite_leakage_rejects_shared_parent_and_family() -> None:
    a0 = _member(member_id="a0", split="A0", parent_scene_id="shared", exact_hash="1" * 64)
    a1 = _member(
        member_id="a1",
        split="A1",
        parent_scene_id="shared",
        origin="synthetic_lab",
        exact_hash="2" * 64,
        perceptual_hash="3" * 64,
    )
    with pytest.raises(FilmStyleSafeR1BError, match="parent_scene_id"):
        audit_suite_leakage([a0, a1])
    a1b = _member(
        member_id="a1b",
        split="A1",
        parent_scene_id="other",
        source_id="src-2",
        origin="synthetic_lab",
        exact_hash="4" * 64,
        perceptual_hash="5" * 64,
        transform_family="safe_lab_global",
        failure_family="neon_chroma_island_or_highlight_speckle",
    )
    with pytest.raises(FilmStyleSafeR1BError, match="family pair"):
        audit_suite_leakage([a0, a1b])
