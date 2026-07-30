from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.color_match import (
    ReferenceMatchContractError,
    build_reference_composition,
    fit_reference_look,
    validate_reference_composition,
)
from src.inference import sha256_file
from src.preprocess import SourceProfile, WorkingImage


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _reference_recipe():
    pixels = np.asarray(
        [
            [[0.08, 0.12, 0.2], [0.3, 0.2, 0.1]],
            [[0.7, 0.6, 0.4], [0.9, 0.8, 0.7]],
        ],
        dtype=np.float32,
    )
    return fit_reference_look(
        WorkingImage(
            pixels=pixels,
            working_space="linear_srgb",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("assumed_srgb", "test fixture"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=16,
            source_path=Path("fixture.png"),
        )
    )


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_default_plan_is_identity_until_reference_algorithm_is_promoted() -> None:
    recipe = _reference_recipe()
    first = build_reference_composition(recipe)
    second = build_reference_composition(recipe)

    assert first == second
    assert first.color_owner == "identity"
    assert first.reference_color_status == "identity-fallback"
    assert first.research_baseline_override is False
    assert first.claim_ceiling == "identity"
    assert first.output_label == "identity"
    assert first.execution_order == ("identity_color",)
    assert first.film_color_profile_id is None
    assert first.film_effects is None
    assert first.film_stock_identity_claimed is False
    validate_reference_composition(first)


def test_explicit_research_plan_owns_exactly_one_colour_stage() -> None:
    plan = build_reference_composition(
        _reference_recipe(),
        allow_research_baseline=True,
    )

    assert plan.color_owner == "reference-look"
    assert plan.reference_color_status == "research-baseline"
    assert plan.research_baseline_override is True
    assert plan.claim_ceiling == "reference-look"
    assert plan.output_label == "reference-look"
    assert plan.execution_order == ("reference_color",)
    validate_reference_composition(plan)


def test_film_profile_can_supply_effect_provenance_but_not_colour_identity() -> None:
    plan = build_reference_composition(
        _reference_recipe(),
        include_film_effects=True,
        film_profile=_profile(),
        film_profile_sha256=sha256_file(PROFILE_PATH),
        allow_research_baseline=True,
    )

    assert plan.execution_order == ("reference_color", "film_effects")
    assert plan.output_label == "reference-look+film-effects"
    assert plan.film_color_profile_id is None
    assert plan.film_stock_identity_claimed is False
    assert plan.film_effects is not None
    assert plan.film_effects.profile_id == "safe-rich-v1"
    validate_reference_composition(plan)


@pytest.mark.parametrize(
    ("profile", "profile_hash", "message"),
    [
        (None, None, "required"),
        (_profile(), "not-a-hash", "SHA-256"),
    ],
)
def test_effect_binding_requires_a_validated_explicit_profile_identity(
    profile: dict | None,
    profile_hash: str | None,
    message: str,
) -> None:
    with pytest.raises(ReferenceMatchContractError, match=message):
        build_reference_composition(
            _reference_recipe(),
            include_film_effects=True,
            film_profile=profile,
            film_profile_sha256=profile_hash,
            allow_research_baseline=True,
        )


def test_unused_profile_inputs_and_hidden_film_colour_claim_fail_closed() -> None:
    recipe = _reference_recipe()
    with pytest.raises(ReferenceMatchContractError, match="require"):
        build_reference_composition(
            recipe,
            film_profile=_profile(),
            film_profile_sha256=sha256_file(PROFILE_PATH),
        )

    plan = build_reference_composition(recipe)
    with pytest.raises(ReferenceMatchContractError, match="silently stacked"):
        validate_reference_composition(
            replace(plan, film_color_profile_id="safe-rich-v1")
        )
    with pytest.raises(ReferenceMatchContractError, match="stock identity"):
        validate_reference_composition(
            replace(plan, film_stock_identity_claimed=True)
        )


def test_film_effects_cannot_hide_unavailable_reference_colour() -> None:
    with pytest.raises(ReferenceMatchContractError, match="separate"):
        build_reference_composition(
            _reference_recipe(),
            include_film_effects=True,
            film_profile=_profile(),
            film_profile_sha256=sha256_file(PROFILE_PATH),
        )
