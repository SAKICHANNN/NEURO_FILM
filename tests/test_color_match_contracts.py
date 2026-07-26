from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.color_match import (
    REFERENCE_LOOK_RECIPE_SCHEMA_ID,
    ReferenceLookPolicy,
    ReferenceMatchContractError,
    fit_reference_look,
    recipe_from_json,
    recipe_to_json,
    validate_recipe,
)
from src.preprocess.types import SourceProfile, WorkingImage


def _working(
    pixels: np.ndarray,
    *,
    working_space: str = "linear_srgb",
    transfer_state: str = "display_linear",
) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "test fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.png"),
    )


def _reference() -> WorkingImage:
    rng = np.random.default_rng(27001)
    pixels = rng.uniform(0.03, 0.88, size=(17, 19, 3)).astype(np.float32)
    return _working(pixels)


def test_fit_reference_recipe_is_deterministic_and_replayable() -> None:
    reference = _reference()
    first = fit_reference_look(reference)
    second = fit_reference_look(reference)
    assert first == second
    assert first.schema_id == REFERENCE_LOOK_RECIPE_SCHEMA_ID
    assert first.claim_ceiling == "reference-look"
    assert first.evidence_grade == "deterministic-statistical-baseline"
    encoded = recipe_to_json(first)
    assert recipe_to_json(second) == encoded
    assert recipe_from_json(encoded) == first


def test_recipe_id_covers_policy_and_reference_descriptor() -> None:
    first = fit_reference_look(_reference())
    weaker = fit_reference_look(
        _reference(),
        policy=replace(ReferenceLookPolicy(), strength=0.4),
    )
    changed_reference = _reference()
    changed_reference.pixels[0, 0, 0] += np.float32(0.01)
    changed = fit_reference_look(changed_reference)
    assert len({first.recipe_id, weaker.recipe_id, changed.recipe_id}) == 3
    assert first.reference_pixel_sha256 != changed.reference_pixel_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_id", "unknown", "unsupported reference-look recipe schema"),
        ("claim_ceiling", "calibrated-reference", "claim ceiling mismatch"),
        ("recipe_id", "0" * 64, "does not match canonical payload"),
    ],
)
def test_recipe_validation_rejects_tampering(
    field: str,
    value: object,
    message: str,
) -> None:
    recipe = fit_reference_look(_reference())
    tampered = replace(recipe, **{field: value})
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_recipe(tampered)


def test_recipe_json_rejects_unknown_fields() -> None:
    encoded = recipe_to_json(fit_reference_look(_reference()))
    payload = encoded.rstrip().removesuffix("}")
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        recipe_from_json(payload + ', "unexpected": 1}')


@pytest.mark.parametrize("transfer_state", ["scene_linear", "display_referred", "unknown"])
def test_fit_fails_closed_outside_display_linear_sdr(transfer_state: str) -> None:
    with pytest.raises(ReferenceMatchContractError, match="display-linear SDR"):
        fit_reference_look(_working(_reference().pixels, transfer_state=transfer_state))


def test_fit_rejects_reference_outside_declared_gamut() -> None:
    pixels = _reference().pixels.copy()
    pixels[2, 3, 1] = np.float32(1.2)
    with pytest.raises(ReferenceMatchContractError, match="outside the declared working gamut"):
        fit_reference_look(_working(pixels))


@pytest.mark.parametrize(
    "policy",
    [
        ReferenceLookPolicy(strength=-0.1),
        ReferenceLookPolicy(gamut_mode="clip"),
        ReferenceLookPolicy(gamut_iterations=0),
        ReferenceLookPolicy(max_chroma_gain=float("nan")),
    ],
)
def test_policy_validation_fails_closed(policy: ReferenceLookPolicy) -> None:
    with pytest.raises(ReferenceMatchContractError):
        fit_reference_look(_reference(), policy=policy)
