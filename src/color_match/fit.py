"""Fit deterministic reference-look recipes from product WorkingImage input."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from src.color_engine import in_working_gamut, linear_rgb_to_lab
from src.preprocess.types import WorkingImage

from .contracts import (
    REFERENCE_LOOK_ALGORITHM_ID,
    REFERENCE_LOOK_RECIPE_SCHEMA_ID,
    SUPPORTED_WORKING_SPACES,
    ReferenceLookPolicy,
    ReferenceLookRecipe,
    ReferenceMatchContractError,
    compute_recipe_id,
    validate_policy,
    validate_recipe,
)


def _validate_reference(reference: WorkingImage) -> None:
    if not isinstance(reference, WorkingImage):
        raise ReferenceMatchContractError("reference must be WorkingImage")
    if reference.transfer_state != "display_linear":
        raise ReferenceMatchContractError(
            "reference matching currently requires display-linear SDR"
        )
    if reference.working_space not in SUPPORTED_WORKING_SPACES:
        raise ReferenceMatchContractError("reference working space is unsupported")
    reference_lab = linear_rgb_to_lab(
        reference.pixels,
        working_space=reference.working_space,
    )
    if not in_working_gamut(
        reference_lab,
        working_space=reference.working_space,
        tolerance=2e-6,
    ).all():
        raise ReferenceMatchContractError(
            "reference pixels are outside the declared working gamut"
        )


def _reference_pixel_sha256(reference: WorkingImage) -> str:
    header = {
        "shape": [int(value) for value in reference.pixels.shape],
        "working_space": reference.working_space,
        "transfer_state": reference.transfer_state,
    }
    digest = hashlib.sha256(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    little_endian = np.asarray(reference.pixels, dtype="<f4", order="C")
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def fit_reference_look(
    reference: WorkingImage,
    *,
    policy: ReferenceLookPolicy | None = None,
) -> ReferenceLookRecipe:
    """Fit one replayable safe-Lab target from one uploaded reference image."""

    _validate_reference(reference)
    selected_policy = policy or ReferenceLookPolicy()
    validate_policy(selected_policy)
    lab = linear_rgb_to_lab(reference.pixels, working_space=reference.working_space)
    flattened = lab.reshape(-1, 3)
    destination_mean = flattened.mean(axis=0, dtype=np.float64)
    destination_std = np.maximum(flattened.std(axis=0, dtype=np.float64), 1e-3)
    provisional = ReferenceLookRecipe(
        schema_id=REFERENCE_LOOK_RECIPE_SCHEMA_ID,
        algorithm_id=REFERENCE_LOOK_ALGORITHM_ID,
        recipe_id="0" * 64,
        claim_ceiling="reference-look",
        evidence_grade="deterministic-statistical-baseline",
        reference_working_space=reference.working_space,
        reference_transfer_state=reference.transfer_state,
        reference_shape=tuple(int(value) for value in reference.pixels.shape),
        reference_pixel_sha256=_reference_pixel_sha256(reference),
        destination_lab_mean=tuple(float(value) for value in destination_mean),
        destination_lab_std=tuple(float(value) for value in destination_std),
        policy=selected_policy,
    )
    recipe = ReferenceLookRecipe(
        **{
            **provisional.to_dict(),
            "recipe_id": compute_recipe_id(provisional),
            "policy": selected_policy,
        }
    )
    validate_recipe(recipe)
    return recipe


__all__ = ["fit_reference_look"]
