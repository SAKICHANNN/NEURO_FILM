from __future__ import annotations

import numpy as np
import pytest

from src.color_match import (
    ContextInvariancePolicy,
    ReferenceMatchContractError,
    evaluate_context_invariance_outputs,
    evaluate_recipe_batch_context_invariance,
    evaluate_recipe_context_invariance,
    fit_reference_look,
    make_context_invariance_probes,
    validate_context_invariance_policy,
)
from src.preprocess import WorkingImage


def _replace_pixels(source: WorkingImage, pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=source.working_space,
        transfer_state=source.transfer_state,
        source_transfer_state=source.source_transfer_state,
        source_profile=source.source_profile,
        hdr_metadata=dict(source.hdr_metadata),
        orientation_applied=source.orientation_applied,
        alpha_policy=source.alpha_policy,
        bit_depth_in=source.bit_depth_in,
        source_path=source.source_path,
    )


def test_context_probes_share_exact_chart_but_not_surroundings() -> None:
    first, second, region = make_context_invariance_probes()

    assert np.array_equal(first.pixels[region], second.pixels[region])
    assert not np.array_equal(first.pixels, second.pixels)


def test_identical_shared_colour_outputs_pass_context_gate() -> None:
    first, second, region = make_context_invariance_probes()

    result = evaluate_context_invariance_outputs(
        first,
        second,
        shared_region=region,
    )

    assert result.passed is True
    assert result.reasons == ()
    assert result.delta_e76_maximum == 0.0


def test_shared_colour_drift_fails_all_tail_levels() -> None:
    first, second, region = make_context_invariance_probes()
    shifted = second.pixels.copy()
    shifted[region] = np.clip(shifted[region] * 0.65 + 0.12, 0.0, 1.0)

    result = evaluate_context_invariance_outputs(
        first,
        _replace_pixels(second, shifted),
        shared_region=region,
    )

    assert result.passed is False
    assert result.reasons == (
        "shared-colour-median-drift",
        "shared-colour-p95-drift",
        "shared-colour-maximum-drift",
    )


def test_safe_lab_recipe_exposes_context_dependent_batch_drift() -> None:
    first, _, _ = make_context_invariance_probes()
    recipe = fit_reference_look(first)

    one = evaluate_recipe_context_invariance(recipe)
    batch = evaluate_recipe_batch_context_invariance([recipe, recipe])

    assert one.passed is False
    assert one.delta_e76_median > 20.0
    assert batch.samples == (one, one)
    assert batch.passed_recipe_count == 0
    assert batch.failed_recipe_count == 2
    assert batch.passed is False


def test_context_policy_rejects_invalid_or_unordered_limits() -> None:
    with pytest.raises(ReferenceMatchContractError, match="ordered"):
        validate_context_invariance_policy(
            ContextInvariancePolicy(
                max_delta_e76_median=2.0,
                max_delta_e76_p95=1.0,
                max_delta_e76=3.0,
            )
        )
    with pytest.raises(ReferenceMatchContractError, match="non-negative"):
        validate_context_invariance_policy(
            ContextInvariancePolicy(max_delta_e76_median=-0.1)
        )
