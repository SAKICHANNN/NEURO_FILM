from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.color_match import (
    BlindAestheticReview,
    KnownOperatorBatchMetrics,
    KnownOperatorSampleMetrics,
    PhotographicSafetyBatchMetrics,
    PhotographicSafetyMetrics,
    adjudicate_promotion,
    evaluate_photographic_probe,
    evaluate_recipe_batch_photographic_safety,
    evaluate_recipe_photographic_safety,
    fit_reference_look,
    make_photographic_probe,
)
from src.preprocess import WorkingImage


def _candidate(source: WorkingImage, pixels: np.ndarray) -> WorkingImage:
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


def _known_metrics(
    *,
    count: int = 12,
    improvement: float = 0.30,
    boundary: float = 0.0,
) -> KnownOperatorBatchMetrics:
    rows = tuple(
        KnownOperatorSampleMetrics(
            sample_id=f"s{index:02d}",
            source_target_delta_e76_median=10.0,
            source_target_delta_e76_p95=15.0,
            candidate_target_delta_e76_median=7.0,
            candidate_target_delta_e76_p95=11.0,
            median_improvement_fraction=improvement,
            candidate_new_boundary_fraction=boundary,
        )
        for index in range(count)
    )
    return KnownOperatorBatchMetrics(
        samples=rows,
        improved_sample_count=count if improvement > 0 else 0,
        regressed_sample_count=count if improvement < 0 else 0,
        improvement_rate=1.0 if improvement > 0 else 0.0,
        median_improvement_fraction=improvement,
        worst_improvement_fraction=improvement,
        maximum_new_boundary_fraction=boundary,
    )


def _passing_safety_sample() -> PhotographicSafetyMetrics:
    return PhotographicSafetyMetrics(
        passed=True,
        reasons=(),
        neutral_chroma_p95=0.0,
        tone_reversal_fraction=0.0,
        tone_largest_reversal_delta_l=0.0,
        tone_plateau_fraction=0.0,
        new_boundary_fraction=0.0,
        semantic_hue_rotation_p95_degrees=0.0,
    )


def _safety_batch(
    sample: PhotographicSafetyMetrics | None = None,
) -> PhotographicSafetyBatchMetrics:
    resolved = sample or _passing_safety_sample()
    return PhotographicSafetyBatchMetrics(
        samples=(resolved,),
        passed_recipe_count=int(resolved.passed),
        failed_recipe_count=int(not resolved.passed),
        maximum_neutral_chroma_p95=resolved.neutral_chroma_p95,
        maximum_tone_reversal_fraction=resolved.tone_reversal_fraction,
        largest_tone_reversal_delta_l=resolved.tone_largest_reversal_delta_l,
        maximum_tone_plateau_fraction=resolved.tone_plateau_fraction,
        maximum_new_boundary_fraction=resolved.new_boundary_fraction,
        maximum_semantic_hue_rotation_p95_degrees=(
            resolved.semantic_hue_rotation_p95_degrees
        ),
    )


def test_identity_probe_passes_all_structural_colour_gates() -> None:
    probe = make_photographic_probe()
    result = evaluate_photographic_probe(probe, _candidate(probe, probe.pixels))

    assert result.passed is True
    assert result.reasons == ()
    assert result.neutral_chroma_p95 < 1.0
    assert result.tone_reversal_fraction == 0.0
    assert result.tone_plateau_fraction == 0.0
    assert result.new_boundary_fraction == 0.0
    assert result.semantic_hue_rotation_p95_degrees == 0.0


def test_recipe_probe_is_deterministic_and_machine_evaluable() -> None:
    probe = make_photographic_probe()
    recipe = fit_reference_look(probe)

    first = evaluate_recipe_photographic_safety(recipe)
    second = evaluate_recipe_photographic_safety(recipe)

    assert first == second
    assert first.passed is True


def test_recipe_batch_retains_all_reference_tails() -> None:
    probe = make_photographic_probe()
    recipes = [fit_reference_look(probe), fit_reference_look(probe)]

    result = evaluate_recipe_batch_photographic_safety(recipes)

    assert len(result.samples) == 2
    assert result.passed_recipe_count == 2
    assert result.failed_recipe_count == 0
    assert result.passed is True


def test_probe_rejects_reversal_plateau_boundary_and_false_colour() -> None:
    probe = make_photographic_probe()
    failed = probe.pixels.copy()
    failed[:4] = failed[:4, ::-1]
    failed[:4, 80:180] = failed[:4, 80:81]
    failed[:4, :, 0] = np.clip(failed[:4, :, 0] + 0.35, 0.0, 1.0)
    failed[4:7] = failed[4:7, :, ::-1]
    failed[7] = 1.0

    result = evaluate_photographic_probe(probe, _candidate(probe, failed))

    assert result.passed is False
    assert "neutral-axis-chroma" in result.reasons
    assert "tone-reversal" in result.reasons
    assert "tone-plateau" in result.reasons
    assert "new-boundary-fraction" in result.reasons
    assert "semantic-hue-rotation" in result.reasons


def test_automated_pass_only_opens_blind_visual_review() -> None:
    decision = adjudicate_promotion(_known_metrics(), _safety_batch())

    assert decision.status == "eligible-for-visual-review"
    assert decision.reasons == ("blind-aesthetic-review-required",)


def test_promotion_requires_blind_preference_and_zero_severe_artifacts() -> None:
    failed = adjudicate_promotion(
        _known_metrics(),
        _safety_batch(),
        visual_review=BlindAestheticReview(
            reviewed_sample_count=12,
            preferred_sample_count=8,
            severe_artifact_count=1,
            blinded=True,
        ),
    )
    passed = adjudicate_promotion(
        _known_metrics(),
        _safety_batch(),
        visual_review=BlindAestheticReview(
            reviewed_sample_count=12,
            preferred_sample_count=8,
            severe_artifact_count=0,
            blinded=True,
        ),
    )

    assert failed.status == "rejected"
    assert failed.reasons == ("visual-severe-artifact-veto",)
    assert passed.status == "promoted"
    assert passed.reasons == ()


def test_known_operator_or_probe_failure_rejects_before_visual_review() -> None:
    weak = adjudicate_promotion(
        _known_metrics(improvement=-0.2),
        _safety_batch(),
    )
    failed_sample = replace(
        _passing_safety_sample(),
        passed=False,
        reasons=("tone-reversal",),
    )
    unsafe = adjudicate_promotion(
        _known_metrics(),
        _safety_batch(failed_sample),
    )

    assert weak.status == "rejected"
    assert "known-operator-improvement-rate" in weak.reasons
    assert "known-operator-tail" in weak.reasons
    assert unsafe.status == "rejected"
    assert unsafe.reasons == ("photographic-safety:tone-reversal",)
