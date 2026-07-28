from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from src.color_match import (
    MAX_REFERENCE_MATCH_BATCH_SOURCES,
    ReferenceMatchContractError,
    ReferenceRenderGuardPolicy,
    fit_reference_look,
    render_reference_batch_guarded,
    render_reference_look,
    render_reference_look_guarded,
)
from src.preprocess import SourceProfile, WorkingImage


def _working(pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
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


def _reference_and_source() -> tuple[WorkingImage, WorkingImage]:
    rng = np.random.default_rng(27002)
    reference = _working(
        rng.uniform(0.02, 0.98, size=(19, 23, 3)).astype(np.float32)
    )
    source = _working(
        rng.uniform(0.15, 0.72, size=(17, 21, 3)).astype(np.float32)
    )
    return reference, source


def test_permissive_guard_delivers_exact_candidate() -> None:
    reference, source = _reference_and_source()
    recipe = fit_reference_look(reference)
    candidate = render_reference_look(recipe, source)
    guarded = render_reference_look_guarded(
        recipe,
        source,
        policy=ReferenceRenderGuardPolicy(
            max_gamut_adjusted_fraction=1.0,
            max_new_boundary_fraction=1.0,
            allow_research_baseline=True,
        ),
    )

    assert guarded.safety.accepted is True
    assert guarded.safety.action == "applied"
    assert guarded.safety.reasons == ()
    assert np.array_equal(guarded.image.pixels, candidate.image.pixels)
    assert guarded.candidate_diagnostics == candidate.diagnostics


def test_zero_tolerance_guard_falls_back_to_unmutated_source() -> None:
    reference, source = _reference_and_source()
    original = source.pixels.copy()
    guarded = render_reference_look_guarded(
        fit_reference_look(reference),
        source,
        policy=ReferenceRenderGuardPolicy(
            max_gamut_adjusted_fraction=0.0,
            max_new_boundary_fraction=0.0,
        ),
    )

    assert guarded.safety.accepted is False
    assert guarded.safety.action == "identity-fallback"
    assert guarded.safety.reasons
    assert np.array_equal(guarded.image.pixels, original)
    assert guarded.image.pixels is not source.pixels
    assert np.array_equal(source.pixels, original)


def test_guarded_batch_is_ordered_and_policy_validation_fails_closed() -> None:
    reference, source = _reference_and_source()
    recipe = fit_reference_look(reference)
    results = render_reference_batch_guarded(
        recipe,
        [source, source],
        policy=ReferenceRenderGuardPolicy(
            max_gamut_adjusted_fraction=1.0,
            max_new_boundary_fraction=1.0,
            allow_research_baseline=True,
        ),
    )
    assert [
        result.candidate_diagnostics.source_index for result in results
    ] == [0, 1]

    with pytest.raises(ReferenceMatchContractError, match="within"):
        render_reference_look_guarded(
            recipe,
            source,
            policy=ReferenceRenderGuardPolicy(
                max_gamut_adjusted_fraction=1.1
            ),
        )


def test_guarded_batch_rejects_more_than_64_sources_before_render() -> None:
    reference, source = _reference_and_source()
    with pytest.raises(
        ReferenceMatchContractError,
        match="supports at most 64 sources",
    ):
        render_reference_batch_guarded(
            fit_reference_look(reference),
            [source] * (MAX_REFERENCE_MATCH_BATCH_SOURCES + 1),
        )


def test_default_guard_blocks_unpromoted_research_algorithm() -> None:
    reference, source = _reference_and_source()
    guarded = render_reference_look_guarded(
        fit_reference_look(reference),
        source,
    )

    assert guarded.safety.accepted is False
    assert guarded.safety.action == "identity-fallback"
    assert "algorithm-not-promoted" in guarded.safety.reasons
    assert guarded.safety.research_baseline_override is False
    assert np.array_equal(guarded.image.pixels, source.pixels)


def test_research_override_is_explicit_in_decision() -> None:
    reference, source = _reference_and_source()
    guarded = render_reference_look_guarded(
        fit_reference_look(reference),
        source,
        policy=ReferenceRenderGuardPolicy(
            max_gamut_adjusted_fraction=1.0,
            max_new_boundary_fraction=1.0,
            allow_research_baseline=True,
        ),
    )

    assert guarded.safety.accepted is True
    assert guarded.safety.research_baseline_override is True


def test_rejected_candidate_pixels_are_released_before_identity_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import safety

    _, source = _reference_and_source()
    candidate_ref: weakref.ReferenceType[WorkingImage] | None = None
    diagnostics = SimpleNamespace(gamut_adjusted_fraction=0.0)

    def render_candidate(recipe, source_image, **kwargs):
        nonlocal candidate_ref
        assert source_image is source
        candidate_image = _working(source.pixels * np.float32(0.75))
        candidate_ref = weakref.ref(candidate_image)
        return SimpleNamespace(
            image=candidate_image,
            diagnostics=diagnostics,
        )

    def clone_identity(source_image: WorkingImage) -> WorkingImage:
        assert candidate_ref is not None
        assert candidate_ref() is None
        return _working(np.array(source_image.pixels, copy=True))

    monkeypatch.setattr(safety, "render_reference_look", render_candidate)
    monkeypatch.setattr(safety, "_clone_source", clone_identity)

    guarded = render_reference_look_guarded(object(), source)
    assert guarded.safety.accepted is False
    assert guarded.candidate_diagnostics is diagnostics
    assert np.array_equal(guarded.image.pixels, source.pixels)
