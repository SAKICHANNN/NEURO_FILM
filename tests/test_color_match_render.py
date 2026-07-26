from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.color_engine import linear_rgb_to_lab
from src.color_match import (
    ReferenceLookPolicy,
    ReferenceMatchContractError,
    fit_reference_look,
    render_reference_batch,
    render_reference_look,
)
from src.preprocess import convert_linear_rgb
from src.preprocess.types import DecodeWarning, SourceProfile, WorkingImage


def _working(
    pixels: np.ndarray,
    *,
    path: str,
    working_space: str = "linear_srgb",
    transfer_state: str = "display_linear",
) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "test fixture"),
        hdr_metadata={"fixture": path},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path(path),
        warnings=[DecodeWarning("fixture", path)],
    )


def _pixels(seed: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(
        low,
        high,
        size=(23, 29, 3),
    ).astype(np.float32)


def _unprotected_policy() -> ReferenceLookPolicy:
    return ReferenceLookPolicy(
        luma_strength=1.0,
        preserve_luma_detail_strength=0.0,
        chroma_curve_strength=0.0,
        neutral_protect=0.0,
        skin_protect=0.0,
        max_chroma_gain=None,
        max_chroma_boost=None,
        max_chroma_absolute=None,
    )


def test_single_render_is_repeat_exact_and_does_not_mutate_inputs() -> None:
    reference = _working(_pixels(27101, 0.12, 0.82), path="reference.png")
    source = _working(_pixels(27102, 0.08, 0.72), path="source.png")
    reference_before = reference.pixels.tobytes()
    source_before = source.pixels.tobytes()
    recipe = fit_reference_look(reference)
    first = render_reference_look(recipe, source)
    second = render_reference_look(recipe, source)
    assert first.image.pixels.tobytes() == second.image.pixels.tobytes()
    assert first.diagnostics == second.diagnostics
    assert reference.pixels.tobytes() == reference_before
    assert source.pixels.tobytes() == source_before
    assert first.image is not source
    assert first.image.pixels is not source.pixels


def test_reference_statistics_move_source_toward_reference() -> None:
    reference = _working(_pixels(27103, 0.30, 0.88), path="reference.png")
    source = _working(_pixels(27104, 0.04, 0.48), path="source.png")
    recipe = fit_reference_look(reference, policy=_unprotected_policy())
    output = render_reference_look(recipe, source).image
    reference_mean = linear_rgb_to_lab(
        reference.pixels,
        working_space=reference.working_space,
    ).reshape(-1, 3).mean(axis=0)
    source_mean = linear_rgb_to_lab(
        source.pixels,
        working_space=source.working_space,
    ).reshape(-1, 3).mean(axis=0)
    output_mean = linear_rgb_to_lab(
        output.pixels,
        working_space=output.working_space,
    ).reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(output_mean - reference_mean) < np.linalg.norm(
        source_mean - reference_mean
    )


def test_batch_reuses_one_recipe_and_preserves_order_and_metadata() -> None:
    reference = _working(_pixels(27105, 0.10, 0.85), path="reference.png")
    sources = [
        _working(_pixels(27106, 0.05, 0.65), path="a.png"),
        _working(_pixels(27107, 0.15, 0.75), path="b.png"),
        _working(_pixels(27108, 0.20, 0.80), path="c.png"),
    ]
    recipe = fit_reference_look(reference)
    results = render_reference_batch(recipe, sources)
    assert len(results) == 3
    assert [result.image.source_path.name for result in results] == ["a.png", "b.png", "c.png"]
    assert [result.diagnostics.source_index for result in results] == [0, 1, 2]
    assert {result.diagnostics.recipe_id for result in results} == {recipe.recipe_id}
    for index, result in enumerate(results):
        individual = render_reference_look(recipe, sources[index], source_index=index)
        assert result.image.pixels.tobytes() == individual.image.pixels.tobytes()
        assert result.image.hdr_metadata == sources[index].hdr_metadata
        assert result.image.hdr_metadata is not sources[index].hdr_metadata
        assert result.image.warnings == sources[index].warnings
        assert result.image.warnings is not sources[index].warnings


def test_batch_result_is_independent_of_other_source_members() -> None:
    reference = _working(_pixels(27109, 0.12, 0.80), path="reference.png")
    a = _working(_pixels(27110, 0.06, 0.64), path="a.png")
    b = _working(_pixels(27111, 0.18, 0.78), path="b.png")
    recipe = fit_reference_look(reference)
    forward = render_reference_batch(recipe, [a, b])
    reversed_results = render_reference_batch(recipe, [b, a])
    assert forward[0].image.pixels.tobytes() == reversed_results[1].image.pixels.tobytes()
    assert forward[1].image.pixels.tobytes() == reversed_results[0].image.pixels.tobytes()


def test_recipe_can_render_a_supported_different_working_space() -> None:
    reference = _working(_pixels(27112, 0.10, 0.82), path="reference.png")
    source_srgb = _pixels(27113, 0.08, 0.70)
    source_rec2020 = convert_linear_rgb(
        source_srgb,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    source = _working(
        source_rec2020,
        path="source-rec2020.png",
        working_space="linear_rec2020",
    )
    result = render_reference_look(fit_reference_look(reference), source)
    assert result.image.working_space == "linear_rec2020"
    assert np.isfinite(result.image.pixels).all()


def test_rendering_reference_again_is_near_identity() -> None:
    reference = _working(_pixels(27114, 0.10, 0.80), path="reference.png")
    policy = replace(_unprotected_policy(), strength=1.0)
    result = render_reference_look(
        fit_reference_look(reference, policy=policy),
        reference,
    )
    assert float(np.max(np.abs(result.image.pixels - reference.pixels))) <= 4e-6


@pytest.mark.parametrize("state", ["scene_linear", "display_referred", "unknown"])
def test_render_fails_closed_for_unsupported_source_state(state: str) -> None:
    reference = _working(_pixels(27115, 0.10, 0.80), path="reference.png")
    source = _working(
        _pixels(27116, 0.10, 0.80),
        path="source.png",
        transfer_state=state,
    )
    with pytest.raises(ReferenceMatchContractError, match="display-linear SDR sources"):
        render_reference_look(fit_reference_look(reference), source)


def test_render_fails_closed_for_out_of_gamut_source() -> None:
    reference = _working(_pixels(27117, 0.10, 0.80), path="reference.png")
    pixels = _pixels(27118, 0.10, 0.80)
    pixels[0, 0, 2] = np.float32(-0.1)
    source = _working(pixels, path="source.png")
    with pytest.raises(ReferenceMatchContractError, match="outside the declared working gamut"):
        render_reference_look(fit_reference_look(reference), source)


@pytest.mark.parametrize("sources", [[], (), "not-a-batch"])
def test_batch_rejects_empty_or_invalid_input(sources: object) -> None:
    reference = _working(_pixels(27119, 0.10, 0.80), path="reference.png")
    with pytest.raises(ReferenceMatchContractError):
        render_reference_batch(fit_reference_look(reference), sources)


def test_diagnostics_preserve_reference_look_claim_ceiling() -> None:
    reference = _working(_pixels(27120, 0.10, 0.80), path="reference.png")
    source = _working(_pixels(27121, 0.12, 0.78), path="source.png")
    result = render_reference_look(fit_reference_look(reference), source)
    assert result.diagnostics.claim_ceiling == "reference-look"
    assert result.diagnostics.source_shape == source.pixels.shape
    assert 0.0 <= result.diagnostics.gamut_adjusted_fraction <= 1.0
