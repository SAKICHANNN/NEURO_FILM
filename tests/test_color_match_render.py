from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.color_engine import (
    apply_safe_lab_transform,
    compress_chroma_to_working_gamut,
    compress_source_to_working_gamut,
    in_working_gamut,
    lab_to_linear_rgb,
    linear_rgb_to_lab,
    safe_lab_context_from_lab,
)
from src.color_match import (
    MAX_REFERENCE_MATCH_BATCH_SOURCES,
    ReferenceLookPolicy,
    ReferenceMatchContractError,
    fit_reference_look,
    render_reference_batch,
    render_reference_look,
)
from src.preprocess import convert_linear_rgb
from src.preprocess.types import DecodeWarning, SourceProfile, WorkingImage
from src.color_match import render as render_module
from src.color_match import row_kernels


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


def test_batch_rejects_unbounded_iterable_after_65_pulls() -> None:
    reference = _working(_pixels(27122, 0.10, 0.80), path="reference.png")
    source = _working(_pixels(27123, 0.10, 0.80), path="source.png")
    pulls = 0

    def unbounded_sources():
        nonlocal pulls
        while True:
            pulls += 1
            if pulls > MAX_REFERENCE_MATCH_BATCH_SOURCES + 1:
                raise AssertionError("renderer over-consumed the source iterable")
            yield source

    with pytest.raises(
        ReferenceMatchContractError,
        match="supports at most 64 sources",
    ):
        render_reference_batch(
            fit_reference_look(reference),
            unbounded_sources(),
        )

    assert pulls == MAX_REFERENCE_MATCH_BATCH_SOURCES + 1


def test_diagnostics_preserve_reference_look_claim_ceiling() -> None:
    reference = _working(_pixels(27120, 0.10, 0.80), path="reference.png")
    source = _working(_pixels(27121, 0.12, 0.78), path="source.png")
    result = render_reference_look(fit_reference_look(reference), source)
    assert result.diagnostics.claim_ceiling == "reference-look"
    assert result.diagnostics.source_shape == source.pixels.shape
    assert 0.0 <= result.diagnostics.gamut_adjusted_fraction <= 1.0


@pytest.mark.parametrize("gamut_mode", ["source", "chroma"])
def test_row_chunked_gamut_is_float32_exact_to_full_frame(
    gamut_mode: str,
) -> None:
    reference = _working(
        _pixels(27124, 0.16, 0.84),
        path="reference.png",
    )
    source_pixels = np.random.default_rng(27125).uniform(
        0.01,
        0.99,
        size=(257, 389, 3),
    ).astype(np.float32)
    source = _working(source_pixels, path="source.png")
    recipe = fit_reference_look(
        reference,
        policy=replace(
            ReferenceLookPolicy(),
            gamut_mode=gamut_mode,
            strength=1.0,
            luma_strength=1.0,
        ),
    )
    source_lab = linear_rgb_to_lab(
        source.pixels,
        working_space=source.working_space,
    )
    styled_lab = render_module._styled_lab(recipe, source_lab)
    if gamut_mode == "source":
        full = compress_source_to_working_gamut(
            source_lab,
            styled_lab,
            working_space=source.working_space,
            iterations=recipe.policy.gamut_iterations,
        )
    else:
        full = compress_chroma_to_working_gamut(
            styled_lab,
            working_space=source.working_space,
            iterations=recipe.policy.gamut_iterations,
        )
    chunked = render_module._gamut_safe_lab(
        recipe,
        source_lab,
        styled_lab,
        working_space=source.working_space,
    )
    np.testing.assert_array_equal(chunked, full)


@pytest.mark.parametrize("working_space", ["linear_srgb", "linear_rec2020"])
@pytest.mark.parametrize("luma_detail", [0.0, 0.35])
def test_halo_row_styled_lab_is_float32_exact_to_full_frame(
    working_space: str,
    luma_detail: float,
) -> None:
    reference = _working(
        np.random.default_rng(27132).uniform(
            0.1,
            0.8,
            size=(79, 83, 3),
        ).astype(np.float32),
        path="reference.png",
        working_space=working_space,
    )
    source = _working(
        np.random.default_rng(27133).uniform(
            0.05,
            0.85,
            size=(257, 389, 3),
        ).astype(np.float32),
        path="source.png",
        working_space=working_space,
    )
    recipe = fit_reference_look(
        reference,
        policy=replace(
            ReferenceLookPolicy(),
            preserve_luma_detail_strength=luma_detail,
        ),
    )
    source_lab = linear_rgb_to_lab(
        source.pixels,
        working_space=working_space,
    )
    policy = recipe.policy
    full = apply_safe_lab_transform(
        source_lab,
        source_context=safe_lab_context_from_lab(source_lab),
        destination_mean=np.asarray(recipe.destination_lab_mean, dtype=np.float32),
        destination_std=np.asarray(recipe.destination_lab_std, dtype=np.float32),
        style="reference_look",
        strength=policy.strength,
        luma_strength=policy.luma_strength,
        tone_rolloff=policy.tone_rolloff,
        shadow_floor_l=policy.shadow_floor_l,
        highlight_ceiling_l=policy.highlight_ceiling_l,
        preserve_luma_detail_strength=policy.preserve_luma_detail_strength,
        chroma_curve_strength=policy.chroma_curve_strength,
        neutral_protect=policy.neutral_protect,
        skin_protect=policy.skin_protect,
        max_chroma_gain=policy.max_chroma_gain,
        max_chroma_boost=policy.max_chroma_boost,
        max_chroma_absolute=policy.max_chroma_absolute,
    )
    chunked = render_module._styled_lab(recipe, source_lab)
    np.testing.assert_array_equal(chunked, full)


def test_halo_row_style_never_exceeds_core_plus_two_halos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []
    original = render_module.apply_safe_lab_transform

    def record_rows(source_lab, **kwargs):
        observed_rows.append(int(source_lab.shape[0]))
        return original(source_lab, **kwargs)

    monkeypatch.setattr(render_module, "apply_safe_lab_transform", record_rows)
    reference = _working(
        np.random.default_rng(27134).uniform(
            0.1,
            0.8,
            size=(71, 73, 3),
        ).astype(np.float32),
        path="reference.png",
    )
    source = _working(
        np.random.default_rng(27135).uniform(
            0.05,
            0.85,
            size=(400, 19, 3),
        ).astype(np.float32),
        path="source.png",
    )
    render_reference_look(fit_reference_look(reference), source)
    assert observed_rows == [133, 138, 138, 21]


def test_row_chunked_lab_to_rgb_and_gamut_check_are_exact() -> None:
    pixels = np.random.default_rng(27126).uniform(
        0.01,
        0.99,
        size=(257, 389, 3),
    ).astype(np.float32)
    lab = linear_rgb_to_lab(pixels, working_space="linear_srgb")
    full_rgb = lab_to_linear_rgb(lab, working_space="linear_srgb")
    chunked_rgb = render_module._lab_to_linear_rgb_rows(
        lab,
        working_space="linear_srgb",
    )
    np.testing.assert_array_equal(chunked_rgb, full_rgb)
    assert render_module._in_working_gamut_rows(
        lab,
        working_space="linear_srgb",
        tolerance=2e-6,
    ) == bool(
        in_working_gamut(
            lab,
            working_space="linear_srgb",
            tolerance=2e-6,
        ).all()
    )
    invalid = lab.copy()
    invalid[-1, -1] = np.asarray([50.0, 300.0, -300.0], dtype=np.float32)
    assert not render_module._in_working_gamut_rows(
        invalid,
        working_space="linear_srgb",
        tolerance=2e-6,
    )


@pytest.mark.parametrize("working_space", ["linear_srgb", "linear_rec2020"])
def test_row_chunked_rgb_to_lab_is_float32_exact(
    working_space: str,
) -> None:
    pixels = np.random.default_rng(27129).uniform(
        0.01,
        0.99,
        size=(257, 389, 3),
    ).astype(np.float32)
    full = linear_rgb_to_lab(pixels, working_space=working_space)
    chunked = row_kernels.linear_rgb_to_lab_rows(
        pixels,
        working_space=working_space,
    )
    np.testing.assert_array_equal(chunked, full)


def test_fit_and_render_lab_ingress_never_exceeds_128_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []
    original = row_kernels.linear_rgb_to_lab

    def record_rows(pixels, **kwargs):
        observed_rows.append(int(pixels.shape[0]))
        return original(pixels, **kwargs)

    monkeypatch.setattr(row_kernels, "linear_rgb_to_lab", record_rows)
    reference = _working(
        np.random.default_rng(27130).uniform(
            0.1,
            0.9,
            size=(131, 17, 3),
        ).astype(np.float32),
        path="reference.png",
    )
    source = _working(
        np.random.default_rng(27131).uniform(
            0.1,
            0.9,
            size=(257, 19, 3),
        ).astype(np.float32),
        path="source.png",
    )
    render_reference_look(fit_reference_look(reference), source)
    assert observed_rows == [128, 3, 128, 128, 1]


def test_chunked_render_never_sends_more_than_128_rows_to_gamut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []
    original = render_module.compress_source_to_working_gamut

    def record_rows(source_lab, target_lab, **kwargs):
        observed_rows.append(int(source_lab.shape[0]))
        return original(source_lab, target_lab, **kwargs)

    monkeypatch.setattr(
        render_module,
        "compress_source_to_working_gamut",
        record_rows,
    )
    reference = _working(
        np.random.default_rng(27127).uniform(
            0.1,
            0.9,
            size=(131, 17, 3),
        ).astype(np.float32),
        path="reference.png",
    )
    source = _working(
        np.random.default_rng(27128).uniform(
            0.1,
            0.9,
            size=(257, 19, 3),
        ).astype(np.float32),
        path="source.png",
    )
    render_reference_look(fit_reference_look(reference), source)
    assert observed_rows == [128, 128, 1]
