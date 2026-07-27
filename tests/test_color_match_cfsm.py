from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.color_match.consistency import (
    evaluate_context_invariance_outputs,
    make_context_invariance_probes,
)
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.evaluation import evaluate_known_operator_batch
from src.color_match.research import (
    CFSMProjectionPolicy,
    cfsm_candidate_from_json,
    cfsm_candidate_to_json,
    fit_cfsm_batch_candidate,
    fit_cfsm_candidate,
    render_cfsm_batch,
    render_cfsm_candidate,
)
from src.preprocess import SourceProfile, WorkingImage


def _working(pixels: np.ndarray, name: str) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("cfsm_test_v0", "synthetic"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path(f"synthetic/{name}.exr"),
    )


def _cube_reference(axis_size: int = 17) -> WorkingImage:
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float32)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    matrix = np.asarray(
        [
            [0.82, 0.12, 0.06],
            [0.08, 0.84, 0.08],
            [0.04, 0.16, 0.80],
        ],
        dtype=np.float32,
    )
    transformed = cube.reshape(-1, 3) @ matrix.T
    return _working(
        transformed.reshape(axis_size, axis_size * axis_size, 3),
        "reference",
    )


def test_cfsm_fit_is_deterministic_constrained_and_non_identity() -> None:
    reference = _cube_reference()
    first = fit_cfsm_candidate(reference)
    second = fit_cfsm_candidate(reference)

    assert first.algorithm_id == second.algorithm_id
    assert first.candidate_id == second.candidate_id
    assert np.array_equal(first.lut.values, second.lut.values)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics.constraint_report.passes
    assert first.diagnostics.projected_strength > 0.0
    assert not first.diagnostics.used_identity_fallback
    assert not np.array_equal(
        first.lut.values,
        np.stack(
            np.meshgrid(
                np.linspace(0.0, 1.0, first.lut.size),
                np.linspace(0.0, 1.0, first.lut.size),
                np.linspace(0.0, 1.0, first.lut.size),
                indexing="ij",
            ),
            axis=-1,
        ),
    )


def test_cfsm_serialization_replays_and_tampering_fails_closed() -> None:
    candidate = fit_cfsm_candidate(_cube_reference())
    encoded = cfsm_candidate_to_json(candidate)
    replay = cfsm_candidate_from_json(encoded)
    probes = _working(
        np.random.default_rng(71).uniform(
            0.0,
            1.0,
            size=(13, 17, 3),
        ),
        "replay-probes",
    )

    assert replay.candidate_id == candidate.candidate_id
    assert np.array_equal(
        render_cfsm_candidate(candidate, probes).pixels,
        render_cfsm_candidate(replay, probes).pixels,
    )
    tampered = encoded.replace(
        '"projected_strength": ',
        '"projected_strength": 0.5, "ignored_original": ',
        1,
    )
    with pytest.raises(ReferenceMatchContractError, match="invalid"):
        cfsm_candidate_from_json(tampered)
    extra = encoded.replace(
        '"source_image_count": 0',
        '"source_image_count": 0, "unknown": 1',
        1,
    )
    with pytest.raises(ReferenceMatchContractError, match="invalid"):
        cfsm_candidate_from_json(extra)


def test_cfsm_fixed_operator_passes_shared_colour_context_gate() -> None:
    candidate = fit_cfsm_candidate(_cube_reference())
    first, second, region = make_context_invariance_probes()
    first_output, second_output = render_cfsm_batch(candidate, (first, second))
    metrics = evaluate_context_invariance_outputs(
        first_output,
        second_output,
        shared_region=region,
    )

    assert metrics.passed
    assert metrics.delta_e76_maximum == pytest.approx(0.0, abs=1e-6)


def test_cfsm_batch_prior_emits_one_fixed_replayable_operator() -> None:
    rng = np.random.default_rng(72)
    sources = tuple(
        _working(
            rng.uniform(0.03, 0.97, size=(17, 19, 3)),
            f"batch-source-{index}",
        )
        for index in range(3)
    )
    candidate = fit_cfsm_batch_candidate(_cube_reference(), sources)
    reversed_candidate = fit_cfsm_batch_candidate(
        _cube_reference(),
        reversed(sources),
    )
    replay = cfsm_candidate_from_json(cfsm_candidate_to_json(candidate))
    first = render_cfsm_batch(candidate, sources)
    second = render_cfsm_batch(replay, sources)

    assert candidate.diagnostics.canonical_prior_mode == "uploaded-source-batch"
    assert candidate.diagnostics.source_image_count == 3
    assert candidate.diagnostics.canonical_prior_sample_count == 969
    assert reversed_candidate.candidate_id == candidate.candidate_id
    assert all(
        np.array_equal(left.pixels, right.pixels)
        for left, right in zip(first, second)
    )


def test_cfsm_known_operator_challenger_improves_cross_content() -> None:
    candidate = fit_cfsm_candidate(_cube_reference())
    rng = np.random.default_rng(2707)
    matrix = np.asarray(
        [
            [0.82, 0.12, 0.06],
            [0.08, 0.84, 0.08],
            [0.04, 0.16, 0.80],
        ],
        dtype=np.float32,
    )
    sources = []
    targets = []
    candidates = []
    for index in range(4):
        pixels = rng.uniform(0.08, 0.92, size=(24, 32, 3)).astype(np.float32)
        source = _working(pixels, f"source-{index}")
        target = _working(pixels @ matrix.T, f"target-{index}")
        sources.append(source)
        targets.append(target)
        candidates.append(render_cfsm_candidate(candidate, source))
    metrics = evaluate_known_operator_batch(
        sources,
        targets,
        candidates,
        sample_ids=(f"sample-{index}" for index in range(4)),
    )

    assert metrics.improvement_rate == 1.0
    assert metrics.median_improvement_fraction > 0.05
    assert metrics.maximum_new_boundary_fraction == 0.0


def test_cfsm_contract_fails_closed_on_wrong_state_and_empty_batch() -> None:
    reference = _cube_reference()
    invalid = WorkingImage(
        pixels=reference.pixels,
        working_space="linear_srgb",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=reference.source_profile,
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=reference.source_path,
    )
    with pytest.raises(ReferenceMatchContractError, match="display-linear"):
        fit_cfsm_candidate(invalid)
    candidate = fit_cfsm_candidate(reference)
    with pytest.raises(ReferenceMatchContractError, match="must not be empty"):
        render_cfsm_batch(candidate, ())
    with pytest.raises(ReferenceMatchContractError, match="prior_axis_size"):
        fit_cfsm_candidate(
            reference,
            policy=CFSMProjectionPolicy(prior_axis_size=3),
        )
