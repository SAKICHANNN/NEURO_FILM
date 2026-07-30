from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image
import pytest
from referencing import Registry, Resource

from src.color_match import (
    ReferenceMatchContractError,
    ReferenceRenderGuardPolicy,
    build_file_match_report,
    build_reference_composition,
    build_reference_run_composition,
    match_reference_files,
    reference_run_composition_from_json,
    reference_run_composition_to_json,
    replay_reference_files,
)
from src.inference import atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "configs" / "schemas"
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).integers(
        24,
        232,
        size=(29, 37, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def _fallback_run(tmp_path: Path, *, seed: int = 28001):
    reference = tmp_path / "reference.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(reference, seed)
    _image(first, seed + 1)
    _image(second, seed + 2)
    return match_reference_files(
        reference,
        [first, second],
        [tmp_path / "first-output.png", tmp_path / "second-output.png"],
        recipe_path=tmp_path / "recipe.json",
        report_path=tmp_path / "report.json",
    )


def _applied_run(tmp_path: Path):
    reference = tmp_path / "reference.png"
    _image(reference, 28004)
    return match_reference_files(
        reference,
        [reference, reference],
        [tmp_path / "first-output.png", tmp_path / "second-output.png"],
        recipe_path=tmp_path / "recipe.json",
        report_path=tmp_path / "report.json",
        guard_policy=ReferenceRenderGuardPolicy(
            allow_research_baseline=True,
        ),
    )


def test_default_fallback_run_binds_only_identity_composition(
    tmp_path: Path,
) -> None:
    result = _fallback_run(tmp_path)
    plan = build_reference_composition(result.recipe)
    binding = build_reference_run_composition(plan, result)
    assert binding.batch_safety_status == "identity-fallback"
    assert binding.research_baseline_requested is False
    assert binding.composition_plan.color_owner == "identity"
    assert binding.composition_plan.film_effects is None
    assert {row.safety_action for row in binding.outputs} == {
        "identity-fallback"
    }


def test_applied_run_can_bind_effects_without_film_colour(
    tmp_path: Path,
) -> None:
    result = _applied_run(tmp_path)
    assert all(row.safety.accepted for row in result.outputs)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = build_reference_composition(
        result.recipe,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=sha256_file(PROFILE),
        allow_research_baseline=True,
    )
    binding = build_reference_run_composition(plan, result)
    assert binding.batch_safety_status == "reference-color-applied"
    assert binding.research_baseline_requested is True
    assert binding.composition_plan.execution_order == (
        "reference_color",
        "film_effects",
    )
    assert binding.composition_plan.film_color_profile_id is None
    assert binding.composition_plan.film_stock_identity_claimed is False


def test_replay_report_can_bind_identity_composition(
    tmp_path: Path,
) -> None:
    fit = _fallback_run(tmp_path)
    source = tmp_path / "replay-source.png"
    _image(source, 28005)
    replay = replay_reference_files(
        fit.recipe_path,
        [source],
        [tmp_path / "replay-output.png"],
        report_path=tmp_path / "replay-report.json",
    )
    binding = build_reference_run_composition(
        build_reference_composition(replay.recipe),
        replay,
    )
    assert (
        binding.run_report_schema_id
        == "neuro-film.reference-match-replay-report.v1"
    )
    assert binding.run_report_sha256 == replay.report_file_sha256


def test_binding_roundtrip_and_language_neutral_schema(
    tmp_path: Path,
) -> None:
    result = _fallback_run(tmp_path)
    binding = build_reference_run_composition(
        build_reference_composition(result.recipe),
        result,
    )
    encoded = reference_run_composition_to_json(binding)
    assert reference_run_composition_from_json(encoded) == binding

    schema = json.loads(
        (SCHEMAS / "reference_run_composition_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    plan_schema = json.loads(
        (SCHEMAS / "reference_composition_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        plan_schema["$id"],
        Resource.from_contents(plan_schema),
    )
    Draft202012Validator(schema, registry=registry).validate(
        json.loads(encoded)
    )


def test_report_tamper_blocks_composition_binding(tmp_path: Path) -> None:
    result = _fallback_run(tmp_path)
    result.report_path.write_bytes(result.report_path.read_bytes() + b" ")
    with pytest.raises(ReferenceMatchContractError, match="hash mismatch"):
        build_reference_run_composition(
            build_reference_composition(result.recipe),
            result,
        )


def test_binding_does_not_require_original_reference_or_sources(
    tmp_path: Path,
) -> None:
    result = _fallback_run(tmp_path)
    result.reference_path.unlink()
    for row in result.outputs:
        row.source_path.unlink()
    binding = build_reference_run_composition(
        build_reference_composition(result.recipe),
        result,
    )
    assert binding.batch_safety_status == "identity-fallback"


def test_output_tamper_blocks_composition_binding(tmp_path: Path) -> None:
    result = _fallback_run(tmp_path)
    result.outputs[0].output_path.write_bytes(b"tampered output")
    with pytest.raises(
        ReferenceMatchContractError,
        match="output file hash mismatch",
    ):
        build_reference_run_composition(
            build_reference_composition(result.recipe),
            result,
        )


def test_mixed_delivery_cannot_form_one_batch_composition(
    tmp_path: Path,
) -> None:
    result = _applied_run(tmp_path)
    rejected = replace(
        result.outputs[1].safety,
        accepted=False,
        action="identity-fallback",
        reasons=("gamut-adjusted-fraction",),
    )
    mixed = replace(
        result,
        outputs=(
            result.outputs[0],
            replace(result.outputs[1], safety=rejected),
        ),
    )
    atomic_write_json(mixed.report_path, build_file_match_report(mixed))
    mixed = replace(
        mixed,
        report_file_sha256=sha256_file(mixed.report_path),
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="mixed delivery",
    ):
        build_reference_run_composition(
            build_reference_composition(mixed.recipe),
            mixed,
        )


def test_fallback_run_cannot_bind_reference_colour_or_effects(
    tmp_path: Path,
) -> None:
    result = _fallback_run(tmp_path)
    plan = build_reference_composition(
        result.recipe,
        allow_research_baseline=True,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="fallback run cannot bind",
    ):
        build_reference_run_composition(plan, result)


def test_composition_recipe_must_match_run_recipe(tmp_path: Path) -> None:
    result = _fallback_run(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = _fallback_run(other_root, seed=28101)
    with pytest.raises(
        ReferenceMatchContractError,
        match="composition recipe does not match",
    ):
        build_reference_run_composition(
            build_reference_composition(other.recipe),
            result,
        )


def test_unreported_low_level_run_cannot_bind_composition(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    _image(reference, 28006)
    _image(source, 28007)
    result = match_reference_files(
        reference,
        [source],
        [tmp_path / "output.png"],
        recipe_path=tmp_path / "recipe.json",
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="transaction-bound report",
    ):
        build_reference_run_composition(
            build_reference_composition(result.recipe),
            result,
        )


def test_recipe_or_binding_identity_drift_fails_closed(
    tmp_path: Path,
) -> None:
    result = _fallback_run(tmp_path)
    binding = build_reference_run_composition(
        build_reference_composition(result.recipe),
        result,
    )
    payload = json.loads(reference_run_composition_to_json(binding))
    payload["outputs"][0]["output_sha256"] = "0" * 64
    with pytest.raises(
        ReferenceMatchContractError,
        match="binding_id does not match",
    ):
        reference_run_composition_from_json(json.dumps(payload))


def test_recomputed_binding_cannot_hide_research_request(
    tmp_path: Path,
) -> None:
    result = _applied_run(tmp_path)
    binding = build_reference_run_composition(
        build_reference_composition(
            result.recipe,
            allow_research_baseline=True,
        ),
        result,
    )
    changed = replace(
        binding,
        binding_id="0" * 64,
        research_baseline_requested=False,
    )
    from src.color_match.canonical import canonical_sha256

    payload = changed.to_dict()
    payload.pop("binding_id")
    changed = replace(changed, binding_id=canonical_sha256(payload))
    with pytest.raises(
        ReferenceMatchContractError,
        match="research-baseline request mismatch",
    ):
        reference_run_composition_from_json(json.dumps(changed.to_dict()))
