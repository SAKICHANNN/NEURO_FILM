from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest
from PIL import Image

from src.color_match import (
    ReferenceMatchContractError,
    build_file_match_report,
    build_reference_composition,
    composition_plan_from_json,
    composition_plan_to_json,
    fit_reference_look,
    match_reference_files,
    recipe_to_json,
)
from src.inference import sha256_file
from src.preprocess import SourceProfile, WorkingImage


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "configs" / "schemas"
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _schema(name: str) -> dict:
    payload = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(payload)
    return payload


def _working() -> WorkingImage:
    pixels = np.random.default_rng(27501).uniform(
        0.04,
        0.9,
        size=(17, 19, 3),
    ).astype(np.float32)
    return WorkingImage(
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


def _image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).integers(
        16,
        240,
        size=(23, 29, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def test_reference_recipe_matches_language_neutral_schema() -> None:
    payload = json.loads(recipe_to_json(fit_reference_look(_working())))
    validator = Draft202012Validator(
        _schema("reference_look_recipe_v1.schema.json")
    )
    validator.validate(payload)

    payload["surprise"] = True
    assert any(
        error.validator == "additionalProperties"
        for error in validator.iter_errors(payload)
    )


def test_composition_roundtrip_matches_schema_with_and_without_effects() -> None:
    recipe = fit_reference_look(_working())
    validator = Draft202012Validator(
        _schema("reference_composition_v1.schema.json")
    )
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plans = [
        build_reference_composition(recipe),
        build_reference_composition(
            recipe,
            include_film_effects=True,
            film_profile=profile,
            film_profile_sha256=sha256_file(PROFILE),
        ),
    ]
    for plan in plans:
        encoded = composition_plan_to_json(plan)
        validator.validate(json.loads(encoded))
        assert composition_plan_from_json(encoded) == plan

    inconsistent = json.loads(composition_plan_to_json(plans[0]))
    inconsistent["output_label"] = "reference-look+film-effects"
    assert any(
        error.validator == "const"
        for error in validator.iter_errors(inconsistent)
    )


def test_composition_parser_rejects_unknown_or_inconsistent_fields() -> None:
    plan = build_reference_composition(fit_reference_look(_working()))
    payload = json.loads(composition_plan_to_json(plan))
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys"):
        composition_plan_from_json(json.dumps(payload))

    payload.pop("surprise")
    payload["film_stock_identity_claimed"] = True
    with pytest.raises(ReferenceMatchContractError, match="stock identity"):
        composition_plan_from_json(json.dumps(payload))


def test_file_report_matches_language_neutral_schema(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    _image(reference, 27502)
    _image(source, 27503)
    result = match_reference_files(
        reference,
        [source],
        [output],
        recipe_path=recipe,
    )
    report = build_file_match_report(result)
    validator = Draft202012Validator(
        _schema("reference_match_report_v1.schema.json")
    )
    validator.validate(report)

    report["outputs"][0]["safety"]["policy_id"] = "unknown"
    assert any(
        error.validator == "const" for error in validator.iter_errors(report)
    )


def test_report_schema_binds_acceptance_action_and_reasons(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _image(reference, 27504)
    _image(source, 27505)
    report = build_file_match_report(
        match_reference_files(reference, [source], [output])
    )
    safety = report["outputs"][0]["safety"]
    safety.update(
        {
            "accepted": True,
            "action": "identity-fallback",
            "reasons": ["gamut-adjusted-fraction"],
        }
    )
    validator = Draft202012Validator(
        _schema("reference_match_report_v1.schema.json")
    )
    assert list(validator.iter_errors(report))
