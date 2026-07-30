from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_REFERENCE_COMPOSITION_STATE,
    ReferenceMatchContractError,
    build_external_reference_composition_v1,
    external_reference_composition_from_json,
    external_reference_composition_to_json,
    validate_external_reference_composition_v1,
)
from src.inference import sha256_file
from tests.test_color_match_core_staging_verification import (
    _committed,
    _verify,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_composition_v1.schema.json"
)
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _verification(tmp_path: Path):
    committed, _outputs, report = _committed(tmp_path)
    return _verify(committed, report)


def test_verified_external_reference_owns_colour_without_effects(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    plan = build_external_reference_composition_v1(verification)
    assert plan.state == EXTERNAL_REFERENCE_COMPOSITION_STATE
    assert plan.staging_verification_id == verification.verification_id
    assert plan.staging_run_id == verification.run_id
    assert plan.reference_intent_id == verification.reference_intent_id
    assert plan.color_owner == "external-reference-look"
    assert plan.reference_color_status == "verified-staging"
    assert plan.output_label == "reference-look"
    assert plan.execution_order == ("verified_reference_color",)
    assert plan.film_color_profile_id is None
    assert plan.film_stock_identity_claimed is False
    assert plan.film_effects is None


def test_verified_reference_can_append_only_procedural_filmfx(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = build_external_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=sha256_file(PROFILE),
    )
    assert plan.output_label == "reference-look+film-effects"
    assert plan.execution_order == (
        "verified_reference_color",
        "film_effects",
    )
    assert plan.film_effects is not None
    assert plan.film_effects.profile_id == profile["profile_id"]
    assert plan.film_color_profile_id is None
    assert plan.film_stock_identity_claimed is False


def test_roundtrip_matches_strict_schema(tmp_path: Path) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plans = (
        build_external_reference_composition_v1(verification),
        build_external_reference_composition_v1(
            verification,
            include_film_effects=True,
            film_profile=profile,
            film_profile_sha256=sha256_file(PROFILE),
        ),
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for plan in plans:
        encoded = external_reference_composition_to_json(plan)
        assert external_reference_composition_from_json(encoded) == plan
        validator.validate(json.loads(encoded))


def test_profile_arguments_are_explicit_and_atomic(tmp_path: Path) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    with pytest.raises(
        ReferenceMatchContractError,
        match="require include_film_effects",
    ):
        build_external_reference_composition_v1(
            verification,
            film_profile=profile,
            film_profile_sha256=sha256_file(PROFILE),
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="profile and hash are required",
    ):
        build_external_reference_composition_v1(
            verification,
            include_film_effects=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is unsupported",
        ),
        (
            lambda value: replace(
                value,
                film_color_profile_id="velvia-colour",
            ),
            "cannot stack film colour",
        ),
        (
            lambda value: replace(
                value,
                film_stock_identity_claimed=True,
            ),
            "cannot claim a film stock identity",
        ),
        (
            lambda value: replace(
                value,
                execution_order=("film_effects", "verified_reference_color"),
            ),
            "execution plan is inconsistent",
        ),
        (
            lambda value: replace(
                value,
                staging_verification_id="0" * 64,
            ),
            "plan_id mismatch",
        ),
    ],
)
def test_claim_and_identity_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    plan = build_external_reference_composition_v1(
        _verification(tmp_path)
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_external_reference_composition_v1(mutation(plan))


def test_effect_strength_mutation_fails_closed(tmp_path: Path) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = build_external_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=sha256_file(PROFILE),
    )
    assert plan.film_effects is not None
    changed = replace(
        plan,
        film_effects=replace(plan.film_effects, grain=1.5),
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="strengths must be within",
    ):
        validate_external_reference_composition_v1(changed)


def test_unknown_serialized_field_fails_closed(tmp_path: Path) -> None:
    plan = build_external_reference_composition_v1(
        _verification(tmp_path)
    )
    payload = plan.to_dict()
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        external_reference_composition_from_json(json.dumps(payload))
