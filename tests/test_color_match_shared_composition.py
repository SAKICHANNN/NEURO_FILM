from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    SHARED_REFERENCE_COMPOSITION_STATE,
    ReferenceMatchContractError,
    build_shared_reference_composition_v1,
    shared_reference_composition_from_json,
    shared_reference_composition_to_json,
    validate_shared_reference_composition_v1,
)
from src.inference import sha256_file
from tests.test_color_match_shared_staging_verification import (
    _committed,
    _verify,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_composition_v1.schema.json"
)
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _verification(tmp_path: Path):
    committed, _outputs, report = _committed(tmp_path)
    return _verify(committed, report)


def test_verified_shared_reference_is_the_only_colour_owner(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    first = build_shared_reference_composition_v1(verification)
    second = build_shared_reference_composition_v1(verification)
    assert first == second
    assert first.state == SHARED_REFERENCE_COMPOSITION_STATE
    assert first.staging_verification_id == verification.verification_id
    assert first.staging_run_id == verification.run_id
    assert first.authorization_id == verification.authorization_id
    assert first.numeric_guard_batch_id == (
        verification.numeric_guard_batch_id
    )
    assert first.operator_id == verification.operator_id
    assert first.reference_view_id == verification.reference_view_id
    assert first.color_owner == "external-shared-reference-look"
    assert first.reference_color_status == "verified-shared-staging"
    assert first.output_label == "reference-look"
    assert first.execution_order == (
        "verified_shared_reference_color",
    )
    assert first.film_color_profile_id is None
    assert not first.film_stock_identity_claimed
    assert not first.calibrated_reference_claimed
    assert first.film_effects is None


def test_shared_reference_can_append_only_procedural_filmfx(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = build_shared_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=sha256_file(PROFILE),
    )
    assert plan.output_label == "reference-look+film-effects"
    assert plan.execution_order == (
        "verified_shared_reference_color",
        "film_effects",
    )
    assert plan.film_effects is not None
    assert plan.film_effects.profile_id == profile["profile_id"]
    assert plan.film_color_profile_id is None
    assert not plan.film_stock_identity_claimed
    assert not plan.calibrated_reference_claimed


def test_roundtrip_and_strict_schema_cover_both_modes(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plans = (
        build_shared_reference_composition_v1(verification),
        build_shared_reference_composition_v1(
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
        encoded = shared_reference_composition_to_json(plan)
        assert shared_reference_composition_from_json(encoded) == plan
        validator.validate(json.loads(encoded))


def test_profile_arguments_are_explicit_and_atomic(tmp_path: Path) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    with pytest.raises(
        ReferenceMatchContractError,
        match="require include_film_effects",
    ):
        build_shared_reference_composition_v1(
            verification,
            film_profile=profile,
            film_profile_sha256=sha256_file(PROFILE),
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="profile and hash are required",
    ):
        build_shared_reference_composition_v1(
            verification,
            include_film_effects=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, state="applied"),
            "state is invalid",
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
                calibrated_reference_claimed=True,
            ),
            "cannot claim calibrated reference",
        ),
        (
            lambda value: replace(
                value,
                execution_order=(
                    "film_effects",
                    "verified_shared_reference_color",
                ),
            ),
            "execution plan is inconsistent",
        ),
        (
            lambda value: replace(
                value,
                staging_verification_id="0" * 64,
            ),
            "plan identity mismatch",
        ),
    ],
)
def test_claim_order_and_chain_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    plan = build_shared_reference_composition_v1(
        _verification(tmp_path)
    )
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_shared_reference_composition_v1(mutation(plan))


def test_effect_strength_and_profile_identity_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    verification = _verification(tmp_path)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = build_shared_reference_composition_v1(
        verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=sha256_file(PROFILE),
    )
    assert plan.film_effects is not None
    with pytest.raises(
        ReferenceMatchContractError,
        match="strengths must be within",
    ):
        validate_shared_reference_composition_v1(
            replace(
                plan,
                film_effects=replace(plan.film_effects, grain=1.5),
            )
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="plan identity mismatch",
    ):
        validate_shared_reference_composition_v1(
            replace(
                plan,
                film_effects=replace(
                    plan.film_effects,
                    profile_sha256="0" * 64,
                ),
            )
        )


def test_unknown_serialized_field_fails_closed(tmp_path: Path) -> None:
    plan = build_shared_reference_composition_v1(
        _verification(tmp_path)
    )
    payload = plan.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError):
        shared_reference_composition_from_json(json.dumps(payload))
