from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    authorize_shared_local_delivery_v1,
    build_shared_reference_composition_v1,
    commit_external_shared_staging_v1,
    commit_shared_filmfx_staging_v1,
    shared_delivery_authorization_from_json,
    shared_delivery_authorization_to_json,
    validate_shared_local_delivery_authorization_v1,
    verify_external_shared_staging_v1,
    verify_shared_filmfx_staging_v1,
)
from tests.test_color_match_shared_filmfx_transaction import (
    _destinations as _filmfx_destinations,
    _profile,
)
from tests.test_color_match_shared_staging_transaction import (
    _destinations as _staging_destinations,
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_local_delivery_authorization_v1.schema.json"
)


def _chain(tmp_path: Path, *, different: bool = False):
    applies, batch, numeric, product = _pipeline(
        second_clip=0.02 if different else 0.01
    )
    staging_outputs, staging_report = _staging_destinations(
        tmp_path / "staging"
    )
    staging = commit_external_shared_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        authorization=product,
        applies=applies,
        output_paths=staging_outputs,
        report_path=staging_report,
    )
    staging_verification = verify_external_shared_staging_v1(
        report_path=staging_report,
        expected_report_sha256=staging.report_file_sha256,
        expected_run_id=staging.run.run_id,
        expected_authorization_id=staging.run.authorization_id,
        expected_numeric_guard_batch_id=staging.run.numeric_guard_batch_id,
        expected_operator_id=staging.run.operator_id,
    )
    profile, profile_sha = _profile(tmp_path)
    composition = build_shared_reference_composition_v1(
        staging_verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=profile_sha,
    )
    filmfx_outputs, filmfx_report = _filmfx_destinations(
        tmp_path / "filmfx"
    )
    filmfx = commit_shared_filmfx_staging_v1(
        plan=composition,
        verification=staging_verification,
        output_paths=filmfx_outputs,
        report_path=filmfx_report,
        seed=37,
    )
    filmfx_verification = verify_shared_filmfx_staging_v1(
        report_path=filmfx_report,
        expected_report_sha256=filmfx.report_file_sha256,
        expected_run_id=filmfx.run.run_id,
    )
    return (
        filmfx_verification,
        composition,
        staging_verification,
        product,
        filmfx_outputs,
    )


def _authorize(chain):
    filmfx, composition, staging, product, _outputs = chain
    return authorize_shared_local_delivery_v1(
        filmfx_verification=filmfx,
        composition=composition,
        staging_verification=staging,
        product_authorization=product,
    )


def test_full_chain_authorizes_without_writing(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    before = {
        path.resolve(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    authorization = _authorize(chain)
    after = {
        path.resolve(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert authorization.delivery_scope == "local-user-export"
    assert authorization.state == "authorized-for-shared-local-delivery"
    assert (
        authorization.claim_ceiling
        == SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING
    )
    assert authorization.filmfx_verification_id == chain[0].verification_id
    assert authorization.composition_plan_id == chain[1].plan_id
    assert authorization.staging_verification_id == chain[2].verification_id
    assert (
        authorization.product_staging_authorization_id
        == chain[3].authorization_id
    )
    encoded = shared_delivery_authorization_to_json(authorization)
    assert shared_delivery_authorization_from_json(encoded) == authorization
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(authorization.to_dict())


def test_live_filmfx_tamper_revokes_authorization(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    chain[4][0].write_bytes(b"tampered")
    with pytest.raises(
        ReferenceMatchContractError,
        match="output 0 hash mismatch",
    ):
        _authorize(chain)


@pytest.mark.parametrize("member", ["composition", "staging", "product"])
def test_valid_foreign_chain_member_fails_binding(
    tmp_path: Path,
    member: str,
) -> None:
    first = list(_chain(tmp_path / "first"))
    second = _chain(tmp_path / "second", different=True)
    index = {"composition": 1, "staging": 2, "product": 3}[member]
    first[index] = second[index]
    with pytest.raises(
        ReferenceMatchContractError,
        match="chain identity mismatch",
    ):
        _authorize(tuple(first))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, delivery_scope="public-share"),
            "scope is invalid",
        ),
        (
            lambda value: replace(value, state="delivered"),
            "state is invalid",
        ),
        (
            lambda value: replace(value, output_label="film-stock"),
            "output label is invalid",
        ),
        (
            lambda value: replace(value, claim_ceiling="committed"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, authorization_id="0" * 64),
            "authorization identity mismatch",
        ),
    ],
)
def test_authorization_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    authorization = _authorize(_chain(tmp_path))
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_shared_local_delivery_authorization_v1(
            mutation(authorization)
        )


def test_unknown_fields_fail_closed(tmp_path: Path) -> None:
    authorization = _authorize(_chain(tmp_path))
    payload = authorization.to_dict()
    payload["destination_path"] = "forbidden"
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_delivery_authorization_from_json(json.dumps(payload))
