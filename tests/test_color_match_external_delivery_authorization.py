from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING,
    ReferenceMatchContractError,
    authorize_external_local_delivery_v1,
    build_external_reference_composition_v1,
    commit_external_core_staging_v1,
    commit_external_filmfx_staging_v1,
    external_delivery_authorization_from_json,
    external_delivery_authorization_to_json,
    validate_external_local_delivery_authorization_v1,
    verify_external_core_staging_v1,
    verify_external_filmfx_staging_v1,
)
from tests.test_color_match_core_staging_transaction import (
    _destinations as _core_destinations,
    _pipeline,
)
from tests.test_color_match_external_filmfx_transaction import (
    _destinations as _filmfx_destinations,
    _profile,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_external_local_delivery_authorization_v1.schema.json"
)


def _chain(tmp_path: Path):
    items, batch, product_authorization = _pipeline()
    core_outputs, core_report = _core_destinations(tmp_path / "core")
    core_committed = commit_external_core_staging_v1(
        batch=batch,
        authorization=product_authorization,
        candidates=tuple(item[1] for item in items),
        output_paths=core_outputs,
        report_path=core_report,
    )
    core_verification = verify_external_core_staging_v1(
        report_path=core_report,
        expected_report_sha256=core_committed.report_file_sha256,
        expected_run_id=core_committed.run.run_id,
    )
    profile, profile_sha = _profile(tmp_path)
    composition = build_external_reference_composition_v1(
        core_verification,
        include_film_effects=True,
        film_profile=profile,
        film_profile_sha256=profile_sha,
    )
    filmfx_outputs, filmfx_report = _filmfx_destinations(
        tmp_path / "filmfx"
    )
    filmfx_committed = commit_external_filmfx_staging_v1(
        plan=composition,
        verification=core_verification,
        output_paths=filmfx_outputs,
        report_path=filmfx_report,
        seed=37,
    )
    filmfx_verification = verify_external_filmfx_staging_v1(
        report_path=filmfx_report,
        expected_report_sha256=filmfx_committed.report_file_sha256,
        expected_run_id=filmfx_committed.run.run_id,
    )
    return (
        filmfx_verification,
        composition,
        core_verification,
        product_authorization,
        filmfx_outputs,
    )


def _authorize(chain):
    filmfx, composition, core, product, _outputs = chain
    return authorize_external_local_delivery_v1(
        filmfx_verification=filmfx,
        composition=composition,
        core_verification=core,
        product_authorization=product,
    )


def test_full_chain_authorizes_local_export_without_writing(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    before = {
        path.resolve(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    authorization = _authorize(chain)
    after = {
        path.resolve(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert authorization.delivery_scope == "local-user-export"
    assert authorization.state == "authorized-for-local-delivery"
    assert (
        authorization.claim_ceiling
        == EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING
    )
    assert authorization.filmfx_verification_id == chain[0].verification_id
    assert authorization.composition_plan_id == chain[1].plan_id
    assert (
        authorization.product_staging_authorization_id
        == chain[3].authorization_id
    )
    encoded = external_delivery_authorization_to_json(authorization)
    assert (
        external_delivery_authorization_from_json(encoded)
        == authorization
    )
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


@pytest.mark.parametrize("member", ["composition", "core"])
def test_valid_foreign_chain_member_fails_identity_binding(
    tmp_path: Path,
    member: str,
) -> None:
    first = list(_chain(tmp_path / "first"))
    second = _chain(tmp_path / "second")
    first[1 if member == "composition" else 2] = (
        second[1] if member == "composition" else second[2]
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="chain identity mismatch",
    ):
        _authorize(tuple(first))


def test_valid_foreign_product_authorization_fails_binding(
    tmp_path: Path,
) -> None:
    chain = list(_chain(tmp_path))
    _items, _batch, foreign = _pipeline(different_intents=True)
    assert foreign.authorization_id != chain[3].authorization_id
    chain[3] = foreign
    with pytest.raises(
        ReferenceMatchContractError,
        match="chain identity mismatch",
    ):
        _authorize(tuple(chain))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(value, delivery_scope="public-share"),
            "scope is unsupported",
        ),
        (
            lambda value: replace(value, state="delivered"),
            "state is unsupported",
        ),
        (
            lambda value: replace(value, output_label="film-stock"),
            "output label is unsupported",
        ),
        (
            lambda value: replace(value, claim_ceiling="committed"),
            "claim ceiling mismatch",
        ),
        (
            lambda value: replace(value, authorization_id="0" * 64),
            "authorization_id mismatch",
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
        validate_external_local_delivery_authorization_v1(
            mutation(authorization)
        )
