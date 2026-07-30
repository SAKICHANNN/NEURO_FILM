from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2,
    DPCT_INVOCATION_PROFILE_SCHEMA_V2,
    DpctInvocationProfileV2,
    ReferenceMatchContractError,
    dpct_invocation_profile_id_v2,
    dpct_invocation_profile_payload_v2,
    issue_dpct_invocation_profile_v2,
    load_dpct_invocation_profile_v2,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_dpct_invocation_profile_v2.schema.json"
)


def _issued() -> DpctInvocationProfileV2:
    return issue_dpct_invocation_profile_v2(
        DpctInvocationProfileV2(
            schema_id="placeholder",
            profile_id="0" * 64,
            producer_repository_id="zhuise-dpct",
            producer_stable_commit="1" * 40,
            producer_package_source_commit="2" * 40,
            producer_fixture_commit="3" * 40,
            producer_package_lock_sha256="4" * 64,
            package_distribution="zhuise-research",
            package_version="0.3.0",
            package_entrypoint="zhuise-producer-invoke",
            wheel_filename="zhuise_research-0.3.0-py3-none-any.whl",
            wheel_size_bytes=123456,
            wheel_sha256="5" * 64,
            python_major_minor="3.12",
            numpy_version="2.4.4",
            pillow_version="12.1.1",
            request_schema="zhuise.invocation-request.v1",
            request_schema_sha256="6" * 64,
            response_schema="zhuise.invocation-response.v1",
            response_schema_sha256="7" * 64,
            fixture_sha256="8" * 64,
            capability_id="zhuise.bmkl.cpu-reference.v1",
            producer_profile_id=(
                "zhuise.display-linear-srgb-d65-relative-f32.v1"
            ),
            lower_compatibility_profile_id=(
                "neuro-film.dpct-consumer.v2"
            ),
            evaluation_allowed=True,
            redistribution_allowed=False,
            rights_reason="fixed local evaluation permission",
            claim_ceiling="placeholder",
        )
    )


def test_profile_is_canonical_strict_and_schema_valid() -> None:
    profile = _issued()
    payload = dpct_invocation_profile_payload_v2(profile)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert profile.schema_id == DPCT_INVOCATION_PROFILE_SCHEMA_V2
    assert profile.claim_ceiling == (
        DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2
    )
    assert profile.profile_id == dpct_invocation_profile_id_v2(profile)
    assert load_dpct_invocation_profile_v2(payload) == profile
    assert (
        issue_dpct_invocation_profile_v2(profile)
        == profile
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value.update(profile_id="0" * 64),
            "identity mismatch",
        ),
        (
            lambda value: value["wire"].update(
                profile_id="zhuise.display-absolute-bt2020.v1"
            ),
            "color compatibility mismatch|schema",
        ),
        (
            lambda value: value["wire"].update(
                capability_id="../mutable"
            ),
            "capability_id is invalid|schema",
        ),
        (
            lambda value: value["package"].update(
                wheel_filename="../candidate.whl"
            ),
            "wheel filename is unsafe|schema",
        ),
        (
            lambda value: value["package"].update(
                wheel_size_bytes=True
            ),
            "wheel size is invalid|schema",
        ),
        (
            lambda value: value["rights"].update(
                evaluation_allowed=False
            ),
            "does not permit evaluation|schema",
        ),
        (
            lambda value: value["rights"].update(
                redistribution_allowed=1
            ),
            "rights flags are invalid|schema",
        ),
        (
            lambda value: value.update(applied=True),
            "fields differ",
        ),
    ],
)
def test_profile_mutations_fail_closed(mutate, match: str) -> None:
    payload = deepcopy(
        dpct_invocation_profile_payload_v2(_issued())
    )
    mutate(payload)
    with pytest.raises(ReferenceMatchContractError, match=match):
        load_dpct_invocation_profile_v2(payload)


def test_profile_identity_binds_capability_wheel_runtime_and_rights() -> None:
    original = _issued()
    mutations = [
        {"capability_id": "zhuise.other.cpu-reference.v1"},
        {"wheel_sha256": "9" * 64},
        {"numpy_version": "2.4.5"},
        {"redistribution_allowed": True},
    ]
    identities = set()
    for mutation in mutations:
        payload = original.__dict__ | mutation
        changed = issue_dpct_invocation_profile_v2(
            DpctInvocationProfileV2(**payload)
        )
        identities.add(changed.profile_id)
    assert original.profile_id not in identities
    assert len(identities) == len(mutations)
