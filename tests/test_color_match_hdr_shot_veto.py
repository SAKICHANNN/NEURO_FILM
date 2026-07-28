from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match import (
    HDR_SHOT_VETO_ASSESSMENT_SCHEMA_SHA256,
    HDR_SHOT_VETO_FIXTURE_SHA256,
    HDR_SHOT_VETO_MODEL_SCHEMA_SHA256,
    ReferenceMatchContractError,
    hdr_shot_reuse_veto_decision_payload_v1,
    load_hdr_shot_reuse_veto_decision_v1,
    map_hdr_shot_reuse_veto_v1,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT.parent / "追色"
PRODUCER_MODEL_SCHEMA = (
    PRODUCER / "schemas/hdr_shot_envelope_v1.schema.json"
)
PRODUCER_ASSESSMENT_SCHEMA = (
    PRODUCER / "schemas/hdr_shot_assessment_v1.schema.json"
)
PRODUCER_FIXTURE = (
    PRODUCER / "tests/fixtures/zhuise_hdr_shot_sentinel_v1.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_hdr_shot_reuse_veto_decision_v1.schema.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _with_id(value: dict, field: str) -> dict:
    result = deepcopy(value)
    result[field] = "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()
    return result


def _model() -> dict:
    floor = 1.0 / 65535.0
    return _with_id(
        {
            "schema": "zhuise.hdr-shot-envelope.v1",
            "capability_id": (
                "zhuise.hdr-shot-envelope.cpu-reference.v1"
            ),
            "profile_id": "zhuise.p3d65-pq-normalized-rgb.v1",
            "descriptor_id": "zhuise.pq-quantiles-0.1-0.5-0.9.v1",
            "training_count": 2,
            "training_descriptor_ids": [
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
            ],
            "center": [0.5] * 9,
            "scale": [floor] * 9,
            "scale_floor": floor,
            "invalidation_threshold": 1.0,
            "semantics": "veto-only-never-authorizes-reuse",
        },
        "model_id",
    )


def _assessment(model_id: str, score: float) -> dict:
    disposition = (
        "invalidate-reuse"
        if score > 1.0
        else "not-invalidated-veto-only"
    )
    return _with_id(
        {
            "schema": "zhuise.hdr-shot-assessment.v1",
            "capability_id": (
                "zhuise.hdr-shot-envelope.cpu-reference.v1"
            ),
            "profile_id": "zhuise.p3d65-pq-normalized-rgb.v1",
            "descriptor_id": "sha256:" + "3" * 64,
            "model_id": model_id,
            "score": score,
            "invalidation_threshold": 1.0,
            "disposition": disposition,
            "claim_ceiling": "veto-only-never-authorizes-reuse",
        },
        "assessment_id",
    )


@pytest.mark.parametrize(
    ("score", "refit_required"),
    [(0.5, False), (1.0, False), (1.0000001, True)],
)
def test_mapping_is_veto_only_and_schema_valid(
    score: float,
    refit_required: bool,
) -> None:
    model = _model()
    assessment = _assessment(model["model_id"], score)
    decision = map_hdr_shot_reuse_veto_v1(
        producer_model=model,
        producer_assessment=assessment,
    )
    assert decision.refit_required is refit_required
    assert decision.reuse_authorized is False
    payload = hdr_shot_reuse_veto_decision_payload_v1(decision)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert load_hdr_shot_reuse_veto_decision_v1(payload) == decision
    assert (
        map_hdr_shot_reuse_veto_v1(
            producer_model=model,
            producer_assessment=assessment,
        )
        == decision
    )


def test_exact_producer_fixture_and_schema_hashes_map() -> None:
    paths = (
        PRODUCER_MODEL_SCHEMA,
        PRODUCER_ASSESSMENT_SCHEMA,
        PRODUCER_FIXTURE,
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("producer HDR shot sentinel artifacts unavailable")
    assert hashlib.sha256(PRODUCER_MODEL_SCHEMA.read_bytes()).hexdigest() == (
        HDR_SHOT_VETO_MODEL_SCHEMA_SHA256
    )
    assert hashlib.sha256(
        PRODUCER_ASSESSMENT_SCHEMA.read_bytes()
    ).hexdigest() == HDR_SHOT_VETO_ASSESSMENT_SCHEMA_SHA256
    assert hashlib.sha256(PRODUCER_FIXTURE.read_bytes()).hexdigest() == (
        HDR_SHOT_VETO_FIXTURE_SHA256
    )
    fixture = json.loads(PRODUCER_FIXTURE.read_text(encoding="utf-8"))
    safe = map_hdr_shot_reuse_veto_v1(
        producer_model=fixture["model"],
        producer_assessment=fixture["safe"]["assessment"],
    )
    harmful = map_hdr_shot_reuse_veto_v1(
        producer_model=fixture["model"],
        producer_assessment=fixture["harmful"]["assessment"],
    )
    assert safe.reuse_authorized is False
    assert safe.refit_required is False
    assert harmful.reuse_authorized is False
    assert harmful.refit_required is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(reuse_authorized=True),
        lambda value: value.update(refit_required=True),
        lambda value: value.update(claim_ceiling="reuse-authorized"),
        lambda value: value.update(decision_id="sha256:" + "0" * 64),
        lambda value: value.update(unexpected=True),
    ],
)
def test_persisted_decision_cannot_gain_reuse_authority(mutation) -> None:
    model = _model()
    decision = map_hdr_shot_reuse_veto_v1(
        producer_model=model,
        producer_assessment=_assessment(model["model_id"], 0.5),
    )
    payload = hdr_shot_reuse_veto_decision_payload_v1(decision)
    mutation(payload)
    with pytest.raises(ReferenceMatchContractError):
        load_hdr_shot_reuse_veto_decision_v1(payload)


@pytest.mark.parametrize(
    ("target", "mutation", "match"),
    [
        (
            "model",
            lambda value: value.update(model_id="sha256:" + "0" * 64),
            "model identity mismatch",
        ),
        (
            "model",
            lambda value: value["scale"].__setitem__(0, 0.0),
            "below the floor",
        ),
        (
            "assessment",
            lambda value: value.update(
                disposition="invalidate-reuse"
            ),
            "contradicts score",
        ),
        (
            "assessment",
            lambda value: value.update(score=math.nan),
            "not finite",
        ),
        (
            "assessment",
            lambda value: value.update(unexpected=True),
            "fields differ",
        ),
    ],
)
def test_mapping_rejects_mutation(
    target: str,
    mutation,
    match: str,
) -> None:
    model = _model()
    assessment = _assessment(model["model_id"], 0.5)
    mutation(model if target == "model" else assessment)
    with pytest.raises(ReferenceMatchContractError, match=match):
        map_hdr_shot_reuse_veto_v1(
            producer_model=model,
            producer_assessment=assessment,
        )
