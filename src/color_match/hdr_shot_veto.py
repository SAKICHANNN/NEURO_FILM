"""Fail-closed consumer mapping for the producer HDR shot veto."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from .contracts import ReferenceMatchContractError


HDR_SHOT_VETO_DECISION_SCHEMA_V1 = (
    "neuro-film.hdr-shot-reuse-veto-decision.v1"
)
HDR_SHOT_VETO_COMPATIBILITY_PROFILE_V1 = (
    "neuro-film.zhuise-hdr-shot-veto.v1"
)
HDR_SHOT_VETO_PRODUCER_COMMIT = (
    "ec717bf538f7f2b1b128e105d4007eaf3acf1d34"
)
HDR_SHOT_VETO_MODEL_SCHEMA_SHA256 = (
    "ec56221d3b6dec4cd8be3abf9fc1a1c65a58cded77735d8c14c362f0a947a717"
)
HDR_SHOT_VETO_ASSESSMENT_SCHEMA_SHA256 = (
    "36c8cd8b8064695fb7173a0a098cb576cf15ba1a705bee29bc5a2e0118621d2d"
)
HDR_SHOT_VETO_FIXTURE_SHA256 = (
    "457769e1f74aa7a9dbcefd9168f92bf6cd8cdcce702ff898ecf0420e2424e54d"
)
HDR_SHOT_VETO_CLAIM_CEILING = "veto-only-never-authorizes-reuse"
HDR_SHOT_VETO_MODEL_SCHEMA = "zhuise.hdr-shot-envelope.v1"
HDR_SHOT_VETO_ASSESSMENT_SCHEMA = "zhuise.hdr-shot-assessment.v1"
HDR_SHOT_VETO_CAPABILITY_ID = (
    "zhuise.hdr-shot-envelope.cpu-reference.v1"
)
HDR_SHOT_VETO_PROFILE_ID = "zhuise.p3d65-pq-normalized-rgb.v1"
HDR_SHOT_VETO_DESCRIPTOR_ID = (
    "zhuise.pq-quantiles-0.1-0.5-0.9.v1"
)
HDR_SHOT_VETO_SCALE_FLOOR = 1.0 / 65535.0
HDR_SHOT_VETO_THRESHOLD = 1.0
HDR_SHOT_INVALIDATE = "invalidate-reuse"
HDR_SHOT_NOT_INVALIDATED = "not-invalidated-veto-only"
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_DECISION_DOMAIN = b"NeuroFilmHdrShotReuseVetoDecisionV1\0"


@dataclass(frozen=True)
class HDRShotReuseVetoDecisionV1:
    schema_id: str
    compatibility_profile_id: str
    producer_commit: str
    producer_capability_id: str
    producer_profile_id: str
    model_id: str
    assessment_id: str
    disposition: str
    refit_required: bool
    reuse_authorized: bool
    claim_ceiling: str
    decision_id: str


def _strict(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from pinned HDR shot veto"
        )
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "HDR shot veto envelope is not canonicalizable"
        ) from exc


def _sha256_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
        raise ReferenceMatchContractError(f"{label} is not a SHA-256 ID")
    return value


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ReferenceMatchContractError(f"{label} is not finite")
    return float(value)


def _vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 9:
        raise ReferenceMatchContractError(f"{label} is not a 9-vector")
    return tuple(_number(item, label) for item in value)


def _verify_model(value: Any) -> Mapping[str, Any]:
    model = _strict(
        value,
        {
            "schema",
            "capability_id",
            "profile_id",
            "descriptor_id",
            "training_count",
            "training_descriptor_ids",
            "center",
            "scale",
            "scale_floor",
            "invalidation_threshold",
            "semantics",
            "model_id",
        },
        "producer HDR shot model",
    )
    if (
        model["schema"] != HDR_SHOT_VETO_MODEL_SCHEMA
        or model["capability_id"] != HDR_SHOT_VETO_CAPABILITY_ID
        or model["profile_id"] != HDR_SHOT_VETO_PROFILE_ID
        or model["descriptor_id"] != HDR_SHOT_VETO_DESCRIPTOR_ID
        or model["scale_floor"] != HDR_SHOT_VETO_SCALE_FLOOR
        or model["invalidation_threshold"] != HDR_SHOT_VETO_THRESHOLD
        or model["semantics"] != HDR_SHOT_VETO_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "producer HDR shot model contract mismatch"
        )
    count = model["training_count"]
    descriptor_ids = model["training_descriptor_ids"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 2
        or not isinstance(descriptor_ids, list)
        or len(descriptor_ids) != count
    ):
        raise ReferenceMatchContractError(
            "producer HDR shot model training identity mismatch"
        )
    for item in descriptor_ids:
        _sha256_id(item, "training descriptor ID")
    _vector(model["center"], "model center")
    scale = _vector(model["scale"], "model scale")
    if any(item < HDR_SHOT_VETO_SCALE_FLOOR for item in scale):
        raise ReferenceMatchContractError(
            "producer HDR shot model scale is below the floor"
        )
    model_id = _sha256_id(model["model_id"], "model ID")
    identity = dict(model)
    identity.pop("model_id")
    expected = "sha256:" + hashlib.sha256(_canonical(identity)).hexdigest()
    if model_id != expected:
        raise ReferenceMatchContractError(
            "producer HDR shot model identity mismatch"
        )
    return model


def _verify_assessment(
    value: Any,
    *,
    model_id: str,
) -> Mapping[str, Any]:
    assessment = _strict(
        value,
        {
            "schema",
            "capability_id",
            "profile_id",
            "descriptor_id",
            "model_id",
            "score",
            "invalidation_threshold",
            "disposition",
            "claim_ceiling",
            "assessment_id",
        },
        "producer HDR shot assessment",
    )
    if (
        assessment["schema"] != HDR_SHOT_VETO_ASSESSMENT_SCHEMA
        or assessment["capability_id"] != HDR_SHOT_VETO_CAPABILITY_ID
        or assessment["profile_id"] != HDR_SHOT_VETO_PROFILE_ID
        or assessment["model_id"] != model_id
        or assessment["invalidation_threshold"]
        != HDR_SHOT_VETO_THRESHOLD
        or assessment["claim_ceiling"] != HDR_SHOT_VETO_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "producer HDR shot assessment contract mismatch"
        )
    _sha256_id(assessment["descriptor_id"], "assessment descriptor ID")
    score = _number(assessment["score"], "assessment score")
    if score < 0.0:
        raise ReferenceMatchContractError(
            "assessment score is negative"
        )
    expected_disposition = (
        HDR_SHOT_INVALIDATE
        if score > HDR_SHOT_VETO_THRESHOLD
        else HDR_SHOT_NOT_INVALIDATED
    )
    if assessment["disposition"] != expected_disposition:
        raise ReferenceMatchContractError(
            "producer HDR shot disposition contradicts score"
        )
    assessment_id = _sha256_id(
        assessment["assessment_id"], "assessment ID"
    )
    identity = dict(assessment)
    identity.pop("assessment_id")
    expected = "sha256:" + hashlib.sha256(_canonical(identity)).hexdigest()
    if assessment_id != expected:
        raise ReferenceMatchContractError(
            "producer HDR shot assessment identity mismatch"
        )
    return assessment


def hdr_shot_reuse_veto_decision_payload_v1(
    value: HDRShotReuseVetoDecisionV1,
) -> dict[str, Any]:
    return {
        "schema_id": value.schema_id,
        "compatibility_profile_id": value.compatibility_profile_id,
        "producer_commit": value.producer_commit,
        "producer_capability_id": value.producer_capability_id,
        "producer_profile_id": value.producer_profile_id,
        "model_id": value.model_id,
        "assessment_id": value.assessment_id,
        "disposition": value.disposition,
        "refit_required": value.refit_required,
        "reuse_authorized": value.reuse_authorized,
        "claim_ceiling": value.claim_ceiling,
        "decision_id": value.decision_id,
    }


def load_hdr_shot_reuse_veto_decision_v1(
    value: Any,
) -> HDRShotReuseVetoDecisionV1:
    payload = _strict(
        value,
        {
            "schema_id",
            "compatibility_profile_id",
            "producer_commit",
            "producer_capability_id",
            "producer_profile_id",
            "model_id",
            "assessment_id",
            "disposition",
            "refit_required",
            "reuse_authorized",
            "claim_ceiling",
            "decision_id",
        },
        "consumer HDR shot veto decision",
    )
    disposition = payload["disposition"]
    refit_required = payload["refit_required"]
    if (
        payload["schema_id"] != HDR_SHOT_VETO_DECISION_SCHEMA_V1
        or payload["compatibility_profile_id"]
        != HDR_SHOT_VETO_COMPATIBILITY_PROFILE_V1
        or payload["producer_commit"] != HDR_SHOT_VETO_PRODUCER_COMMIT
        or payload["producer_capability_id"]
        != HDR_SHOT_VETO_CAPABILITY_ID
        or payload["producer_profile_id"] != HDR_SHOT_VETO_PROFILE_ID
        or payload["claim_ceiling"] != HDR_SHOT_VETO_CLAIM_CEILING
        or disposition not in {
            HDR_SHOT_INVALIDATE,
            HDR_SHOT_NOT_INVALIDATED,
        }
        or not isinstance(refit_required, bool)
        or refit_required != (disposition == HDR_SHOT_INVALIDATE)
        or payload["reuse_authorized"] is not False
    ):
        raise ReferenceMatchContractError(
            "consumer HDR shot veto decision contract mismatch"
        )
    model_id = _sha256_id(payload["model_id"], "decision model ID")
    assessment_id = _sha256_id(
        payload["assessment_id"], "decision assessment ID"
    )
    decision_id = _sha256_id(
        payload["decision_id"], "decision ID"
    )
    identity = dict(payload)
    identity.pop("decision_id")
    expected = "sha256:" + hashlib.sha256(
        _DECISION_DOMAIN + _canonical(identity)
    ).hexdigest()
    if decision_id != expected:
        raise ReferenceMatchContractError(
            "consumer HDR shot veto decision identity mismatch"
        )
    return HDRShotReuseVetoDecisionV1(
        schema_id=HDR_SHOT_VETO_DECISION_SCHEMA_V1,
        compatibility_profile_id=(
            HDR_SHOT_VETO_COMPATIBILITY_PROFILE_V1
        ),
        producer_commit=HDR_SHOT_VETO_PRODUCER_COMMIT,
        producer_capability_id=HDR_SHOT_VETO_CAPABILITY_ID,
        producer_profile_id=HDR_SHOT_VETO_PROFILE_ID,
        model_id=model_id,
        assessment_id=assessment_id,
        disposition=str(disposition),
        refit_required=refit_required,
        reuse_authorized=False,
        claim_ceiling=HDR_SHOT_VETO_CLAIM_CEILING,
        decision_id=decision_id,
    )


def map_hdr_shot_reuse_veto_v1(
    *,
    producer_model: Any,
    producer_assessment: Any,
) -> HDRShotReuseVetoDecisionV1:
    model = _verify_model(producer_model)
    assessment = _verify_assessment(
        producer_assessment,
        model_id=str(model["model_id"]),
    )
    disposition = str(assessment["disposition"])
    identity = {
        "schema_id": HDR_SHOT_VETO_DECISION_SCHEMA_V1,
        "compatibility_profile_id": (
            HDR_SHOT_VETO_COMPATIBILITY_PROFILE_V1
        ),
        "producer_commit": HDR_SHOT_VETO_PRODUCER_COMMIT,
        "producer_capability_id": HDR_SHOT_VETO_CAPABILITY_ID,
        "producer_profile_id": HDR_SHOT_VETO_PROFILE_ID,
        "model_id": model["model_id"],
        "assessment_id": assessment["assessment_id"],
        "disposition": disposition,
        "refit_required": disposition == HDR_SHOT_INVALIDATE,
        "reuse_authorized": False,
        "claim_ceiling": HDR_SHOT_VETO_CLAIM_CEILING,
    }
    decision_id = "sha256:" + hashlib.sha256(
        _DECISION_DOMAIN + _canonical(identity)
    ).hexdigest()
    return load_hdr_shot_reuse_veto_decision_v1(
        {
            **identity,
            "decision_id": decision_id,
        }
    )


__all__ = [
    "HDR_SHOT_INVALIDATE",
    "HDR_SHOT_NOT_INVALIDATED",
    "HDR_SHOT_VETO_ASSESSMENT_SCHEMA_SHA256",
    "HDR_SHOT_VETO_CLAIM_CEILING",
    "HDR_SHOT_VETO_COMPATIBILITY_PROFILE_V1",
    "HDR_SHOT_VETO_DECISION_SCHEMA_V1",
    "HDR_SHOT_VETO_FIXTURE_SHA256",
    "HDR_SHOT_VETO_MODEL_SCHEMA_SHA256",
    "HDR_SHOT_VETO_PRODUCER_COMMIT",
    "HDRShotReuseVetoDecisionV1",
    "hdr_shot_reuse_veto_decision_payload_v1",
    "load_hdr_shot_reuse_veto_decision_v1",
    "map_hdr_shot_reuse_veto_v1",
]
