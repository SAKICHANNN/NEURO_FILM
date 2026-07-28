"""Versioned, capability-neutral lock for exact producer invocation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .contracts import ReferenceMatchContractError
from .dpct_adapter import DPCT_PRODUCER_PROFILE_ID


DPCT_INVOCATION_PROFILE_SCHEMA_V2 = (
    "neuro-film.dpct-invocation-profile.v2"
)
DPCT_INVOCATION_PROFILE_DOMAIN_V2 = (
    b"NeuroFilmDpctInvocationProfileV2\0"
)
DPCT_INVOCATION_LOWER_PROFILE_V2 = "neuro-film.dpct-consumer.v2"
DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2 = (
    "candidate-only-not-promoted-not-applied-not-delivered"
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class DpctInvocationProfileV2:
    schema_id: str
    profile_id: str
    producer_repository_id: str
    producer_stable_commit: str
    producer_package_source_commit: str
    producer_fixture_commit: str
    producer_package_lock_sha256: str
    package_distribution: str
    package_version: str
    package_entrypoint: str
    wheel_filename: str
    wheel_size_bytes: int
    wheel_sha256: str
    python_major_minor: str
    numpy_version: str
    pillow_version: str
    request_schema: str
    request_schema_sha256: str
    response_schema: str
    response_schema_sha256: str
    fixture_sha256: str
    capability_id: str
    producer_profile_id: str
    lower_compatibility_profile_id: str
    evaluation_allowed: bool
    redistribution_allowed: bool
    rights_reason: str
    claim_ceiling: str


def _strict(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from invocation profile v2"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ReferenceMatchContractError(f"{label} is invalid")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ReferenceMatchContractError(f"{label} is invalid")
    return result


def _hash(value: Any, label: str) -> str:
    result = _text(value, label)
    if _HEX64.fullmatch(result) is None:
        raise ReferenceMatchContractError(f"{label} is invalid")
    return result


def _commit(value: Any, label: str) -> str:
    result = _text(value, label)
    if _HEX40.fullmatch(result) is None:
        raise ReferenceMatchContractError(f"{label} is invalid")
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
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
            "invocation profile v2 is not canonicalizable"
        ) from exc


def dpct_invocation_profile_payload_v2(
    value: DpctInvocationProfileV2,
) -> dict[str, Any]:
    return {
        "schema_id": value.schema_id,
        "profile_id": value.profile_id,
        "producer": {
            "repository_id": value.producer_repository_id,
            "stable_commit": value.producer_stable_commit,
            "package_source_commit": (
                value.producer_package_source_commit
            ),
            "fixture_commit": value.producer_fixture_commit,
            "package_lock_sha256": value.producer_package_lock_sha256,
        },
        "package": {
            "distribution": value.package_distribution,
            "version": value.package_version,
            "entrypoint": value.package_entrypoint,
            "wheel_filename": value.wheel_filename,
            "wheel_size_bytes": value.wheel_size_bytes,
            "wheel_sha256": value.wheel_sha256,
        },
        "runtime": {
            "python_major_minor": value.python_major_minor,
            "numpy": value.numpy_version,
            "pillow": value.pillow_version,
        },
        "wire": {
            "request_schema": value.request_schema,
            "request_schema_sha256": value.request_schema_sha256,
            "response_schema": value.response_schema,
            "response_schema_sha256": value.response_schema_sha256,
            "fixture_sha256": value.fixture_sha256,
            "capability_id": value.capability_id,
            "profile_id": value.producer_profile_id,
            "lower_compatibility_profile_id": (
                value.lower_compatibility_profile_id
            ),
        },
        "rights": {
            "evaluation_allowed": value.evaluation_allowed,
            "redistribution_allowed": value.redistribution_allowed,
            "reason": value.rights_reason,
        },
        "claim_ceiling": value.claim_ceiling,
    }


def dpct_invocation_profile_id_v2(
    value: DpctInvocationProfileV2,
) -> str:
    payload = dpct_invocation_profile_payload_v2(value)
    payload["profile_id"] = "0" * 64
    return hashlib.sha256(
        DPCT_INVOCATION_PROFILE_DOMAIN_V2 + _canonical_bytes(payload)
    ).hexdigest()


def load_dpct_invocation_profile_v2(
    value: Any,
) -> DpctInvocationProfileV2:
    root = _strict(
        value,
        {
            "schema_id",
            "profile_id",
            "producer",
            "package",
            "runtime",
            "wire",
            "rights",
            "claim_ceiling",
        },
        "invocation profile v2",
    )
    producer = _strict(
        root["producer"],
        {
            "repository_id",
            "stable_commit",
            "package_source_commit",
            "fixture_commit",
            "package_lock_sha256",
        },
        "invocation profile v2 producer",
    )
    package = _strict(
        root["package"],
        {
            "distribution",
            "version",
            "entrypoint",
            "wheel_filename",
            "wheel_size_bytes",
            "wheel_sha256",
        },
        "invocation profile v2 package",
    )
    runtime = _strict(
        root["runtime"],
        {"python_major_minor", "numpy", "pillow"},
        "invocation profile v2 runtime",
    )
    wire = _strict(
        root["wire"],
        {
            "request_schema",
            "request_schema_sha256",
            "response_schema",
            "response_schema_sha256",
            "fixture_sha256",
            "capability_id",
            "profile_id",
            "lower_compatibility_profile_id",
        },
        "invocation profile v2 wire",
    )
    rights = _strict(
        root["rights"],
        {
            "evaluation_allowed",
            "redistribution_allowed",
            "reason",
        },
        "invocation profile v2 rights",
    )
    wheel_size = package["wheel_size_bytes"]
    if (
        isinstance(wheel_size, bool)
        or not isinstance(wheel_size, int)
        or not 0 < wheel_size <= 1024 * 1024 * 1024
    ):
        raise ReferenceMatchContractError(
            "invocation profile v2 wheel size is invalid"
        )
    filename = _text(package["wheel_filename"], "wheel filename")
    path = PurePosixPath(filename.replace("\\", "/"))
    if path.name != filename or len(path.parts) != 1:
        raise ReferenceMatchContractError(
            "invocation profile v2 wheel filename is unsafe"
        )
    evaluation_allowed = rights["evaluation_allowed"]
    redistribution_allowed = rights["redistribution_allowed"]
    if not isinstance(evaluation_allowed, bool) or not isinstance(
        redistribution_allowed, bool
    ):
        raise ReferenceMatchContractError(
            "invocation profile v2 rights flags are invalid"
        )
    if not evaluation_allowed:
        raise ReferenceMatchContractError(
            "invocation profile v2 does not permit evaluation"
        )
    result = DpctInvocationProfileV2(
        schema_id=_text(root["schema_id"], "schema_id"),
        profile_id=_hash(root["profile_id"], "profile_id"),
        producer_repository_id=_identifier(
            producer["repository_id"], "producer repository_id"
        ),
        producer_stable_commit=_commit(
            producer["stable_commit"], "producer stable_commit"
        ),
        producer_package_source_commit=_commit(
            producer["package_source_commit"],
            "producer package_source_commit",
        ),
        producer_fixture_commit=_commit(
            producer["fixture_commit"], "producer fixture_commit"
        ),
        producer_package_lock_sha256=_hash(
            producer["package_lock_sha256"],
            "producer package_lock_sha256",
        ),
        package_distribution=_identifier(
            package["distribution"], "package distribution"
        ),
        package_version=_identifier(
            package["version"], "package version"
        ),
        package_entrypoint=_identifier(
            package["entrypoint"], "package entrypoint"
        ),
        wheel_filename=filename,
        wheel_size_bytes=wheel_size,
        wheel_sha256=_hash(package["wheel_sha256"], "wheel_sha256"),
        python_major_minor=_text(
            runtime["python_major_minor"], "runtime python"
        ),
        numpy_version=_text(runtime["numpy"], "runtime numpy"),
        pillow_version=_text(runtime["pillow"], "runtime pillow"),
        request_schema=_identifier(
            wire["request_schema"], "request schema"
        ),
        request_schema_sha256=_hash(
            wire["request_schema_sha256"], "request schema hash"
        ),
        response_schema=_identifier(
            wire["response_schema"], "response schema"
        ),
        response_schema_sha256=_hash(
            wire["response_schema_sha256"], "response schema hash"
        ),
        fixture_sha256=_hash(
            wire["fixture_sha256"], "fixture hash"
        ),
        capability_id=_identifier(
            wire["capability_id"], "capability_id"
        ),
        producer_profile_id=_identifier(
            wire["profile_id"], "producer profile_id"
        ),
        lower_compatibility_profile_id=_identifier(
            wire["lower_compatibility_profile_id"],
            "lower compatibility profile_id",
        ),
        evaluation_allowed=evaluation_allowed,
        redistribution_allowed=redistribution_allowed,
        rights_reason=_text(rights["reason"], "rights reason"),
        claim_ceiling=_text(root["claim_ceiling"], "claim ceiling"),
    )
    if result.schema_id != DPCT_INVOCATION_PROFILE_SCHEMA_V2:
        raise ReferenceMatchContractError(
            "invocation profile v2 schema_id mismatch"
        )
    if (
        result.producer_profile_id != DPCT_PRODUCER_PROFILE_ID
        or result.lower_compatibility_profile_id
        != DPCT_INVOCATION_LOWER_PROFILE_V2
    ):
        raise ReferenceMatchContractError(
            "invocation profile v2 color compatibility mismatch"
        )
    if (
        result.claim_ceiling
        != DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2
    ):
        raise ReferenceMatchContractError(
            "invocation profile v2 claim ceiling mismatch"
        )
    if result.profile_id != dpct_invocation_profile_id_v2(result):
        raise ReferenceMatchContractError(
            "invocation profile v2 identity mismatch"
        )
    return result


def issue_dpct_invocation_profile_v2(
    value: DpctInvocationProfileV2,
) -> DpctInvocationProfileV2:
    provisional = replace(
        value,
        schema_id=DPCT_INVOCATION_PROFILE_SCHEMA_V2,
        profile_id="0" * 64,
        claim_ceiling=DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2,
    )
    issued = replace(
        provisional,
        profile_id=dpct_invocation_profile_id_v2(provisional),
    )
    return load_dpct_invocation_profile_v2(
        dpct_invocation_profile_payload_v2(issued)
    )


__all__ = [
    "DPCT_INVOCATION_LOWER_PROFILE_V2",
    "DPCT_INVOCATION_PROFILE_CLAIM_CEILING_V2",
    "DPCT_INVOCATION_PROFILE_SCHEMA_V2",
    "DpctInvocationProfileV2",
    "dpct_invocation_profile_id_v2",
    "dpct_invocation_profile_payload_v2",
    "issue_dpct_invocation_profile_v2",
    "load_dpct_invocation_profile_v2",
]
