"""Deterministic factual cards for the private Look Approximation runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.film_physics.create_only_file import publish_create_only

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_SCHEMA = "kmcfm.private-product-runtime-receipt.v4"
_PROFILE_SCHEMA = "kmcfm.render-profile.v1"
_BUNDLE_SCHEMA = "kmcfm.private-look-approximation-fact-cards.v1"


class ProductFactCardError(ValueError):
    """Raised when a frozen fact-card input or output is inconsistent."""


def sha256_file(path: Path) -> str:
    """Hash one file without interpreting its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode canonical UTF-8 JSON with one trailing LF."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _exact_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductFactCardError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_frozen_inputs(
    *,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    profile: Mapping[str, Any],
    profile_sha256: str,
    catalog: Sequence[Mapping[str, Any]],
    evidence_sha256: Mapping[str, str],
    root_license_present: bool,
) -> list[dict[str, Any]]:
    inputs = config.get("input")
    if not isinstance(inputs, Mapping):
        raise ProductFactCardError("contract input is missing")
    if receipt_sha256 != inputs.get("runtime_receipt_sha256"):
        raise ProductFactCardError("runtime receipt SHA-256 mismatch")
    if profile_sha256 != inputs.get("profile_sha256"):
        raise ProductFactCardError("profile SHA-256 mismatch")
    if inputs.get("root_license_expected") != "absent" or root_license_present:
        raise ProductFactCardError("root licence state is not the frozen unresolved state")
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise ProductFactCardError("unsupported runtime receipt schema")
    claim = receipt.get("claim")
    if claim != {
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "public_release": False,
    }:
        raise ProductFactCardError("runtime claim ceiling drifted")
    binding = receipt.get("repository_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("installed_source_commit") != receipt.get("source_commit")
        or binding.get("runtime_scope_policy") != "exact-to-installed-source-commit"
        or binding.get("tracked_repository_policy") != "clean"
        or binding.get("runtime_scope_untracked_policy") != "reject"
    ):
        raise ProductFactCardError("runtime repository binding drifted")
    if profile.get("schema_id") != _PROFILE_SCHEMA:
        raise ProductFactCardError("unsupported render profile schema")
    evidence = profile.get("evidence")
    if (
        profile.get("profile_id") != "safe-rich-product-v1"
        or not isinstance(evidence, Mapping)
        or evidence.get("data_grade") != "none"
        or evidence.get("expert_grade") != "none"
        or evidence.get("method") != "heuristic"
        or evidence.get("calibrated_reference_allowed") is not False
    ):
        raise ProductFactCardError("render profile evidence boundary drifted")
    frozen_evidence = config.get("evidence")
    if not isinstance(frozen_evidence, Mapping) or set(evidence_sha256) != set(
        frozen_evidence
    ):
        raise ProductFactCardError("evidence inventory mismatch")
    for role, source in frozen_evidence.items():
        if not isinstance(source, Mapping):
            raise ProductFactCardError("evidence binding is malformed")
        expected = _exact_sha(source.get("sha256"), f"evidence {role}")
        if evidence_sha256[role] != expected:
            raise ProductFactCardError(f"evidence SHA-256 mismatch: {role}")

    expected_order = config.get("required_profile_order")
    rows = [dict(row) for row in catalog]
    if [row.get("look_id") for row in rows] != expected_order:
        raise ProductFactCardError("product catalog order or membership drifted")
    if len({row["look_id"] for row in rows}) != len(rows):
        raise ProductFactCardError("product catalog contains duplicates")
    required_keys = {
        "look_id",
        "display_name",
        "film_stock_id",
        "process_family",
        "look_family",
        "availability",
        "unavailable_reason",
        "evidence_tier",
        "claim_ceiling",
    }
    for index, row in enumerate(rows):
        if not required_keys.issubset(row):
            raise ProductFactCardError("product catalog row is incomplete")
        if index < 3:
            if (
                row["availability"] != "available"
                or row["unavailable_reason"] is not None
                or not isinstance(row["film_stock_id"], str)
                or "look-approximation" not in str(row["evidence_tier"])
                or "Look Approximation" not in str(row["claim_ceiling"])
            ):
                raise ProductFactCardError("available colour Look claim drifted")
        else:
            if (
                row["look_id"] != "generic_bw"
                or row["availability"] != "blocked_severe_artifact"
                or not isinstance(row["unavailable_reason"], str)
                or not row["unavailable_reason"]
                or row.get("availability_evidence_path")
                != "docs/evidence/BW2_D2_GENERIC_BW_POPULATION_SEVERE_REVIEW_RESULT.json"
                or row.get("availability_evidence_sha256")
                != "205040017bc15122d1d59c2e29708012395ca85e40baed65d7b58725161e796d"
            ):
                raise ProductFactCardError("generic B&W severe-veto boundary drifted")
    return rows


def build_product_fact_cards(
    *,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    profile: Mapping[str, Any],
    profile_sha256: str,
    catalog: Sequence[Mapping[str, Any]],
    evidence_sha256: Mapping[str, str],
    root_license_present: bool,
) -> dict[str, Any]:
    """Build and validate one fact-card bundle from frozen authoritative inputs."""

    rows = _validate_frozen_inputs(
        config=config,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        profile=profile,
        profile_sha256=profile_sha256,
        catalog=catalog,
        evidence_sha256=evidence_sha256,
        root_license_present=root_license_present,
    )
    profile_cards = []
    for row in rows:
        profile_cards.append(
            {
                "look_id": row["look_id"],
                "display_name": row["display_name"],
                "film_stock_id": row["film_stock_id"],
                "process_family": row["process_family"],
                "look_family": row["look_family"],
                "availability": row["availability"],
                "unavailable_reason": row["unavailable_reason"],
                "evidence_tier": row["evidence_tier"],
                "claim_ceiling": row["claim_ceiling"],
                "calibrated_stock_response": False,
                "physical_film_reproduction": False,
                **(
                    {
                        "availability_evidence": {
                            "path": row["availability_evidence_path"],
                            "sha256": row["availability_evidence_sha256"],
                        }
                    }
                    if row["availability"] != "available"
                    else {}
                ),
            }
        )
    frozen_evidence = config["evidence"]
    evidence_bindings = [
        {
            "role": role,
            "path": frozen_evidence[role]["path"],
            "sha256": evidence_sha256[role],
        }
        for role in sorted(frozen_evidence)
    ]
    cards = {
        "product": {
            "name": "K-MCFM",
            "product_mode": "private-repository-bound",
            "renderer": "deterministic bounded explicit colour operator",
            "entrypoints": ["cli", "desktop"],
            "platform": "Windows",
            "python": {
                "implementation": receipt["python"]["implementation"],
                "version": receipt["python"]["version"],
            },
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
        },
        "data": {
            "calibrated_stock_response_training_corpus": False,
            "product_training_dataset": None,
            "profile_data_grade": profile["evidence"]["data_grade"],
            "prohibited_inferences": [
                "manufacturer datasheets establish rendered stock authenticity",
                "unpaired scans establish a digital-to-film operator",
                "display proxies or presets establish calibrated stock response",
            ],
        },
        "model": {
            "learned_final_rgb_generator": False,
            "method": profile["evidence"]["method"],
            "engine_id": profile["engine_id"],
            "deterministic": True,
            "content_resampling": False,
            "claim_ceiling": profile["evidence"]["claim_ceiling"],
        },
        "profiles": profile_cards,
        "release": {
            "public_release": False,
            "root_license": "unresolved",
            "legal_clearance": False,
            "calibrated_stock_evidence": False,
            "signing": "pending",
            "installer_certification": "pending",
            "closed_beta": "pending",
            "runtime_receipt": {
                "schema": receipt["schema"],
                "sha256": receipt_sha256,
                "source_commit": receipt["source_commit"],
                "requirements_sha256": receipt["requirements"]["sha256"],
            },
        },
    }
    bundle = {
        "schema": _BUNDLE_SCHEMA,
        "card_order": list(config["required_card_order"]),
        "cards": cards,
        "evidence_bindings": evidence_bindings,
        "claim_ceiling": dict(config["claim_ceiling"]),
    }
    validate_product_fact_cards(bundle, config=config)
    return bundle


def validate_product_fact_cards(
    bundle: Mapping[str, Any], *, config: Mapping[str, Any]
) -> None:
    """Reject incomplete, reordered, or overclaiming fact-card bundles."""

    if bundle.get("schema") != _BUNDLE_SCHEMA:
        raise ProductFactCardError("fact-card schema mismatch")
    order = config.get("required_card_order")
    if bundle.get("card_order") != order:
        raise ProductFactCardError("fact-card order mismatch")
    cards = bundle.get("cards")
    if not isinstance(cards, Mapping) or list(cards) != order:
        raise ProductFactCardError("fact-card membership mismatch")
    profiles = cards["profiles"]
    if (
        not isinstance(profiles, list)
        or [row.get("look_id") for row in profiles]
        != config.get("required_profile_order")
    ):
        raise ProductFactCardError("profile-card membership mismatch")
    if [row["availability"] for row in profiles] != [
        "available",
        "available",
        "available",
        "blocked_severe_artifact",
    ]:
        raise ProductFactCardError("profile availability drifted")
    if any(
        row.get("calibrated_stock_response") is not False
        or row.get("physical_film_reproduction") is not False
        for row in profiles
    ):
        raise ProductFactCardError("profile claim exceeds Look Approximation")
    release = cards["release"]
    if (
        release.get("public_release") is not False
        or release.get("root_license") != "unresolved"
        or release.get("legal_clearance") is not False
        or release.get("calibrated_stock_evidence") is not False
    ):
        raise ProductFactCardError("release card overclaims readiness")
    claim = bundle.get("claim_ceiling")
    if claim != config.get("claim_ceiling"):
        raise ProductFactCardError("bundle claim ceiling drifted")


def publish_product_fact_cards(destination: Path, payload: bytes) -> Path:
    """Publish canonical cards create-only and clean only the owned stage."""

    destination = Path(destination)
    if not destination.parent.is_dir():
        raise ProductFactCardError("destination parent must exist")
    stage = destination.with_name(f".{destination.name}.u8-2b-{uuid.uuid4().hex}.stage")
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        publish_create_only(stage, destination)
    finally:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass
    return destination


__all__ = [
    "ProductFactCardError",
    "build_product_fact_cards",
    "canonical_json",
    "publish_product_fact_cards",
    "sha256_file",
    "validate_product_fact_cards",
]
