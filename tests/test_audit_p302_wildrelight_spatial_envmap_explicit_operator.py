from __future__ import annotations

import hashlib
import json

from scripts.audit_p302_wildrelight_spatial_envmap_explicit_operator import (
    build_evidence,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_evidence_preserves_candidate_boundary_and_development_pass() -> None:
    config = {
        "claim_ceiling": "private precursor only",
        "source": {"member_manifest": {"bytes": 10, "sha256": "manifest-sha"}},
    }
    config_body = _canonical(config)
    source_lock = {
        "archive_or_payload_requests": 0,
        "member_manifest_bytes": 10,
        "member_manifest_sha256": "manifest-sha",
        "pixel_decodes": 0,
    }
    acquisition = {
        "config_sha256": hashlib.sha256(config_body).hexdigest(),
        "members_exact": True,
        "roles": ["training", "development"],
    }
    report = {
        "candidate_count": "2/3",
        "confirmation_member_reads": 0,
        "decision": "PASS_PRIVATE_P302_DEVELOPMENT",
        "gates": {"all": True},
        "model_bytes": 123,
        "model_sha256": "model-sha",
        "network_bytes": 0,
        "phase": "development",
        "reserve_member_reads": 0,
        "role_partition": {"scene_overlap_zero": True},
        "scientific_identity": "sha256:science",
        "summary": {"candidate_control_rate": 1.0},
        "target_reads_before_model_lock": 0,
    }
    evidence = build_evidence(
        config_body=config_body,
        config=config,
        source_lock_body=_canonical(source_lock),
        source_lock=source_lock,
        acquisition_body=_canonical(acquisition),
        acquisition=acquisition,
        report_body=_canonical(report),
        report=report,
        head="committed-head",
        phase="development",
    )
    assert evidence["decision"] == "PASS_PRIVATE_P302_DEVELOPMENT"
    assert evidence["candidate_count"] == "2/3"
    assert all(evidence["gates"].values())


def test_evidence_fails_closed_on_confirmation_read() -> None:
    config = {
        "claim_ceiling": "private precursor only",
        "source": {"member_manifest": {"bytes": 10, "sha256": "manifest-sha"}},
    }
    config_body = _canonical(config)
    source_lock = {
        "archive_or_payload_requests": 0,
        "member_manifest_bytes": 10,
        "member_manifest_sha256": "manifest-sha",
        "pixel_decodes": 0,
    }
    acquisition = {
        "config_sha256": hashlib.sha256(config_body).hexdigest(),
        "members_exact": True,
        "roles": ["training", "development"],
    }
    report = {
        "candidate_count": "2/3",
        "confirmation_member_reads": 1,
        "decision": "PASS_PRIVATE_P302_DEVELOPMENT",
        "gates": {"all": True},
        "network_bytes": 0,
        "phase": "development",
        "reserve_member_reads": 0,
        "role_partition": {"scene_overlap_zero": True},
    }
    evidence = build_evidence(
        config_body=config_body,
        config=config,
        source_lock_body=_canonical(source_lock),
        source_lock=source_lock,
        acquisition_body=_canonical(acquisition),
        acquisition=acquisition,
        report_body=_canonical(report),
        report=report,
        head="committed-head",
        phase="development",
    )
    assert evidence["decision"] == "FAIL_CLOSED_P302_DEVELOPMENT"
