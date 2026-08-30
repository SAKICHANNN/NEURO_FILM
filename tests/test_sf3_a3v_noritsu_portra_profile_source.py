from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.noritsu_portra_profile_source import run_source_audit


def _payloads(*, admitted: bool) -> tuple[dict[str, bytes], dict[str, object]]:
    texts = {
        "readme": b"real pair statement three frames",
        "license": b"MIT software documentation",
        "paired_audit": b"private not redistributed raw sha reference sha",
    }
    config: dict[str, object] = {
        "experiment_id": "test",
        "repositories": {
            "tool": {"repo": "owner/tool", "commit": "c1", "tree": "t1"},
            "audit": {"repo": "owner/audit", "commit": "c2", "tree": "t2"},
        },
        "assets": {
            "readme": {
                "repo_role": "tool",
                "path": "README.md",
                "bytes": len(texts["readme"]),
                "git_blob": "b1",
                "sha256": hashlib.sha256(texts["readme"]).hexdigest(),
                "raw_url": "https://raw/owner/tool/c1/README.md",
                "decode_text": True,
                "required_phrases": ["real pair statement", "three frames"],
            },
            "license": {
                "repo_role": "tool",
                "path": "LICENSE",
                "bytes": len(texts["license"]),
                "git_blob": "b2",
                "sha256": hashlib.sha256(texts["license"]).hexdigest(),
                "raw_url": "https://raw/owner/tool/c1/LICENSE",
                "decode_text": True,
                "required_phrases": ["MIT", "software documentation"],
            },
            "portra_profile": {
                "repo_role": "tool",
                "path": "profile.npz",
                "bytes": 12,
                "git_blob": "b3",
                "raw_url": "https://raw/owner/tool/c1/profile.npz",
                "decode_text": False,
                "required_phrases": [],
            },
            "paired_audit": {
                "repo_role": "audit",
                "path": "audit.py",
                "bytes": len(texts["paired_audit"]),
                "git_blob": "b4",
                "sha256": hashlib.sha256(texts["paired_audit"]).hexdigest(),
                "raw_url": "https://raw/owner/audit/c2/audit.py",
                "decode_text": True,
                "required_phrases": [
                    "private",
                    "not redistributed",
                    "raw sha",
                    "reference sha",
                ],
            },
        },
        "target_stock": {"film_stock_id": "portra", "public_label": "Portra"},
        "published_pair_assets": ["pair"] if admitted else [],
        "published_pair_rights": ["rights"] if admitted else [],
        "published_group_roles": ["groups"] if admitted else [],
        "published_pair_manifest": ["manifest"] if admitted else [],
        "operation_limits": {
            "commit_pinned_text_requests": 3,
            "profile_head_requests": 1,
            "profile_body_requests": 0,
            "owner_pair_requests": 0,
            "sample_image_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    responses: dict[str, bytes] = {}
    for role, asset in config["assets"].items():
        if asset["decode_text"]:
            responses[asset["raw_url"]] = texts[role]
    return responses, config


def _run(tmp_path: Path, *, admitted: bool, reverse: bool = False) -> dict[str, object]:
    responses, config = _payloads(admitted=admitted)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    order = (
        ("paired_audit", "portra_profile", "license", "readme")
        if reverse
        else ("readme", "license", "portra_profile", "paired_audit")
    )
    return run_source_audit(
        path,
        body_reader=responses.__getitem__,
        head_reader=lambda url: {
            "status": 200,
            "content_length": 12,
            "content_type": "application/octet-stream",
        },
        asset_order=order,
    )


def test_private_pair_source_fails_without_reading_profile(tmp_path: Path) -> None:
    report = _run(tmp_path, admitted=False)
    assert report["decision"] == "FAIL"
    assert report["gates"]["real_portra_machine_pair_statements_present"]
    assert not report["gates"]["owner_pair_payload_publicly_addressable"]
    assert not report["source_identity"]["assets"]["portra_profile"]["body_read"]
    assert report["operation_counts"]["profile_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_mock_source_only_opens_next_audit(tmp_path: Path) -> None:
    forward = _run(tmp_path, admitted=True)
    reverse = _run(tmp_path, admitted=True, reverse=True)
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["gates"].values())


def test_asset_order_must_be_exact(tmp_path: Path) -> None:
    responses, config = _payloads(admitted=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="every frozen asset"):
        run_source_audit(
            path,
            body_reader=responses.__getitem__,
            head_reader=lambda url: {
                "status": 200,
                "content_length": 12,
                "content_type": "application/octet-stream",
            },
            asset_order=("readme", "readme", "license", "paired_audit"),
        )


def test_project_contract_forbids_profile_pair_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root / "configs/sf3_a3v_noritsu_portra_physical_profile_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["target_stock"]["film_stock_id"] == "kodak_portra_400"
    assert config["assets"]["portra_profile"]["decode_text"] is False
    assert all(
        config["operation_limits"][key] == 0
        for key in config["operation_limits"]
        if key not in {"commit_pinned_text_requests", "profile_head_requests"}
    )
