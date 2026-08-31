from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.hershenson_allen_velvia_archive_source import (
    HershensonAllenSourceError,
    _canonical_html_identity,
    _parse_html,
    run_hershenson_allen_source_audit,
)


def _fixture(
    tmp_path: Path, *, data_ready: bool = False
) -> tuple[Path, dict[str, tuple[int, bytes]]]:
    pages = {
        "home": b"<html><body>count fixed workflow Velvia rights</body></html>",
        "collection": b"<html><body>collection history</body></html>",
        "explore": b"<html><body>inventory statement</body></html>",
        "prospectus": b'<html><body><a href="https://example.test/p.pdf">PDF</a></body></html>',
        "robots": b"Content-Signal: search=yes,ai-train=no,use=reference\nrights",
    }
    urls = {role: f"https://example.test/{role}" for role in pages}
    keys = [
        "public_item_level_velvia_identity_count",
        "public_machine_readable_manifest_count",
        "public_payload_checksum_count",
        "explicit_image_data_fitting_license_count",
        "public_scanner_process_group_count",
        "independent_same_object_reference_pair_count",
        "prospective_development_group_count",
        "sealed_confirmation_group_count",
    ]
    config = {
        "experiment_id": "test",
        "source": {
            "home_url": urls["home"],
            "collection_url": urls["collection"],
            "explore_url": urls["explore"],
            "prospectus_page_url": urls["prospectus"],
            "robots_url": urls["robots"],
            "prospectus_pdf_url": "https://example.test/p.pdf",
            "expected_responses": {
                role: (
                    {
                        "identity_kind": "raw_bytes",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                    if role == "robots"
                    else {
                        "identity_kind": "canonical_visible_text_and_non_protection_links",
                        "sha256": _canonical_html_identity(*_parse_html(payload)),
                    }
                )
                for role, payload in pages.items()
            },
            "required_home_phrases": ["count fixed workflow Velvia rights"],
            "required_collection_phrases": ["collection history"],
            "required_explore_phrases": ["inventory statement"],
            "required_robots_phrases": [
                "Content-Signal: search=yes,ai-train=no,use=reference",
                "rights",
            ],
            "archive_image_count": 10,
            "unique_film_count": 8,
            "stock_example": "velvia_50",
        },
        "data_evidence": {key: int(data_ready) for key in keys},
        "admission_minimums": {key: 1 for key in keys},
        "operation_limits": {
            "official_text_get_requests": 5,
            "prospectus_pdf_requests": 0,
            "database_queries": 0,
            "poster_image_requests": 0,
            "transparency_requests": 0,
            "thumbnail_requests": 0,
            "media_body_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    payloads = {urls[role]: (200, payload) for role, payload in pages.items()}
    return path, payloads


def test_ready_mock_source_is_exact_across_request_order(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path, data_ready=True)
    requests: list[str] = []

    def fetcher(url: str) -> tuple[int, bytes]:
        requests.append(url)
        return payloads[url]

    forward = run_hershenson_allen_source_audit(config, fetcher=fetcher)
    reverse = run_hershenson_allen_source_audit(
        config, reverse=True, fetcher=fetcher
    )
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["audit_gates"].values())
    assert len(requests) == 10


def test_missing_item_rights_manifest_and_roles_fail_before_media(
    tmp_path: Path,
) -> None:
    config, payloads = _fixture(tmp_path)
    report = run_hershenson_allen_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["operation_counts"]["database_queries"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_changed_response_identity_fails_source_gate(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    payloads[data["source"]["home_url"]] = (200, b"changed")
    report = run_hershenson_allen_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert not report["audit_gates"]["official_response_identities_exact"]
    assert report["decision"] == "FAIL"


def test_missing_prospectus_link_fails_without_pdf_request(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    payloads[data["source"]["prospectus_page_url"]] = (
        200,
        b"<html><body>no link</body></html>",
    )
    data["source"]["expected_responses"]["prospectus"] = {
        "identity_kind": "canonical_visible_text_and_non_protection_links",
        "sha256": _canonical_html_identity(
            *_parse_html(payloads[data["source"]["prospectus_page_url"]][1])
        ),
    }
    config.write_text(json.dumps(data), encoding="utf-8")
    report = run_hershenson_allen_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert not report["audit_gates"]["prospectus_link_exact_without_pdf_read"]
    assert report["operation_counts"]["prospectus_pdf_requests"] == 0


def test_non_200_source_is_atomic(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    payloads[data["source"]["explore_url"]] = (503, b"")
    with pytest.raises(HershensonAllenSourceError):
        run_hershenson_allen_source_audit(config, fetcher=lambda url: payloads[url])


def test_cloudflare_email_fragment_is_excluded_from_html_identity() -> None:
    first = b'<html><body>stable<a href="/cdn-cgi/l/email-protection#abc">mail</a></body></html>'
    second = b'<html><body>stable<a href="/cdn-cgi/l/email-protection#def">mail</a></body></html>'
    assert _canonical_html_identity(*_parse_html(first)) == _canonical_html_identity(
        *_parse_html(second)
    )


def test_project_contract_forbids_database_media_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs/sf3_a3z_hershenson_allen_velvia_archive_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "prospectus_pdf_requests",
        "database_queries",
        "poster_image_requests",
        "transparency_requests",
        "thumbnail_requests",
        "media_body_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
