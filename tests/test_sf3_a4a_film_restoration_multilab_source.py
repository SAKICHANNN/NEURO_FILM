from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.film_restoration_multilab_source import (
    FilmRestorationMultilabSourceError,
    run_film_restoration_multilab_source_audit,
)


def _fixture(
    tmp_path: Path, *, data_ready: bool = False
) -> tuple[Path, dict[str, tuple[int, bytes]]]:
    urls = {
        role: f"https://example.test/{role}"
        for role in ("elsevier", "crossref", "datacite")
    }
    keys = [
        "public_observation_payload_count",
        "public_machine_readable_manifest_count",
        "public_payload_checksum_count",
        "explicit_observation_data_fitting_license_count",
        "exact_laboratory_identity_count",
        "exact_scanner_workflow_identity_count",
        "prospective_development_group_count",
        "sealed_confirmation_group_count",
        "exact_film_stock_identity_count",
    ]
    topology = {
        "film_title": "fixture",
        "film_year": 1972,
        "laboratory_count": 6,
        "scene_count": 3,
        "frames_per_scene": 24,
        "common_source": "original_camera_negative",
        "participant_reference": "positive_print",
        "independent_reference": "director_of_photography_supervised_restoration",
        "independent_reference_withheld_from_participants": True,
        "film_stock_identity": "unknown",
    }
    config = {
        "experiment_id": "test",
        "source": {
            "doi": "10.test/example",
            **{f"{role}_url": url for role, url in urls.items()},
            "official_article_url": "https://example.test/article",
            "institutional_record_url": "https://example.test/record",
            "expected": {
                "title": "title",
                "journal": "journal",
                "cover_date": "2026-06-30",
                "pii_compact": "PII",
                "pii_formatted": "PII-F",
                "openaccess": "0",
                "authors": ["A One", "B Two"],
                "crossref_relation": {},
                "datacite_exact_title_total": 0,
            },
            "public_article_topology": topology,
        },
        "data_evidence": {key: 6 if data_ready else 0 for key in keys},
        "admission_minimums": {key: 1 for key in keys},
        "operation_limits": {
            "official_metadata_get_requests": 3,
            "official_article_html_requests": 0,
            "institutional_file_requests": 0,
            "article_pdf_requests": 0,
            "article_media_requests": 0,
            "source_negative_requests": 0,
            "positive_print_requests": 0,
            "laboratory_version_requests": 0,
            "independent_reference_requests": 0,
            "frame_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    payloads = {
        urls["elsevier"]: (
            200,
            json.dumps(
                {
                    "full-text-retrieval-response": {
                        "coredata": {
                            "prism:doi": "10.test/example",
                            "dc:title": "title ",
                            "prism:publicationName": "journal",
                            "prism:coverDate": "2026-06-30",
                            "eid": "1-s2.0-PII",
                            "pii": "PII-F",
                            "openaccess": "0",
                            "openaccessUserLicense": None,
                        }
                    }
                }
            ).encode(),
        ),
        urls["crossref"]: (
            200,
            json.dumps(
                {
                    "message": {
                        "DOI": "10.test/example",
                        "title": ["title"],
                        "author": [
                            {"given": "A", "family": "One"},
                            {"given": "B", "family": "Two"},
                        ],
                        "relation": {},
                        "license": [
                            {
                                "content-version": "tdm",
                                "URL": "https://example.test/tdm",
                            }
                        ],
                    }
                }
            ).encode(),
        ),
        urls["datacite"]: (
            200,
            json.dumps({"data": [], "meta": {"total": 0}}).encode(),
        ),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, payloads


def test_ready_mock_source_is_exact_across_request_order(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path, data_ready=True)
    requests: list[str] = []

    def fetcher(url: str) -> tuple[int, bytes]:
        requests.append(url)
        return payloads[url]

    forward = run_film_restoration_multilab_source_audit(
        config, fetcher=fetcher
    )
    reverse = run_film_restoration_multilab_source_audit(
        config, reverse=True, fetcher=fetcher
    )
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["audit_gates"].values())
    assert len(requests) == 6


def test_missing_data_rights_groups_and_stock_fail_before_media(
    tmp_path: Path,
) -> None:
    config, payloads = _fixture(tmp_path)
    report = run_film_restoration_multilab_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["operation_counts"]["source_negative_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_changed_bibliographic_identity_fails_source_gate(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    role_url = data["source"]["elsevier_url"]
    value = json.loads(payloads[role_url][1])
    value["full-text-retrieval-response"]["coredata"]["dc:title"] = "changed"
    payloads[role_url] = (200, json.dumps(value).encode())
    report = run_film_restoration_multilab_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert not report["audit_gates"]["elsevier_bibliographic_identity_exact"]
    assert report["decision"] == "FAIL"


def test_datacite_dataset_record_changes_zero_dataset_gate(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    role_url = data["source"]["datacite_url"]
    payloads[role_url] = (
        200,
        json.dumps({"data": [{"id": "new"}], "meta": {"total": 1}}).encode(),
    )
    report = run_film_restoration_multilab_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert not report["audit_gates"][
        "datacite_exact_title_has_no_dataset_record"
    ]
    assert report["decision"] == "FAIL"


def test_non_200_source_is_atomic(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    payloads[data["source"]["crossref_url"]] = (503, b"")
    with pytest.raises(FilmRestorationMultilabSourceError):
        run_film_restoration_multilab_source_audit(
            config, fetcher=lambda url: payloads[url]
        )


def test_malformed_json_is_atomic(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    payloads[data["source"]["datacite_url"]] = (200, b"not-json")
    with pytest.raises(FilmRestorationMultilabSourceError):
        run_film_restoration_multilab_source_audit(
            config, fetcher=lambda url: payloads[url]
        )


def test_project_contract_forbids_media_pixel_and_model_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs/sf3_a4a_film_restoration_multilab_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "official_article_html_requests",
        "institutional_file_requests",
        "article_pdf_requests",
        "article_media_requests",
        "source_negative_requests",
        "positive_print_requests",
        "laboratory_version_requests",
        "independent_reference_requests",
        "frame_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
