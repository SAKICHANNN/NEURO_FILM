from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.denmark_velvia_natural_ageing_source import (
    DenmarkVelviaSourceError,
    run_denmark_velvia_source_audit,
)


def _html(meta: list[tuple[str, str]]) -> bytes:
    rows = "".join(f'<meta name="{name}" content="{value}">' for name, value in meta)
    return f"<html><head>{rows}</head></html>".encode()


def _fixture(
    tmp_path: Path, *, data_ready: bool = False
) -> tuple[Path, dict[str, tuple[int, bytes]], str]:
    title = "Controlled Velvia study"
    authors = ["One Author", "Two Author"]
    pdf = b"exact-pdf"
    phrases = ["Velvia ISO 50", "colour checker", "Excel datasheet"]
    article_url = "https://example.test/article"
    pdf_url = "https://example.test/article.pdf"
    pure_url = "https://example.test/pure"
    config = {
        "experiment_id": "test",
        "source": {
            "article_url": article_url,
            "pdf_url": pdf_url,
            "pure_url": pure_url,
            "crossref_url": "https://example.test/crossref",
            "datacite_url": "https://example.test/datacite",
            "title": title,
            "authors": authors,
            "article_license_url": "https://creativecommons.org/licenses/by/4.0/",
            "ojs_doi_literal": "10.test//one",
            "pure_doi_rendering": "10.test/one",
            "pdf_bytes": len(pdf),
            "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
            "required_pdf_phrases": phrases,
            "expected_public_document_count": 1,
            "expected_public_document_kind": "article_pdf",
            "expected_crossref_status": 404,
            "expected_datacite_status": 404,
        },
        "physical_design": {"target_stock": "velvia_50"},
        "data_evidence": {
            "public_measurement_payload_count": int(data_ready),
            "public_sample_manifest_count": int(data_ready),
            "public_payload_checksum_count": int(data_ready),
            "explicit_measurement_data_license_count": int(data_ready),
            "public_exact_sample_identifier_count": int(data_ready),
            "prospective_development_group_count": int(data_ready),
            "sealed_confirmation_group_count": int(data_ready),
        },
        "admission_minimums": {
            "public_measurement_payload_count": 1,
            "public_sample_manifest_count": 1,
            "public_payload_checksum_count": 1,
            "explicit_measurement_data_license_count": 1,
            "public_exact_sample_identifier_count": 1,
            "prospective_development_group_count": 1,
            "sealed_confirmation_group_count": 1,
        },
        "operation_limits": {
            "official_document_get_requests": 5,
            "collection_object_requests": 0,
            "spreadsheet_requests": 0,
            "slide_image_requests": 0,
            "media_body_requests": 0,
            "pixel_decodes": 0,
            "figure_digitization_calls": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    article = _html(
        [("citation_title", title)]
        + [("citation_author", author) for author in authors]
        + [
            ("DC.Rights", "Copyright authors"),
            ("DC.Rights", config["source"]["article_license_url"]),
            ("citation_doi", config["source"]["ojs_doi_literal"]),
            ("citation_pdf_url", pdf_url),
        ]
    )
    pure = _html(
        [
            ("citation_title", title),
            ("citation_doi", config["source"]["pure_doi_rendering"]),
            ("citation_pdf_url", "https://example.test/pure/article.pdf"),
        ]
    )
    payloads = {
        article_url: (200, article),
        pdf_url: (200, pdf),
        pure_url: (200, pure),
        config["source"]["crossref_url"]: (404, b"missing"),
        config["source"]["datacite_url"]: (404, b"missing"),
    }
    return config_path, payloads, " ".join(phrases)


def test_ready_mock_source_is_exact_across_request_order(tmp_path: Path) -> None:
    config, payloads, pdf_text = _fixture(tmp_path, data_ready=True)
    requests: list[str] = []

    def fetcher(url: str) -> tuple[int, bytes]:
        requests.append(url)
        return payloads[url]

    forward = run_denmark_velvia_source_audit(
        config, fetcher=fetcher, pdf_text_extractor=lambda _: pdf_text
    )
    reverse = run_denmark_velvia_source_audit(
        config, reverse=True, fetcher=fetcher, pdf_text_extractor=lambda _: pdf_text
    )
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["audit_gates"].values())
    assert len(requests) == 10


def test_missing_public_data_fails_before_payload_reads(tmp_path: Path) -> None:
    config, payloads, pdf_text = _fixture(tmp_path)
    report = run_denmark_velvia_source_audit(
        config,
        fetcher=lambda url: payloads[url],
        pdf_text_extractor=lambda _: pdf_text,
    )
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert not any(report["admission_gates"].values())
    assert report["operation_counts"]["spreadsheet_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_wrong_pdf_identity_fails_source_gate(tmp_path: Path) -> None:
    config, payloads, pdf_text = _fixture(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    payloads[config_data["source"]["pdf_url"]] = (200, b"changed")
    report = run_denmark_velvia_source_audit(
        config,
        fetcher=lambda url: payloads[url],
        pdf_text_extractor=lambda _: pdf_text,
    )
    assert not report["audit_gates"]["ojs_pdf_identity_exact"]
    assert report["decision"] == "FAIL"


def test_rights_url_and_author_whitespace_are_canonicalized(tmp_path: Path) -> None:
    config, payloads, pdf_text = _fixture(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    article_url = config_data["source"]["article_url"]
    payloads[article_url] = (
        payloads[article_url][0],
        payloads[article_url][1]
        .replace(b"Two Author", b"Two  Author")
        .replace(b"licenses/by/4.0/", b"licenses/by/4.0"),
    )
    report = run_denmark_velvia_source_audit(
        config,
        fetcher=lambda url: payloads[url],
        pdf_text_extractor=lambda _: pdf_text,
    )
    assert report["audit_gates"]["article_title_authors_exact"]
    assert report["audit_gates"]["article_cc_by_4_exact"]


def test_missing_required_physical_statement_fails(tmp_path: Path) -> None:
    config, payloads, _ = _fixture(tmp_path)
    report = run_denmark_velvia_source_audit(
        config,
        fetcher=lambda url: payloads[url],
        pdf_text_extractor=lambda _: "Velvia ISO 50 colour checker",
    )
    assert not report["audit_gates"]["physical_design_statements_exact"]


def test_non_200_official_document_is_atomic(tmp_path: Path) -> None:
    config, payloads, pdf_text = _fixture(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    payloads[config_data["source"]["pure_url"]] = (503, b"")
    with pytest.raises(DenmarkVelviaSourceError):
        run_denmark_velvia_source_audit(
            config,
            fetcher=lambda url: payloads[url],
            pdf_text_extractor=lambda _: pdf_text,
        )


def test_project_contract_forbids_data_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root / "configs/sf3_a3y_denmark_velvia_natural_ageing_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "collection_object_requests",
        "spreadsheet_requests",
        "slide_image_requests",
        "media_body_requests",
        "pixel_decodes",
        "figure_digitization_calls",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
