"""Zero-pixel Denmark Velvia 50 natural-ageing source readiness audit."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class DenmarkVelviaSourceError(ValueError):
    """Raised when an official source violates the frozen contract."""


FetchResult = tuple[int, bytes]
Fetcher = Callable[[str], FetchResult]
PdfTextExtractor = Callable[[bytes], str]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _default_fetcher(url: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "K-MCFM-source-audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


def _default_pdf_text_extractor(payload: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:  # pragma: no cover - dependency error detail is wrapped
        raise DenmarkVelviaSourceError("published PDF could not be parsed") from error


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        row = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            name = row.get("name") or row.get("property")
            if name:
                self.meta.setdefault(name, []).append(row.get("content", ""))
        elif tag.lower() in {"a", "link"}:
            self.links.append(row)


def _parse_html(payload: bytes) -> _MetadataParser:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DenmarkVelviaSourceError("official HTML is not UTF-8") from error
    parser = _MetadataParser()
    parser.feed(text)
    return parser


def _require_single(values: list[str] | None, label: str) -> str:
    unique = sorted(set(values or []))
    if len(unique) != 1:
        raise DenmarkVelviaSourceError(f"expected one exact {label}")
    return unique[0]


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def run_denmark_velvia_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
    pdf_text_extractor: PdfTextExtractor | None = None,
) -> dict[str, Any]:
    """Run the frozen official-document-only source-readiness audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    fetch = fetcher or _default_fetcher
    extract_pdf_text = pdf_text_extractor or _default_pdf_text_extractor
    roles = ["article", "pdf", "pure", "crossref", "datacite"]
    if reverse:
        roles.reverse()
    urls = {
        "article": source["article_url"],
        "pdf": source["pdf_url"],
        "pure": source["pure_url"],
        "crossref": source["crossref_url"],
        "datacite": source["datacite_url"],
    }
    fetched: dict[str, FetchResult] = {}
    for role in roles:
        fetched[role] = fetch(urls[role])

    for role in ("article", "pdf", "pure"):
        if fetched[role][0] != 200:
            raise DenmarkVelviaSourceError(f"{role} source returned non-200 status")

    article = _parse_html(fetched["article"][1])
    pure = _parse_html(fetched["pure"][1])
    article_title = _require_single(article.meta.get("citation_title"), "article title")
    article_authors = article.meta.get("citation_author", [])
    article_rights = sorted(set(article.meta.get("DC.Rights", [])))
    article_doi = _require_single(article.meta.get("citation_doi"), "OJS DOI")
    article_pdf_url = _require_single(
        article.meta.get("citation_pdf_url"), "article PDF URL"
    )
    pure_title = _require_single(pure.meta.get("citation_title"), "Pure title")
    pure_doi = _require_single(pure.meta.get("citation_doi"), "Pure DOI")
    pure_pdf_urls = sorted(set(pure.meta.get("citation_pdf_url", [])))
    pdf_payload = fetched["pdf"][1]
    pdf_text = _normalized_text(extract_pdf_text(pdf_payload))
    phrase_results = {
        phrase: _normalized_text(phrase) in pdf_text
        for phrase in source["required_pdf_phrases"]
    }

    operation_counts = dict(config["operation_limits"])
    audit_gates = {
        "official_document_statuses_exact": all(
            fetched[role][0] == 200 for role in ("article", "pdf", "pure")
        ),
        "article_title_authors_exact": article_title == source["title"]
        and article_authors == source["authors"],
        "article_cc_by_4_exact": source["article_license_url"] in article_rights,
        "ojs_pdf_identity_exact": article_pdf_url == source["pdf_url"]
        and len(pdf_payload) == int(source["pdf_bytes"])
        and _sha256(pdf_payload) == source["pdf_sha256"],
        "physical_design_statements_exact": all(phrase_results.values()),
        "pure_article_identity_exact": pure_title == source["title"]
        and pure_doi == source["pure_doi_rendering"]
        and len(pure_pdf_urls) == int(source["expected_public_document_count"]),
        "doi_catalogue_renderings_recorded": article_doi == source["ojs_doi_literal"]
        and pure_doi == source["pure_doi_rendering"],
        "registry_absence_exact": fetched["crossref"][0]
        == int(source["expected_crossref_status"])
        and fetched["datacite"][0] == int(source["expected_datacite_status"]),
        "zero_collection_data_pixel_and_model_reads": all(
            int(operation_counts[key]) == 0
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
            )
        ),
    }
    data_evidence = config["data_evidence"]
    minimums = config["admission_minimums"]
    admission_gates = {
        key: int(data_evidence[key]) >= int(minimums[key]) for key in sorted(minimums)
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    canonical_source = {
        "title": article_title,
        "authors": article_authors,
        "article_license_url": source["article_license_url"],
        "ojs_doi_literal": article_doi,
        "pure_doi_rendering": pure_doi,
        "pdf_bytes": len(pdf_payload),
        "pdf_sha256": _sha256(pdf_payload),
        "physical_design": config["physical_design"],
    }
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3y-denmark-velvia-natural-ageing-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source": canonical_source,
        "source_identity_sha256": _json_sha256(canonical_source),
        "official_statuses": {role: fetched[role][0] for role in sorted(fetched)},
        "public_documents": {
            "count": len(pure_pdf_urls),
            "kind": source["expected_public_document_kind"],
            "pdf_urls": pure_pdf_urls,
        },
        "required_pdf_phrase_results": phrase_results,
        "data_evidence": data_evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report
