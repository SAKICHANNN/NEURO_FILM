"""Zero-media multi-lab colour-film restoration source-readiness audit."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


class FilmRestorationMultilabSourceError(ValueError):
    """Raised when an official metadata source violates the frozen contract."""


FetchResult = tuple[int, bytes]
Fetcher = Callable[[str], FetchResult]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return _sha256(_json_bytes(value))


def _default_fetcher(url: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "K-MCFM-source-audit/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:
            status = int(error.code)
            payload = error.read()
            if status != 429 and status < 500:
                return status, payload
            last_error = error
            if attempt < 2:
                time.sleep(1 << attempt)
                continue
            return status, payload
        except urllib.error.URLError as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 << attempt)
    raise FilmRestorationMultilabSourceError(
        "official metadata request failed after bounded retries"
    ) from last_error


def _load_json(payload: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FilmRestorationMultilabSourceError(
            f"{role} response is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise FilmRestorationMultilabSourceError(
            f"{role} response is not a JSON object"
        )
    return value


def _selected_elsevier(value: dict[str, Any]) -> dict[str, Any]:
    try:
        core = value["full-text-retrieval-response"]["coredata"]
        return {
            "doi": str(core["prism:doi"]).strip(),
            "title": str(core["dc:title"]).strip(),
            "journal": str(core["prism:publicationName"]).strip(),
            "cover_date": str(core["prism:coverDate"]).strip(),
            "pii_compact": str(core["eid"]).removeprefix("1-s2.0-").strip(),
            "pii_formatted": str(core["pii"]).strip(),
            "openaccess": str(core["openaccess"]).strip(),
            "openaccess_user_license": core.get("openaccessUserLicense"),
        }
    except (KeyError, TypeError) as error:
        raise FilmRestorationMultilabSourceError(
            "Elsevier response lacks frozen core metadata"
        ) from error


def _selected_crossref(value: dict[str, Any]) -> dict[str, Any]:
    try:
        message = value["message"]
        authors = [
            f"{row['given']} {row['family']}" for row in message["author"]
        ]
        licenses = [
            {
                "content_version": row.get("content-version"),
                "url": row.get("URL"),
            }
            for row in message.get("license", [])
        ]
        return {
            "doi": str(message["DOI"]).strip(),
            "title": str(message["title"][0]).strip(),
            "authors": authors,
            "relation": message.get("relation", {}),
            "licenses": licenses,
        }
    except (KeyError, TypeError, IndexError) as error:
        raise FilmRestorationMultilabSourceError(
            "Crossref response lacks frozen work metadata"
        ) from error


def _selected_datacite(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "total": int(value["meta"]["total"]),
            "record_count": len(value["data"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise FilmRestorationMultilabSourceError(
            "DataCite response lacks frozen exact-title result"
        ) from error


def run_film_restoration_multilab_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen official-metadata-only source-readiness audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    fetch = fetcher or _default_fetcher
    roles = ["elsevier", "crossref", "datacite"]
    if reverse:
        roles.reverse()
    urls = {role: source[f"{role}_url"] for role in roles}
    fetched: dict[str, FetchResult] = {}
    for role in roles:
        fetched[role] = fetch(urls[role])
    if any(status != 200 for status, _ in fetched.values()):
        raise FilmRestorationMultilabSourceError(
            "an official metadata source returned non-200"
        )

    selected = {
        "elsevier": _selected_elsevier(
            _load_json(fetched["elsevier"][1], "Elsevier")
        ),
        "crossref": _selected_crossref(
            _load_json(fetched["crossref"][1], "Crossref")
        ),
        "datacite": _selected_datacite(
            _load_json(fetched["datacite"][1], "DataCite")
        ),
    }
    expected = source["expected"]
    elsevier = selected["elsevier"]
    crossref = selected["crossref"]
    datacite = selected["datacite"]
    operation_counts = dict(config["operation_limits"])
    forbidden_operations = (
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
    )
    license_rows = crossref["licenses"]
    audit_gates = {
        "elsevier_bibliographic_identity_exact": all(
            (
                elsevier["doi"] == source["doi"],
                elsevier["title"] == expected["title"],
                elsevier["journal"] == expected["journal"],
                elsevier["cover_date"] == expected["cover_date"],
                elsevier["pii_compact"] == expected["pii_compact"],
                elsevier["pii_formatted"] == expected["pii_formatted"],
            )
        ),
        "elsevier_closed_access_without_user_license": (
            elsevier["openaccess"] == expected["openaccess"]
            and elsevier["openaccess_user_license"] is None
        ),
        "crossref_identity_authors_and_relation_exact": all(
            (
                crossref["doi"] == source["doi"],
                crossref["title"] == expected["title"],
                crossref["authors"] == expected["authors"],
                crossref["relation"] == expected["crossref_relation"],
            )
        ),
        "crossref_licenses_are_policy_not_open_observation_data": bool(
            license_rows
        )
        and all(
            row["content_version"] in {"tdm", "stm-asf"}
            for row in license_rows
        ),
        "datacite_exact_title_has_no_dataset_record": (
            datacite["total"] == expected["datacite_exact_title_total"]
            and datacite["record_count"] == 0
        ),
        "public_article_multilab_topology_prospectively_bound": all(
            (
                source["public_article_topology"]["laboratory_count"] == 6,
                source["public_article_topology"]["scene_count"] == 3,
                source["public_article_topology"]["frames_per_scene"] == 24,
                source["public_article_topology"]["common_source"]
                == "original_camera_negative",
                source["public_article_topology"]["participant_reference"]
                == "positive_print",
                source["public_article_topology"]["independent_reference"]
                == "director_of_photography_supervised_restoration",
                source["public_article_topology"][
                    "independent_reference_withheld_from_participants"
                ]
                is True,
            )
        ),
        "film_stock_identity_explicitly_unknown": (
            source["public_article_topology"]["film_stock_identity"]
            == "unknown"
        ),
        "zero_media_pixel_and_model_reads": all(
            int(operation_counts[key]) == 0 for key in forbidden_operations
        ),
    }
    evidence = config["data_evidence"]
    minimums = config["admission_minimums"]
    admission_gates = {
        key: int(evidence[key]) >= int(minimums[key]) for key in sorted(minimums)
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    source_identity = {
        "doi": source["doi"],
        "official_article_url": source["official_article_url"],
        "institutional_record_url": source["institutional_record_url"],
        "selected_metadata": selected,
        "selected_metadata_sha256": _json_sha256(selected),
        "public_article_topology": source["public_article_topology"],
        "urls": {role: urls[role] for role in sorted(urls)},
    }
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a4a-film-restoration-multilab-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source": source_identity,
        "source_identity_sha256": _json_sha256(source_identity),
        "official_statuses": {
            role: fetched[role][0] for role in sorted(fetched)
        },
        "data_evidence": evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report
