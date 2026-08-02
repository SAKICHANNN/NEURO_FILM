"""U6.P4BE official Kodak Research aperture-table source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

SCHEMA = "neuro_film.u6_p4be_kodak_aperture_table_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4be_kodak_aperture_table_source_report.v1"


class KodakApertureTableSourceError(RuntimeError):
    """Raised when frozen P4BE source or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    acquisition = payload.get("acquisition", {})
    gate = payload.get("source_gate", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("search_context", {}).get("newer_primary_measured_open_table_found")
        is not False
        or source.get("doi") != "10.1364/JOSA.49.000925"
        or source.get("expected_host") != "opg.optica.org"
        or source.get("expected_content_type_prefix") != "text/html"
        or source.get("maximum_bytes") != 2_000_000
        or acquisition.get("download_official_html_only") is not True
        or acquisition.get("paid_pdf_or_subscription_access_allowed") is not False
        or acquisition.get("remote_assets_allowed") is not False
        or gate.get("required_title_exact") is not True
        or gate.get("required_doi_exact") is not True
        or gate.get("required_table_count") != 4
        or gate.get("minimum_density_group_count") != 23
        or gate.get("required_apertures_micrometres")
        != [7.25, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
        or gate.get("minimum_numeric_measurement_count") != 161
        or len(gate.get("required_film_names", [])) != 4
        or gate.get("no_figure_digitization") is not True
        or gate.get("no_parameter_fit") is not True
        or gate.get("two_independent_audits") is not True
    ):
        raise KodakApertureTableSourceError("P4BE frozen contract drift")
    return payload


def _source_path(contract: dict[str, Any], root: Path) -> Path:
    path = Path(contract["acquisition"]["destination"])
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise KodakApertureTableSourceError("P4BE source path must be relative")
    return root / path


def _meta(soup: BeautifulSoup, name: str) -> str:
    node = soup.find("meta", attrs={"name": name})
    if node is None or not node.get("content"):
        raise KodakApertureTableSourceError(f"missing publisher metadata: {name}")
    return str(node["content"]).strip()


def _parse_tables(
    soup: BeautifulSoup, required_films: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    found_films: list[str] = []
    for index, required_film in enumerate(required_films, start=1):
        label = soup.find("h4", id=f"tablet{index}Label")
        if label is None:
            raise KodakApertureTableSourceError(f"missing official table {index}")
        caption = label.find_next("h4")
        table = label.find_next("table")
        if caption is None or required_film not in caption.get_text(" ", strip=True):
            raise KodakApertureTableSourceError(f"film caption mismatch: {index}")
        if table is None:
            raise KodakApertureTableSourceError(f"missing table body: {index}")
        found_films.append(required_film)
        current: dict[str, Any] | None = None
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 4:
                continue
            values = [cell.get_text(" ", strip=True) for cell in cells]
            if values[0]:
                current = {
                    "table": index,
                    "film": required_film,
                    "density": float(values[0]),
                    "measurements": [],
                }
                groups.append(current)
            if current is None:
                raise KodakApertureTableSourceError("table row lacks density group")
            current["measurements"].append(
                {
                    "aperture_diameter_micrometres": float(values[1]),
                    "density_standard_deviation": float(values[2]),
                    "selwyn_granularity": float(values[3]),
                }
            )
    return groups, found_films


def audit_source(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _source_path(contract, root)
    if not path.is_file():
        raise KodakApertureTableSourceError("P4BE official HTML is unavailable")
    raw = path.read_bytes()
    if len(raw) > int(contract["source"]["maximum_bytes"]):
        raise KodakApertureTableSourceError("P4BE official HTML exceeds byte cap")
    soup = BeautifulSoup(raw, "html.parser")
    publisher_title = _meta(soup, "citation_title").rstrip("*").strip()
    publisher_doi = _meta(soup, "citation_doi")
    groups, found_films = _parse_tables(
        soup, list(contract["source_gate"]["required_film_names"])
    )
    required_apertures = {
        float(value)
        for value in contract["source_gate"]["required_apertures_micrometres"]
    }
    all_apertures_exact = all(
        {
            float(measurement["aperture_diameter_micrometres"])
            for measurement in group["measurements"]
        }
        == required_apertures
        for group in groups
    )
    measurement_count = sum(len(group["measurements"]) for group in groups)
    gate_results = {
        "title_exact": publisher_title == contract["source"]["title"],
        "doi_exact": publisher_doi.lower() == contract["source"]["doi"].lower(),
        "table_count": len(found_films)
        == int(contract["source_gate"]["required_table_count"]),
        "density_group_count": len(groups)
        >= int(contract["source_gate"]["minimum_density_group_count"]),
        "film_names_exact": found_films
        == list(contract["source_gate"]["required_film_names"]),
        "apertures_exact_per_group": all_apertures_exact,
        "measurement_count": measurement_count
        >= int(contract["source_gate"]["minimum_numeric_measurement_count"]),
        "no_figure_digitization": True,
        "no_parameter_fit": True,
        "rights_internal_only": "internal research"
        in contract["source"]["rights_scope"],
    }
    source_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "publisher_title": publisher_title,
        "publisher_doi": publisher_doi,
        "table_count": len(found_films),
        "density_group_count": len(groups),
        "numeric_measurement_count": measurement_count,
        "film_names": found_films,
        "gate_results": gate_results,
        "source_pass": source_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "density_groups": groups,
        "decision": (
            "open_fixed_kodak_aperture_law_confirmation"
            if source_pass
            else "close_source_without_pdf_or_table_repair"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "KodakApertureTableSourceError",
    "audit_source",
    "load_contract",
    "write_report",
]
