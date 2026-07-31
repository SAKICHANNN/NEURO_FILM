"""Bounded source audit for the U6.P4AA NASA grain/MTF report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.eval.physical_callier_source import extract_pdf, hash_file


SCHEMA = "neuro_film.u6_p4aa_nasa_density_grain_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4aa_nasa_density_grain_source_report.v1"
_ALLOWED_HOST = "ntrs.nasa.gov"


class DensityGrainSourceError(RuntimeError):
    """Raised when the P4AA source contract or local payload drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DensityGrainSourceError("P4AA destination must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    acquisition = payload.get("acquisition", {})
    gate = payload.get("source_gate", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("record_id") != "19730022682"
        or source.get("expected_bytes") != 2_179_830
        or source.get("distribution") != "PUBLIC"
        or acquisition.get("maximum_total_bytes") != 2_179_830
        or acquisition.get("redirect_host_allowlist") != [_ALLOWED_HOST]
        or not acquisition.get("exact_pdf_only")
        or not acquisition.get("no_other_payloads")
        or gate.get("pdf_page_count_range") != [20, 200]
        or gate.get("minimum_machine_extractable_numeric_density_granularity_rows")
        != 3
        or gate.get("required_mechanism_terms")
        != [
            "root-mean-square",
            "graininess",
            "power spectral density",
            "modulation transfer function",
            "aperture",
        ]
        or not gate.get("no_figure_digitization")
        or not gate.get("no_parameter_fit")
    ):
        raise DensityGrainSourceError("P4AA frozen contract drift")
    parsed = urlparse(source.get("url", ""))
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise DensityGrainSourceError("P4AA source URL outside frozen host")
    _relative_path(acquisition.get("destination", ""))
    return payload


def acquire_source(config: dict[str, Any], root: Path) -> Path:
    target = root / _relative_path(config["acquisition"]["destination"])
    expected = int(config["source"]["expected_bytes"])
    if target.exists():
        if target.is_file() and target.stat().st_size == expected:
            return target
        raise DensityGrainSourceError("existing P4AA source integrity mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = Request(config["source"]["url"], headers={"User-Agent": "neuro-film-p4aa/1"})
    try:
        with urlopen(request, timeout=60) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
                raise DensityGrainSourceError("P4AA redirect left frozen host")
            if not response.headers.get_content_type().startswith("application/pdf"):
                raise DensityGrainSourceError("P4AA content type drift")
            written = 0
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected:
                        raise DensityGrainSourceError("P4AA download exceeds cap")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if written != expected:
            raise DensityGrainSourceError("P4AA download size mismatch")
        os.replace(partial, target)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return target


def _term_presence(text: str, terms: list[str]) -> dict[str, bool]:
    normalized = " ".join(text.lower().split())
    return {term: term in normalized for term in terms}


def _numeric_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_label_end = 0
    for table_number in (4, 5, 6):
        label = re.search(rf"Table\s+{table_number}\.\s+RMS Granularity", text)
        if label is None:
            continue
        page_start = text.rfind("\f", 0, label.start()) + 1
        block_start = max(page_start, previous_label_end)
        block = text[block_start : label.start()]
        for line in block.splitlines():
            match = re.match(
                r"^\s*(\d+)\.\s+(\.?\d+(?:\.\d+)?)\s+(.+?)\s*$", line
            )
            if match is None:
                continue
            value_matches = list(re.finditer(r"\d+(?:\.\d+)?", match.group(3)))
            values = [float(item.group()) for item in value_matches]
            if len(values) < 2:
                continue
            tail_start = match.start(3)
            apertures: dict[str, float] = {}
            for item, value in zip(value_matches, values, strict=True):
                column = tail_start + item.start()
                aperture = "13" if column < 32 else "27" if column < 42 else "57"
                if aperture in apertures:
                    raise DensityGrainSourceError(
                        "ambiguous P4AA RMS granularity table columns"
                    )
                apertures[aperture] = value
            density_text = match.group(2)
            rows.append(
                {
                    "table": table_number,
                    "row": int(match.group(1)),
                    "density": float(
                        f"0{density_text}" if density_text.startswith(".") else density_text
                    ),
                    "rms_granularity_by_aperture_um": apertures,
                }
            )
        previous_label_end = label.end()
    return rows


def audit_source(config: dict[str, Any], root: Path) -> dict[str, Any]:
    pdf = root / _relative_path(config["acquisition"]["destination"])
    expected = int(config["source"]["expected_bytes"])
    if not pdf.is_file() or pdf.stat().st_size != expected:
        raise DensityGrainSourceError("P4AA local source integrity mismatch")
    text, pages, tools = extract_pdf(pdf)
    terms = _term_presence(text, config["source_gate"]["required_mechanism_terms"])
    numeric_rows = _numeric_rows(text)
    low, high = config["source_gate"]["pdf_page_count_range"]
    page_gate = low <= pages <= high
    terms_gate = all(terms.values())
    numeric_gate = len(numeric_rows) >= int(
        config["source_gate"][
            "minimum_machine_extractable_numeric_density_granularity_rows"
        ]
    )
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": hash_file(pdf),
        "source_bytes": pdf.stat().st_size,
        "page_count": pages,
        "extracted_text_bytes": len(text.encode("utf-8")),
        "extracted_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pdf_tools": tools,
        "mechanism_terms_present": terms,
        "numeric_density_granularity_row_count": len(numeric_rows),
        "numeric_density_granularity_rows": numeric_rows,
        "gate_results": {
            "page_count": page_gate,
            "mechanism_terms": terms_gate,
            "numeric_table_rows": numeric_gate,
            "no_figure_digitization": True,
            "no_parameter_fit": True,
        },
    }
    passed = all(stable["gate_results"].values())
    stable["source_pass"] = passed
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_fixed_density_granularity_compatibility"
            if passed
            else "close_source_without_figure_digitization"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "DensityGrainSourceError",
    "acquire_source",
    "audit_source",
    "load_contract",
]
