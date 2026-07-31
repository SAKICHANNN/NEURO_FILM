"""Reproducible four-sample Callier measurement audit for U6.P6O."""

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


SCHEMA = "neuro_film.u6_p6o_callier_measurement_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6o_callier_measurement_source_report.v1"
_ALLOWED_HOST = "edoc.unibas.ch"
_FROZEN_ROWS = [
    {
        "sample_id": "silverHD",
        "material": "silver-based",
        "density_level": "high",
        "directed_density": 1.53,
        "diffuse_density": 0.87,
        "reported_q": 1.76,
    },
    {
        "sample_id": "silverLD",
        "material": "silver-based",
        "density_level": "low",
        "directed_density": 0.29,
        "diffuse_density": 0.2,
        "reported_q": 1.4,
    },
    {
        "sample_id": "dyeHD",
        "material": "dye-based",
        "density_level": "high",
        "directed_density": 1.59,
        "diffuse_density": 1.55,
        "reported_q": 1.02,
    },
    {
        "sample_id": "dyeLD",
        "material": "dye-based",
        "density_level": "low",
        "directed_density": 0.75,
        "diffuse_density": 0.69,
        "reported_q": 1.09,
    },
]


class CallierMeasurementSourceError(RuntimeError):
    """Raised when the P6O measurement source or contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CallierMeasurementSourceError("P6O destination must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    acquisition = payload.get("acquisition", {})
    gate = payload.get("source_gate", {})
    rows = payload.get("frozen_measurement_table", [])
    if (
        payload.get("schema") != SCHEMA
        or source.get("expected_bytes") != 35_683_532
        or source.get("expected_md5") != "ee6433a8421e5070c89b32c0dc226fb8"
        or acquisition.get("maximum_total_bytes") != 35_683_532
        or acquisition.get("redirect_host_allowlist") != [_ALLOWED_HOST]
        or not acquisition.get("exact_bitstream_only")
        or not acquisition.get("no_other_payloads")
        or rows != _FROZEN_ROWS
        or gate.get("pdf_page_count_range") != [130, 150]
        or not gate.get("raw_density_ratio_rounding_discrepancy_must_be_reported")
        or not gate.get("two_independent_audits")
        or not gate.get("no_figure_digitization")
        or not gate.get("no_parameter_fit")
    ):
        raise CallierMeasurementSourceError("P6O frozen contract drift")
    parsed = urlparse(source.get("url", ""))
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise CallierMeasurementSourceError("P6O source URL is outside frozen host")
    _relative_path(acquisition.get("destination", ""))
    return payload


def acquire_source(config: dict[str, Any], root: Path) -> Path:
    target = root / _relative_path(config["acquisition"]["destination"])
    expected_bytes = int(config["source"]["expected_bytes"])
    expected_md5 = config["source"]["expected_md5"]
    if target.exists():
        if (
            target.is_file()
            and target.stat().st_size == expected_bytes
            and hash_file(target, "md5") == expected_md5
        ):
            return target
        raise CallierMeasurementSourceError("existing P6O source integrity mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = Request(config["source"]["url"], headers={"User-Agent": "neuro-film-p6o/1"})
    try:
        with urlopen(request, timeout=60) as response:
            final = urlparse(response.geturl())
            content_type = response.headers.get_content_type()
            if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
                raise CallierMeasurementSourceError("P6O redirect left frozen host")
            if not content_type.startswith("application/pdf"):
                raise CallierMeasurementSourceError("P6O content type drift")
            written = 0
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected_bytes:
                        raise CallierMeasurementSourceError("P6O download exceeds cap")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if (
            written != expected_bytes
            or hash_file(partial, "md5") != expected_md5
        ):
            raise CallierMeasurementSourceError("P6O download identity mismatch")
        os.replace(partial, target)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return target


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _table_rows_present(text: str, rows: list[dict[str, Any]]) -> dict[str, bool]:
    normalized = _normalized(text)
    results = {}
    for row in rows:
        sample = re.escape(row["sample_id"])
        directed = re.escape(f"{row['directed_density']:.2f}")
        diffuse = re.escape(f"{row['diffuse_density']:.2f}")
        density_pattern = rf"{sample}\s+{directed}\s+{diffuse}"
        reported_q = re.escape(f"{row['reported_q']:.2f}")
        q_pattern = rf"{sample}\s+{reported_q}"
        results[row["sample_id"]] = bool(
            re.search(density_pattern, normalized)
            and re.search(q_pattern, normalized)
        )
    return results


def audit_source(config: dict[str, Any], root: Path) -> dict[str, Any]:
    pdf = root / _relative_path(config["acquisition"]["destination"])
    if (
        not pdf.is_file()
        or pdf.stat().st_size != int(config["source"]["expected_bytes"])
        or hash_file(pdf, "md5") != config["source"]["expected_md5"]
    ):
        raise CallierMeasurementSourceError("P6O local source integrity mismatch")
    text, pages, tools = extract_pdf(pdf)
    rows_present = _table_rows_present(text, config["frozen_measurement_table"])
    measurements = []
    for row in config["frozen_measurement_table"]:
        raw_ratio = row["directed_density"] / row["diffuse_density"]
        measurements.append(
            {
                **row,
                "raw_density_ratio": raw_ratio,
                "reported_minus_raw_ratio": row["reported_q"] - raw_ratio,
            }
        )
    low, high = config["source_gate"]["pdf_page_count_range"]
    source_pass = low <= pages <= high and all(rows_present.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": hash_file(pdf),
        "source_md5": hash_file(pdf, "md5"),
        "source_bytes": pdf.stat().st_size,
        "page_count": pages,
        "extracted_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pdf_tools": tools,
        "rows_present": rows_present,
        "measurements": measurements,
        "maximum_reported_vs_raw_ratio_difference": max(
            abs(row["reported_minus_raw_ratio"]) for row in measurements
        ),
        "source_pass": source_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_fixed_p6n_measurement_compatibility_audit"
            if source_pass
            else "close_callier_measurement_source"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CallierMeasurementSourceError",
    "acquire_source",
    "audit_source",
    "load_contract",
]
