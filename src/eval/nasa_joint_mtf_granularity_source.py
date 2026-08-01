"""Measured-table source audit for the U6.P4AT joint MTF/granularity leaf."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.eval.physical_callier_source import extract_pdf, hash_file

SCHEMA = "neuro_film.u6_p4at_nasa_joint_mtf_granularity_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4at_nasa_joint_mtf_granularity_source_report.v1"


class JointTableSourceError(RuntimeError):
    """Raised when the frozen source, upstream evidence, or table layout drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise JointTableSourceError("P4AT paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    upstream = payload.get("upstream_granularity_report", {})
    gate = payload.get("table_gate", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("record_id") != "19730022682"
        or source.get("expected_bytes") != 2_179_830
        or source.get("expected_pages") != 85
        or source.get("expected_sha256")
        != "8773bb57946c38573d2feb42a08ddf3b0d5001745bc46ac23d825d6cdad90814"
        or upstream.get("expected_sha256")
        != "98be0f689842b9ec92ecc6ea9b51c2fd73269345da5508dcc9e07dba477c9b42"
        or upstream.get("expected_numeric_rows") != 14
        or upstream.get("expected_tables") != [4, 5, 6]
        or gate.get("expected_mtf_blocks") != [f"25{letter}" for letter in "abcdefghi"]
        or gate.get("expected_frequencies_cycles_per_mm")
        != [0, 2, 4, 6, 8, 10, 12, 14, 16]
        or gate.get("expected_columns_per_block") != 3
        or gate.get("minimum_machine_readable_mtf_values") != 240
        or not gate.get("record_printed_series_count_even_if_prose_disagrees")
        or not gate.get("same_package_statement_required")
        or not gate.get("same_process_statement_required")
        or not gate.get("composite_system_nuisance_statement_required")
        or not gate.get("psd_is_figure_only_statement_required")
        or not gate.get("two_independent_audits")
        or not gate.get("no_figure_digitization")
        or not gate.get("no_parameter_fit")
    ):
        raise JointTableSourceError("P4AT frozen contract drift")
    _relative_path(source.get("local_path", ""))
    _relative_path(upstream.get("path", ""))
    return payload


def _normalized_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)


def _normalize_exposure(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _parse_mtf_tables(
    text: str, expected_frequencies: list[int]
) -> list[dict[str, Any]]:
    labels = list(re.finditer(r"25([a-i])\.\s+Exposure\s+([^\r\n]+)", text))
    blocks: list[dict[str, Any]] = []
    for label in labels:
        start = label.start()
        heading = text.rfind("Frequency", max(0, start - 2500), start)
        if heading < 0:
            raise JointTableSourceError("P4AT MTF heading is unavailable")
        rows: list[dict[str, Any]] = []
        for line in text[heading:start].splitlines():
            match = re.match(
                r"^\s*(0|2|4|6|8|10|12|14|16)\.?\s+"
                r"([01]\.\d+)\s+([01]\.\d+)\s+([01]\.\d+)\s*$",
                line,
            )
            if match is None:
                continue
            rows.append(
                {
                    "frequency_cycles_per_mm": int(match.group(1)),
                    "mtf_by_edge": {
                        "3": float(match.group(2)),
                        "5": float(match.group(3)),
                        "7": float(match.group(4)),
                    },
                }
            )
        if [row["frequency_cycles_per_mm"] for row in rows] != expected_frequencies:
            raise JointTableSourceError("P4AT MTF frequency rows drifted")
        blocks.append(
            {
                "table_block": f"25{label.group(1)}",
                "exposure_id": _normalize_exposure(label.group(2)),
                "rows": rows,
            }
        )
    return blocks


def _load_granularity_report(config: dict[str, Any], root: Path) -> dict[str, Any]:
    locked = config["upstream_granularity_report"]
    path = root / _relative_path(locked["path"])
    if not path.is_file() or hash_file(path) != locked["expected_sha256"]:
        raise JointTableSourceError("P4AT upstream granularity report drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("numeric_density_granularity_rows", [])
    if (
        report.get("schema") != "neuro_film.u6_p4aa_nasa_density_grain_source_report.v1"
        or report.get("source_sha256") != config["source"]["expected_sha256"]
        or report.get("stable_evidence_id") != locked["expected_stable_evidence_id"]
        or len(rows) != locked["expected_numeric_rows"]
        or sorted({row.get("table") for row in rows}) != locked["expected_tables"]
    ):
        raise JointTableSourceError("P4AT upstream granularity facts drift")
    return report


def audit_source(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = config["source"]
    pdf = root / _relative_path(source["local_path"])
    if (
        not pdf.is_file()
        or pdf.stat().st_size != source["expected_bytes"]
        or hash_file(pdf) != source["expected_sha256"]
    ):
        raise JointTableSourceError("P4AT local source integrity mismatch")
    granularity = _load_granularity_report(config, root)
    text, pages, tools = extract_pdf(pdf)
    expected_frequencies = config["table_gate"]["expected_frequencies_cycles_per_mm"]
    mtf_blocks = _parse_mtf_tables(text, expected_frequencies)
    normalized = _normalized_text(text)
    statements = {
        "same_film_packages": (
            "from the same packages as the film used in the quantum mottle and mtf portions"
            in normalized
        ),
        "same_process_conditions": (
            "processed at msfc immediately following exposure under the same conditions as those employed for the quantum mottle radiographs"
            in normalized
        ),
        "composite_system_mtf": (
            "mtf for the entire x-ray scanning system (including x-ray source, film, processing and scanner)"
            in normalized
        ),
        "psd_results_are_figures": "the results are presented in figures 15-22"
        in normalized,
        "prose_claims_twenty_six_edges": "mtf's for twenty-six edges" in normalized,
    }
    printed_series_count = sum(
        len(block["rows"][0]["mtf_by_edge"]) for block in mtf_blocks
    )
    mtf_value_count = sum(
        len(row["mtf_by_edge"]) for block in mtf_blocks for row in block["rows"]
    )
    expected_blocks = config["table_gate"]["expected_mtf_blocks"]
    gate_results = {
        "page_count": pages == source["expected_pages"],
        "granularity_rows": len(granularity["numeric_density_granularity_rows"])
        == config["upstream_granularity_report"]["expected_numeric_rows"],
        "mtf_blocks": [block["table_block"] for block in mtf_blocks] == expected_blocks,
        "mtf_numeric_values": mtf_value_count
        >= config["table_gate"]["minimum_machine_readable_mtf_values"],
        "required_statements": all(statements.values()),
        "no_figure_digitization": True,
        "no_parameter_fit": True,
    }
    source_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": hash_file(pdf),
        "source_bytes": pdf.stat().st_size,
        "page_count": pages,
        "extracted_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pdf_tools": tools,
        "upstream_granularity_report_sha256": config["upstream_granularity_report"][
            "expected_sha256"
        ],
        "upstream_granularity_stable_evidence_id": granularity["stable_evidence_id"],
        "numeric_density_granularity_rows": granularity[
            "numeric_density_granularity_rows"
        ],
        "mtf_blocks": mtf_blocks,
        "mtf_numeric_value_count": mtf_value_count,
        "printed_mtf_series_count": printed_series_count,
        "prose_mtf_series_count": 26,
        "printed_prose_series_count_contradiction": printed_series_count != 26,
        "factual_statements": statements,
        "gate_results": gate_results,
        "source_pass": source_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_nuisance_aware_measured_compatibility"
            if source_pass
            else "close_source_without_figure_digitization"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "JointTableSourceError",
    "audit_source",
    "load_contract",
]
