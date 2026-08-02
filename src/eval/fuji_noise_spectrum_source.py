"""U6.P4BK exact source audit for Ooue's 1960 Fuji grain spectra."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pypdf import PdfReader

SCHEMA = "neuro_film.u6_p4bk_fuji_noise_spectrum_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bk_fuji_noise_spectrum_source_report.v1"

_EXPECTED_ROWS = (
    (
        "fuji_neopan_sss_pandol_20c_10min",
        0.41,
        (140.0, 126.0),
        (57.2, 113.0),
    ),
    (
        "fuji_neopan_sss_pandol_20c_10min",
        1.72,
        (308.0, 248.0, 7.6),
        (49.5, 119.0, 360.0),
    ),
    (
        "fuji_medical_xray_rendol_20c_5min",
        0.58,
        (355.0, 270.0),
        (32.8, 112.0),
    ),
    (
        "fuji_medical_xray_rendol_20c_5min",
        1.03,
        (680.0, 70.0),
        (66.4, 150.0),
    ),
    (
        "fuji_medical_xray_rendol_20c_5min",
        1.23,
        (625.0, 560.0),
        (41.0, 99.0),
    ),
)

_EXPECTED_MODELS = {
    "one_dimensional_model": "f(u)=sum_i k_i*exp(-u^2/(2*p_i^2))",
    "aperture_convolved_two_dimensional_model": (
        "Q(u,v)=1/sqrt(2*pi)*sum_i(k_i/p_i)*"
        "exp(-(u^2+v^2)/(2*p_i^2))"
    ),
    "circular_aperture_mtf": (
        "G(s)=2*J1(pi*s*a)/(pi*s*a), s=sqrt(u^2+v^2), a in millimetres"
    ),
    "intrinsic_two_dimensional_model": "F(u,v)=Q(u,v)/abs(G(u,v))^2",
}

_PAGE_MARKERS = {
    0: (
        "Graininess and Granularity of Photographic Materials",
        "I. Measurement of Power Spectrum",
        "Shingo OOUE",
        "Fuji Photo Film Co., Ltd.",
    ),
    4: (
        "Fig. 5 The one dimensional power spectra",
        "Fuji Neopan",
        "Fuji Pandol developer",
        "Table1",
    ),
    5: (
        "Fig. 6 The one dimensional power spectra",
        "Fuji Medical",
        "Fuji Rendol developer",
        "Fig. 7 The two dimensional power spectra",
        "Fig. 8 The two dimensional power spectra",
    ),
}


class FujiNoiseSpectrumSourceError(RuntimeError):
    """Raised when the frozen P4BK source or transcription drifts."""


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


def _source_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujiNoiseSpectrumSourceError("P4BK source path must be relative")
    return root / path


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("material_id"),
        row.get("diffuse_density"),
        tuple(row.get("k", ())),
        tuple(row.get("p_lines_per_mm", ())),
    )


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    measurement = payload.get("measurement", {})
    audit = payload.get("audit", {})
    rows = payload.get("table_1", [])
    if (
        payload.get("schema") != SCHEMA
        or source.get("doi") != "10.11470/oubutsu1932.29.169"
        or source.get("pdf_sha256")
        != "f03541cacdd0deacc0475f8a5ea3aa8bf2761e87d8183037e27a51283f4d92bf"
        or source.get("pdf_bytes") != 1171684
        or source.get("pdf_pages") != 7
        or source.get("redistribution_allowed") is not False
        or measurement.get("scanning_aperture_shape") != "circular"
        or measurement.get("scanning_aperture_diameter_micrometres") != 1.0
        or measurement.get("equation_numbers") != [5, 6, 7, 8, 9]
        or any(measurement.get(key) != value for key, value in _EXPECTED_MODELS.items())
        or tuple(_row_identity(row) for row in rows) != _EXPECTED_ROWS
        or audit.get("required_table_rows") != 5
        or audit.get("required_gaussian_components") != 11
        or audit.get("require_no_graph_digitization") is not True
        or audit.get("require_two_byte_identical_reports") is not True
    ):
        raise FujiNoiseSpectrumSourceError("P4BK frozen contract drift")
    return payload


def evaluate_fuji_noise_spectrum_source(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    source = contract["source"]
    pdf_path = _source_path(root, str(source["pdf_path"]))
    if not pdf_path.is_file():
        raise FujiNoiseSpectrumSourceError("P4BK official PDF is missing")
    pdf_sha256 = _hash_file(pdf_path)
    pdf_bytes = pdf_path.stat().st_size
    if pdf_sha256 != source["pdf_sha256"] or pdf_bytes != source["pdf_bytes"]:
        raise FujiNoiseSpectrumSourceError("P4BK official PDF identity mismatch")
    reader = PdfReader(str(pdf_path))
    marker_results: dict[str, bool] = {}
    for page_index, markers in _PAGE_MARKERS.items():
        text = reader.pages[page_index].extract_text() or ""
        for marker in markers:
            marker_results[f"page_{page_index + 1}:{marker}"] = marker in text

    rows = list(contract["table_1"])
    component_count = sum(len(row["k"]) for row in rows)
    positive_finite = all(
        math.isfinite(float(value)) and float(value) > 0.0
        for row in rows
        for key in ("k", "p_lines_per_mm")
        for value in row[key]
    )
    unique_rows = len(
        {(row["material_id"], float(row["diffuse_density"])) for row in rows}
    ) == len(rows)
    process_contexts = {
        (
            row["material_id"],
            row["developer"],
            float(row["temperature_celsius"]),
            float(row["development_minutes"]),
        )
        for row in rows
    }
    transcription_payload = {
        "scanning_aperture_diameter_micrometres": contract["measurement"][
            "scanning_aperture_diameter_micrometres"
        ],
        "models": {
            key: contract["measurement"][key] for key in sorted(_EXPECTED_MODELS)
        },
        "table_1": rows,
    }
    transcription_sha256 = hashlib.sha256(
        _canonical_json(transcription_payload)
    ).hexdigest()
    gates = {
        "pdf_identity": pdf_sha256 == source["pdf_sha256"],
        "pdf_size": pdf_bytes == source["pdf_bytes"],
        "page_count": len(reader.pages) == source["pdf_pages"],
        "english_source_markers": all(marker_results.values()),
        "table_row_count": len(rows) == contract["audit"]["required_table_rows"],
        "gaussian_component_count": component_count
        == contract["audit"]["required_gaussian_components"],
        "positive_finite_coefficients": positive_finite,
        "unique_material_density_rows": unique_rows,
        "two_material_process_contexts": len(process_contexts) == 2,
        "equation_semantics": all(
            contract["measurement"][key] == value
            for key, value in _EXPECTED_MODELS.items()
        ),
        "no_graph_digitization": contract["audit"]["require_no_graph_digitization"]
        is True,
        "internal_research_only": source["redistribution_allowed"] is False,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "source": {
            "title": source["title"],
            "author": source["author"],
            "doi": source["doi"],
            "official_article_url": source["official_article_url"],
            "official_pdf_url": source["official_pdf_url"],
            "pdf_path": source["pdf_path"],
            "pdf_sha256": pdf_sha256,
            "pdf_bytes": pdf_bytes,
            "pdf_pages": len(reader.pages),
        },
        "measurement": {
            "material_class": contract["measurement"]["material_class"],
            "spatial_frequency_unit": contract["measurement"][
                "spatial_frequency_unit"
            ],
            "scanning_aperture_shape": contract["measurement"][
                "scanning_aperture_shape"
            ],
            "scanning_aperture_diameter_micrometres": contract["measurement"][
                "scanning_aperture_diameter_micrometres"
            ],
            "equation_numbers": contract["measurement"]["equation_numbers"],
        },
        "table_1": rows,
        "table_row_count": len(rows),
        "gaussian_component_count": component_count,
        "material_process_context_count": len(process_contexts),
        "marker_results": marker_results,
        "transcription_sha256": transcription_sha256,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "open_historical_bw_measured_nps_compiler"
            if automatic_pass
            else "close_fuji_measured_nps_source"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    stable_payload.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
