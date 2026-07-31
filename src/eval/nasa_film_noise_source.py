"""Bounded acquisition and mechanism audit for NASA TN D-4501."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen


SCHEMA = "neuro_film.u6_p4ak_nasa_film_noise_mechanism_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ak_nasa_film_noise_mechanism_source_report.v1"


class NasaFilmNoiseSourceError(RuntimeError):
    """Raised for contract, rights, identity or bounded-acquisition failure."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    source, acquisition, gates = value.get("source", {}), value.get("acquisition", {}), value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or source.get("record_id") != "19680012473"
        or source.get("report_number") != "NASA-TN-D-4501"
        or source.get("expected_distribution") != "PUBLIC"
        or source.get("expected_rights_determination") != "GOV_PUBLIC_USE_PERMITTED"
        or source.get("maximum_pdf_bytes") != 25000000
        or source.get("maximum_text_bytes") != 5000000
        or acquisition.get("downloads") != 2
        or acquisition.get("ocr_allowed")
        or acquisition.get("figure_digitization_allowed")
        or gates.get("minimum_text_bytes") != 20000
        or not gates.get("repeat_download_sha256_exact")
        or not gates.get("no_ocr")
        or not gates.get("no_figure_digitization")
    ):
        raise NasaFilmNoiseSourceError("P4AK frozen contract drift")
    return value


def _fetch(url: str, *, maximum_bytes: int, timeout: int, user_agent: str) -> bytes:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > maximum_bytes:
            raise NasaFilmNoiseSourceError("P4AK response exceeds byte bound")
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise NasaFilmNoiseSourceError("P4AK response exceeds byte bound")
    return data


def analyze_machine_text(text: str, anchors: list[str]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).casefold()
    anchor_results = {anchor: anchor.casefold() in normalized for anchor in anchors}
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    numeric_rows = 0
    for index, line in enumerate(lines):
        lower = line.casefold()
        if "exposure" not in lower or not ("granular" in lower or "rms noise" in lower):
            continue
        following = lines[index + 1 : index + 16]
        numeric_rows = max(
            numeric_rows,
            sum(len(re.findall(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", row)) >= 2 for row in following),
        )
    observations = {
        "exposure_dependent_film_response": "film response now becomes a function of the background exposure" in normalized,
        "separate_exposure_dependent_density_noise": "exposure-dependent film granularity noise" in normalized and "standard or rms deviation about the mean density" in normalized,
        "density_to_transmission_noise_relation": "film transmission noise" in normalized and "2.3" in normalized,
        "scanning_aperture_is_separate": "film scanning aperture area" in normalized and "communication link" in normalized,
        "machine_readable_numeric_granularity_table_rows": numeric_rows,
        "reusable_numeric_granularity_table": numeric_rows >= 5,
    }
    return {"anchor_results": anchor_results, "observations": observations}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def acquire_and_evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / config["parent"]["p4aj_decision_path"]
    if not parent_path.is_file() or _sha256(parent_path.read_bytes()) != config["parent"]["p4aj_decision_sha256"]:
        raise NasaFilmNoiseSourceError("P4AK parent identity mismatch")
    source, acquisition = config["source"], config["acquisition"]
    timeout, user_agent = int(acquisition["timeout_seconds"]), str(acquisition["user_agent"])
    metadata_runs, pdf_runs, text_runs = [], [], []
    for _ in range(int(acquisition["downloads"])):
        metadata_runs.append(_fetch(source["metadata_url"], maximum_bytes=1000000, timeout=timeout, user_agent=user_agent))
        pdf_runs.append(_fetch(source["pdf_url"], maximum_bytes=int(source["maximum_pdf_bytes"]), timeout=timeout, user_agent=user_agent))
        text_runs.append(_fetch(source["text_url"], maximum_bytes=int(source["maximum_text_bytes"]), timeout=timeout, user_agent=user_agent))
    metadata = json.loads(metadata_runs[0].decode("utf-8"))
    metadata_identity = (
        str(metadata.get("id")) == source["record_id"]
        and metadata.get("title") == source["title"]
        and source["report_number"] in metadata.get("otherReportNumbers", [])
    )
    rights = (
        metadata.get("distribution") == source["expected_distribution"]
        and metadata.get("copyright", {}).get("determinationType") == source["expected_rights_determination"]
        and metadata.get("exportControl", {}).get("isExportControl") == "NO"
    )
    repeat_exact = len({_sha256(data) for data in pdf_runs}) == 1 and len({_sha256(data) for data in text_runs}) == 1
    text = text_runs[0].decode("utf-8", errors="strict")
    analysis = analyze_machine_text(text, list(config["observations"]["required_text_anchors"]))
    destination = root / acquisition["destination"]
    if not destination.resolve().is_relative_to((root / "data").resolve()):
        raise NasaFilmNoiseSourceError("P4AK destination escaped data root")
    _atomic_write(destination / "19680012473.pdf", pdf_runs[0])
    _atomic_write(destination / "19680012473.txt", text_runs[0])
    _atomic_write(destination / "metadata.json", _canonical(metadata))
    checks = {
        "metadata_identity": metadata_identity,
        "public_rights": rights,
        "bounded_payloads": len(pdf_runs[0]) <= source["maximum_pdf_bytes"] and len(text_runs[0]) <= source["maximum_text_bytes"],
        "pdf_magic": pdf_runs[0].startswith(config["gates"]["pdf_magic"].encode()),
        "minimum_text": len(text_runs[0]) >= config["gates"]["minimum_text_bytes"],
        "anchors": all(analysis["anchor_results"].values()),
        "repeat_exact": repeat_exact,
        "no_ocr": True,
        "no_figure_digitization": True,
        "parent_identity": True,
    }
    source_pass = all(checks.values())
    numeric = analysis["observations"]["reusable_numeric_granularity_table"]
    branch = "source_pass_numeric_table" if source_pass and numeric else ("source_pass_no_numeric_table" if source_pass else "source_fail")
    stable = {
        "experiment_id": config["experiment_id"],
        "source_identity": {
            "record_id": source["record_id"],
            "metadata_sha256": _sha256(_canonical(metadata)),
            "pdf_sha256": _sha256(pdf_runs[0]),
            "pdf_bytes": len(pdf_runs[0]),
            "text_sha256": _sha256(text_runs[0]),
            "text_bytes": len(text_runs[0]),
        },
        "analysis": analysis,
        "checks": checks,
        "source_pass": source_pass,
        "branch": branch,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"][branch],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["NasaFilmNoiseSourceError", "REPORT_SCHEMA", "SCHEMA", "acquire_and_evaluate", "analyze_machine_text", "load_contract"]
