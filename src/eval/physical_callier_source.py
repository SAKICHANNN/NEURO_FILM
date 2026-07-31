"""Bounded primary-source audit for the U6.P6M Callier mechanism."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCHEMA = "neuro_film.u6_p6m_callier_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6m_callier_source_report.v1"
_ALLOWED_HOST = "library.imaging.org"


class CallierSourceError(RuntimeError):
    """Raised when the frozen Callier source or audit contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CallierSourceError("source destination must be a bounded relative path")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    acquisition = payload.get("acquisition", {})
    gate = payload.get("source_gate", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("expected_bytes") != 6_128_203
        or source.get("expected_pages") != 4
        or acquisition.get("maximum_total_bytes") != 6_128_203
        or source.get("expected_bytes") != acquisition.get("maximum_total_bytes")
        or not acquisition.get("exact_url_only")
        or acquisition.get("redirect_host_allowlist") != [_ALLOWED_HOST]
        or not acquisition.get("no_other_payloads")
        or not gate.get("two_independent_audits")
        or not gate.get("no_numeric_curve_digitization")
        or not gate.get("no_parameter_fit")
        or set(payload.get("required_observations", {}))
        != {
            "callier_definition",
            "transparent_anchor",
            "material_control",
            "density_support",
            "spectral_support",
            "peak_region",
            "component_equation",
            "residual_warning",
        }
    ):
        raise CallierSourceError("P6M frozen contract drift")
    parsed = urlparse(source.get("url", ""))
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise CallierSourceError("P6M source URL is outside the frozen host")
    _relative_path(acquisition.get("destination", ""))
    return payload


def acquire_source(config: dict[str, Any], root: Path) -> Path:
    target = root / _relative_path(config["acquisition"]["destination"])
    expected = int(config["source"]["expected_bytes"])
    if target.exists():
        if target.is_file() and target.stat().st_size == expected:
            return target
        raise CallierSourceError("existing P6M source does not match frozen size")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = Request(config["source"]["url"], headers={"User-Agent": "neuro-film-p6m/1"})
    try:
        with urlopen(request, timeout=60) as response:
            final = urlparse(response.geturl())
            content_type = response.headers.get_content_type()
            if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
                raise CallierSourceError("P6M download redirected off frozen host")
            if content_type != config["source"]["expected_content_type"]:
                raise CallierSourceError("P6M source content type drift")
            written = 0
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected:
                        raise CallierSourceError("P6M download exceeds frozen size")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if written != expected:
            raise CallierSourceError("P6M download ended before frozen size")
        os.replace(partial, target)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return target


def _tool_identity(executable: str) -> dict[str, str]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise CallierSourceError(f"required PDF tool is unavailable: {executable}")
    version = subprocess.run(
        [resolved, "-v"], capture_output=True, text=True, check=False
    )
    banner = (version.stdout + "\n" + version.stderr).strip().splitlines()
    if version.returncode != 0 or not banner:
        raise CallierSourceError(f"cannot identify PDF tool: {executable}")
    return {
        "name": executable,
        "version": banner[0].strip(),
        "sha256": _hash_file(Path(resolved)),
    }


def _extract_pdf(pdf: Path) -> tuple[str, int, list[dict[str, str]]]:
    pdftotext = _tool_identity("pdftotext.exe")
    pdfinfo = _tool_identity("pdfinfo.exe")
    text_run = subprocess.run(
        [
            shutil.which("pdftotext.exe"),
            "-layout",
            "-enc",
            "UTF-8",
            str(pdf),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    try:
        text = text_run.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CallierSourceError("P6M extracted text is not UTF-8") from exc
    info_run = subprocess.run(
        [shutil.which("pdfinfo.exe"), str(pdf)],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", info_run.stdout)
    if match is None:
        raise CallierSourceError("P6M PDF page count is unavailable")
    return text, int(match.group(1)), [pdftotext, pdfinfo]


def evaluate_observations(text: str) -> dict[str, bool]:
    normalized = " ".join(text.split()).lower()
    return {
        "callier_definition": all(
            token in normalized
            for token in ("q = od", "as a function of od", "directed", "diffuse")
        ),
        "transparent_anchor": all(
            token in normalized
            for token in ("completely transparent point", "equal to 1")
        ),
        "material_control": all(
            token in normalized
            for token in (
                "chromogenic monopack",
                "dyes that scarcely scatter light",
                "q is close to 1",
            )
        ),
        "density_support": all(
            token in normalized for token in ("2.2", "range [0, 2.4]")
        ),
        "spectral_support": all(
            token in normalized
            for token in (
                "31 images",
                "between 400 and 700 nm",
                "spectral step of 10 nm",
                "stronger for short wavelengths",
            )
        ),
        "peak_region": all(
            token in normalized
            for token in (
                "maximum callier effect is between 1.3",
                "and 1.5 od",
            )
        ),
        "component_equation": all(
            token in normalized for token in ("oddye", "× q] + oddye")
        ),
        "residual_warning": all(
            token in normalized
            for token in ("residual discrepancies", "silver particles")
        ),
    }


def audit_source(config: dict[str, Any], root: Path) -> dict[str, Any]:
    pdf = root / _relative_path(config["acquisition"]["destination"])
    expected = int(config["source"]["expected_bytes"])
    if not pdf.is_file() or pdf.stat().st_size != expected:
        raise CallierSourceError("P6M local source integrity mismatch")
    text, pages, tools = _extract_pdf(pdf)
    observations = evaluate_observations(text)
    source_pass = (
        pages == int(config["source"]["expected_pages"])
        and all(observations.values())
    )
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": _hash_file(pdf),
        "source_bytes": pdf.stat().st_size,
        "page_count": pages,
        "extracted_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pdf_tools": tools,
        "observations": observations,
        "source_pass": source_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_generic_bounded_callier_operator"
            if source_pass
            else "close_callier_source"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CallierSourceError",
    "acquire_source",
    "audit_source",
    "evaluate_observations",
    "load_contract",
]
