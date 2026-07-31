"""Bounded source/readability audit for Takano's 1969 film-granularity paper."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.request import Request, urlopen


SCHEMA = "neuro_film.u6_p4al_jstage_film_granularity_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4al_jstage_film_granularity_source_report.v1"


class JstageFilmGranularitySourceError(RuntimeError):
    """Raised when the frozen P4AL source contract cannot be evaluated safely."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    source = value.get("source", {})
    acquisition = value.get("acquisition", {})
    machine_text = value.get("machine_text", {})
    gates = value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or source.get("doi") != "10.3169/itej1954.23.13"
        or source.get("maximum_article_bytes") != 2_000_000
        or source.get("maximum_pdf_bytes") != 10_000_000
        or acquisition.get("downloads") != 2
        or acquisition.get("ocr_allowed")
        or acquisition.get("figure_digitization_allowed")
        or acquisition.get("redistribution_allowed")
        or machine_text.get("extractor") != "pdftotext"
        or machine_text.get("minimum_characters") != 12_000
        or machine_text.get("minimum_japanese_characters") != 3_000
        or len(machine_text.get("required_terms", [])) != 5
        or not gates.get("repeat_download_sha256_exact")
        or not gates.get("repeat_extraction_sha256_exact")
        or not gates.get("no_ocr")
        or not gates.get("no_figure_digitization")
        or not gates.get("no_redistribution")
    ):
        raise JstageFilmGranularitySourceError("P4AL frozen contract drift")
    return value


def _fetch(url: str, *, maximum_bytes: int, timeout: int, user_agent: str) -> bytes:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > maximum_bytes:
            raise JstageFilmGranularitySourceError("P4AL response exceeds byte bound")
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise JstageFilmGranularitySourceError("P4AL response exceeds byte bound")
    return data


def analyze_article_html(raw: bytes, source: dict[str, Any]) -> dict[str, bool]:
    text = html.unescape(raw.decode("utf-8", errors="strict"))
    visible = re.sub(r"<[^>]+>", " ", text)
    visible = re.sub(r"\s+", " ", visible)
    return {
        "title": source["title"] in visible,
        "author": source["author"] in visible,
        "doi": source["doi"] in visible,
        "volume_issue_pages": "1969 Volume 23 Issue 1 Pages 13-23" in visible,
        "free_access": source["expected_access_label"] in visible,
        "copyright_holder": source["expected_copyright_holder"] in visible,
    }


def analyze_machine_text(text: str, required_terms: list[str]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", "", text)
    term_results = {term: term in normalized for term in required_terms}
    japanese_characters = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    return {
        "characters": len(text),
        "japanese_characters": japanese_characters,
        "term_results": term_results,
    }


def _extract_text(pdf: bytes, executable: str, timeout: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="neuro-film-p4al-") as directory:
        root = Path(directory)
        source, destination = root / "source.pdf", root / "source.txt"
        source.write_bytes(pdf)
        completed = subprocess.run(
            [executable, "-enc", "UTF-8", "-layout", str(source), str(destination)],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if completed.returncode != 0 or not destination.is_file():
            raise JstageFilmGranularitySourceError("P4AL pdftotext extraction failed")
        return destination.read_bytes()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def acquire_and_evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    parent = root / config["parent"]["p4ak_decision_path"]
    if not parent.is_file() or _sha256(parent.read_bytes()) != config["parent"]["p4ak_decision_sha256"]:
        raise JstageFilmGranularitySourceError("P4AL parent identity mismatch")
    source, acquisition, machine = config["source"], config["acquisition"], config["machine_text"]
    timeout, agent = int(acquisition["timeout_seconds"]), str(acquisition["user_agent"])
    articles, pdfs = [], []
    for _ in range(int(acquisition["downloads"])):
        articles.append(_fetch(source["article_url"], maximum_bytes=int(source["maximum_article_bytes"]), timeout=timeout, user_agent=agent))
        pdfs.append(_fetch(source["pdf_url"], maximum_bytes=int(source["maximum_pdf_bytes"]), timeout=timeout, user_agent=agent))
    executable = shutil.which(machine["extractor"])
    if not executable:
        raise JstageFilmGranularitySourceError("P4AL frozen extractor unavailable")
    texts = [_extract_text(pdf, executable, timeout) for pdf in pdfs]
    article = analyze_article_html(articles[0], source)
    decoded = texts[0].decode("utf-8", errors="strict")
    analysis = analyze_machine_text(decoded, list(machine["required_terms"]))
    destination = root / acquisition["destination"]
    if not destination.resolve().is_relative_to((root / "data").resolve()):
        raise JstageFilmGranularitySourceError("P4AL destination escaped data root")
    _atomic_write(destination / "takano_1969.pdf", pdfs[0])
    _atomic_write(destination / "takano_1969.txt", texts[0])
    checks = {
        "article_identity": all(article[name] for name in ("title", "author", "doi", "volume_issue_pages")),
        "free_access_label": article["free_access"],
        "copyright_holder": article["copyright_holder"],
        "bounded_pdf": len(pdfs[0]) <= int(source["maximum_pdf_bytes"]),
        "pdf_magic": pdfs[0].startswith(config["gates"]["pdf_magic"].encode()),
        "repeat_download": len({_sha256(value) for value in pdfs}) == 1,
        "repeat_extraction": len({_sha256(value) for value in texts}) == 1,
        "minimum_text": analysis["characters"] >= int(machine["minimum_characters"]),
        "minimum_japanese_text": analysis["japanese_characters"] >= int(machine["minimum_japanese_characters"]),
        "required_terms": all(analysis["term_results"].values()),
        "no_ocr": True,
        "no_figure_digitization": True,
        "no_redistribution": True,
        "parent_identity": True,
    }
    source_pass = all(checks.values())
    branch = "source_pass" if source_pass else "source_fail"
    stable = {
        "experiment_id": config["experiment_id"],
        "source_identity": {
            "doi": source["doi"],
            "article_sha256": _sha256(articles[0]),
            "article_bytes": len(articles[0]),
            "pdf_sha256": _sha256(pdfs[0]),
            "pdf_bytes": len(pdfs[0]),
            "text_sha256": _sha256(texts[0]),
            "text_bytes": len(texts[0]),
            "extractor_path": str(Path(executable).resolve()),
            "extractor_sha256": _sha256(Path(executable).read_bytes()),
        },
        "article_analysis": article,
        "machine_text_analysis": analysis,
        "checks": checks,
        "source_pass": source_pass,
        "branch": branch,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": _sha256(_canonical(stable)),
        "decision": config["branch_rule"][branch],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "JstageFilmGranularitySourceError",
    "REPORT_SCHEMA",
    "SCHEMA",
    "acquire_and_evaluate",
    "analyze_article_html",
    "analyze_machine_text",
    "load_contract",
]
