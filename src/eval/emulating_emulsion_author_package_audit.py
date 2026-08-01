"""Exact primary-author package audit for the Emulating Emulsion baseline."""

from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pypdf import PdfReader


class EmulatingEmulsionPackageError(ValueError):
    """Raised when a frozen author asset drifts or cannot be parsed exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise EmulatingEmulsionPackageError(f"missing author asset: {path}")
    if path.stat().st_size != expected_bytes:
        raise EmulatingEmulsionPackageError(f"author asset byte count drift: {path}")
    if sha256_file(path) != expected_sha256:
        raise EmulatingEmulsionPackageError(f"author asset SHA-256 drift: {path}")


class _AuthorPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self.assets: list[str] = []
        self.text_parts: list[str] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "a" and values.get("href"):
            self._active_href = values["href"]
            self._active_text = []
        if values.get("src"):
            self.assets.append(values["src"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_href is not None:
            self.links.append(
                {
                    "href": self._active_href,
                    "text": " ".join(" ".join(self._active_text).split()),
                }
            )
            self._active_href = None
            self._active_text = []

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._active_href is not None:
            self._active_text.append(data)


def audit_author_page(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    _verify_file(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    parser = _AuthorPageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    normalized_text = " ".join(html.unescape(" ".join(parser.text_parts)).split())
    folded = normalized_text.casefold()
    for dash in ("‐", "‑", "‒", "–", "—", "−", "\u00ad"):
        folded = folded.replace(dash, "-")
    anchors = {
        "title": "emulating emulsion",
        "one_roll": "single 36-exposure roll",
        "unique_rows": "3168 unique correspondences",
        "capture_devices": "fujifilm x-pro3 cameras",
        "film_camera": "nikon f2as",
        "scan_backlight": "d50 backlight",
        "provided_source_code_claim": "provided source code will also operate in prophoto rgb space",
    }
    anchor_presence = {key: value in folded for key, value in anchors.items()}
    if not all(anchor_presence.values()):
        missing = [key for key, present in anchor_presence.items() if not present]
        raise EmulatingEmulsionPackageError(f"author page anchor drift: {missing}")

    github_links = [
        row for row in parser.links if urlparse(row["href"]).netloc == "github.com"
    ]
    repository_links = []
    for row in github_links:
        parts = [part for part in urlparse(row["href"]).path.split("/") if part]
        if len(parts) >= 2:
            repository_links.append(row)
    package_suffix = re.compile(
        r"(?i)(?:\.zip|\.tar\.gz|\.py|\.ipynb|\.csv|\.json|\.npy|\.npz|\.pt|\.pth|\.pkl)$"
    )
    downloadable_package_links = [
        row for row in parser.links if package_suffix.search(urlparse(row["href"]).path)
    ]
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "text_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        "anchor_presence": anchor_presence,
        "links": parser.links,
        "github_links": github_links,
        "explicit_repository_links": repository_links,
        "downloadable_code_or_data_links": downloadable_package_links,
        "embedded_asset_count": len(parser.assets),
        "claims_provided_source_code": anchor_presence["provided_source_code_claim"],
        "published_complete_paired_patch_rows": 0,
        "published_exact_fitted_parameter_values": 0,
    }


def _normalized_pdf_text(reader: PdfReader) -> str:
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    return "\n".join(" ".join(line.split()) for line in raw.splitlines())


def audit_pdf(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_pages: int,
    required_phrases: dict[str, str],
) -> dict[str, Any]:
    _verify_file(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    reader = PdfReader(path)
    if len(reader.pages) != expected_pages:
        raise EmulatingEmulsionPackageError(f"author PDF page count drift: {path}")
    text = _normalized_pdf_text(reader)
    folded = " ".join(text.split()).casefold()
    anchor_presence = {
        key: phrase.casefold() in folded for key, phrase in required_phrases.items()
    }
    if not all(anchor_presence.values()):
        missing = [key for key, present in anchor_presence.items() if not present]
        raise EmulatingEmulsionPackageError(f"author PDF anchor drift: {missing}")
    uris: list[str] = []
    for page in reader.pages:
        for annotation in page.annotations or []:
            obj = annotation.get_object()
            action = obj.get("/A")
            if action and action.get("/URI"):
                uris.append(str(action.get("/URI")))
    code_links = [
        uri
        for uri in sorted(set(uris))
        if urlparse(uri).netloc in {"github.com", "gitlab.com"}
    ]
    metadata = reader.metadata or {}
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "page_count": len(reader.pages),
        "title": str(metadata.get("/Title", "")),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "anchor_presence": anchor_presence,
        "external_uris": sorted(set(uris)),
        "explicit_code_repository_links": code_links,
    }
