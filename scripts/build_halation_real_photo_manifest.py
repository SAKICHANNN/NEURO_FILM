#!/usr/bin/env python3
"""Build a license-aware real-photo source manifest for halation validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_CATEGORY = "Photographs taken on Kodak Vision3 500T film"
USER_AGENT = "neuro-film-halation-validation/0.1 (research script; Wikimedia Commons API)"

FIELDNAMES = [
    "source_id",
    "url",
    "page_url",
    "author",
    "title",
    "license",
    "license_url",
    "stock_claim",
    "stock_claim_confidence",
    "source_type",
    "download_allowed",
    "redistribution_allowed",
    "derivative_allowed",
    "local_path",
    "sha256",
    "width",
    "height",
    "notes",
    "audit_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-film halation source manifest.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eval/halation_real_photo_v1"))
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        help="Additional Commons file search query, for example '\"Kodak Vision3 500T\" night'.",
    )
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--download", action="store_true", help="Download only rows whose license permits local analysis.")
    parser.add_argument("--max-downloads", type=int, default=12)
    parser.add_argument("--thumb-width", type=int, default=1600, help="Prefer Commons scaled URL for analysis cache.")
    return parser.parse_args()


def fetch_json(url: str, params: dict[str, str | int]) -> dict:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def commons_category_members(category: str, max_pages: int) -> list[str]:
    titles: list[str] = []
    cmcontinue: str | None = None
    cmtitle = category if category.startswith("Category:") else f"Category:{category}"
    while len(titles) < max_pages:
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": cmtitle,
            "cmtype": "file",
            "cmlimit": min(50, max_pages - len(titles)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = fetch_json(COMMONS_API, params)
        titles.extend(item["title"] for item in data.get("query", {}).get("categorymembers", []))
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles[:max_pages]


def commons_imageinfo(titles: list[str], thumb_width: int) -> list[dict]:
    if not titles:
        return []
    rows: list[dict] = []
    for start in range(0, len(titles), 25):
        batch = titles[start : start + 25]
        data = fetch_json(
            COMMONS_API,
            {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "titles": "|".join(batch),
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": thumb_width,
            },
        )
        rows.extend(data.get("query", {}).get("pages", {}).values())
        time.sleep(0.1)
    return rows


def metadata_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        return str(value.get("value") or "").strip()
    return ""


def plain_text(value: str) -> str:
    return " ".join(value.replace("<span", " <span").split())


def license_policy(license_name: str, usage_terms: str) -> tuple[bool, bool, bool, str]:
    text = f"{license_name} {usage_terms}".lower()
    if any(token in text for token in ["copyrighted", "all rights reserved", "fair use"]):
        return False, False, False, "copyright-like license; URL-only"
    if any(token in text for token in ["cc0", "public domain", "pd-old", "pd-self"]):
        return True, True, True, "open/public-domain Commons metadata"
    if any(token in text for token in ["cc by-sa", "cc-by-sa", "cc by ", "cc-by-", "attribution"]):
        return True, True, True, "open Commons license; keep attribution in manifest"
    if "gfdl" in text:
        return True, False, False, "GFDL; local analysis only, avoid derivative redistribution"
    return False, False, False, "unknown license; URL-only"


def commons_page_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/_')}"


def source_id(prefix: str, index: int, title: str) -> str:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{index:03d}_{digest}"


def commons_search_pages(query: str, max_pages: int, thumb_width: int) -> list[dict]:
    data = fetch_json(
        COMMONS_API,
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": max_pages,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": thumb_width,
        },
    )
    pages = list(data.get("query", {}).get("pages", {}).values())
    pages.sort(key=lambda page: int(page.get("index", 9999)))
    return pages


def rows_from_pages(
    pages: list[dict],
    *,
    prefix: str,
    stock_claim: str,
    stock_claim_confidence: str,
    source_type: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, page in enumerate(pages, start=1):
        title = page.get("title", "")
        imageinfo = (page.get("imageinfo") or [{}])[0]
        metadata = imageinfo.get("extmetadata") or {}
        license_name = metadata_value(metadata, "LicenseShortName") or metadata_value(metadata, "UsageTerms")
        usage_terms = metadata_value(metadata, "UsageTerms")
        license_url = metadata_value(metadata, "LicenseUrl")
        author = plain_text(metadata_value(metadata, "Artist") or metadata_value(metadata, "Credit"))
        object_name = metadata_value(metadata, "ObjectName") or title
        allowed, redistribute, derivative, note = license_policy(license_name, usage_terms)
        rows.append(
            {
                "source_id": source_id(prefix, index, title),
                "url": imageinfo.get("thumburl") or imageinfo.get("url") or "",
                "page_url": metadata_value(metadata, "ImageDescriptionUrl") or commons_page_url(title),
                "author": author,
                "title": plain_text(object_name),
                "license": plain_text(license_name or usage_terms or "unknown"),
                "license_url": license_url,
                "stock_claim": stock_claim,
                "stock_claim_confidence": stock_claim_confidence,
                "source_type": source_type,
                "download_allowed": str(bool(allowed)),
                "redistribution_allowed": str(bool(redistribute)),
                "derivative_allowed": str(bool(derivative)),
                "local_path": "",
                "sha256": "",
                "width": str(imageinfo.get("thumbwidth") or imageinfo.get("width") or ""),
                "height": str(imageinfo.get("thumbheight") or imageinfo.get("height") or ""),
                "notes": note,
                "audit_date": date.today().isoformat(),
            }
        )
    return rows


def build_commons_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    titles = commons_category_members(args.category, args.max_pages)
    pages = commons_imageinfo(titles, args.thumb_width)
    return rows_from_pages(
        pages,
        prefix="commons_vision3_500t",
        stock_claim=args.category,
        stock_claim_confidence="user-tagged Commons category",
        source_type="wikimedia_commons_category",
    )


def build_search_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for search_index, query in enumerate(args.search, start=1):
        pages = commons_search_pages(query, args.max_pages, args.thumb_width)
        rows.extend(
            rows_from_pages(
                pages,
                prefix=f"commons_search_{search_index:02d}",
                stock_claim=query,
                stock_claim_confidence="Commons search result; confirm file description/category before treating as stock-specific",
                source_type="wikimedia_commons_search",
            )
        )
    return rows


def reference_rows() -> list[dict[str, str]]:
    today = date.today().isoformat()
    refs = [
        (
            "kodak_vision3_500t_reference",
            "https://www.kodak.com/en/motion/product/camera-films/500t-5219-7219",
            "Kodak VISION3 500T official product reference",
            "official Kodak stock reference; cite-only, do not download sample imagery",
        ),
        (
            "cinestill_remjet_reference",
            "https://help.cinestillfilm.com/hc/en-us/articles/360028874012-What-is-Remjet",
            "CineStill remjet support reference",
            "behavioral reference for no-remjet halation; cite-only",
        ),
        (
            "dehancer_cinestill_800t_reference",
            "https://www.dehancer.com/profiles/film/cinestill-800t",
            "Dehancer CineStill 800T reference",
            "behavioral/reference page only; do not download example imagery",
        ),
    ]
    rows = []
    for source_id_value, url, title, notes in refs:
        rows.append(
            {
                "source_id": source_id_value,
                "url": url,
                "page_url": url,
                "author": "",
                "title": title,
                "license": "citation-only",
                "license_url": "",
                "stock_claim": title,
                "stock_claim_confidence": "official/reference page",
                "source_type": "citation_reference",
                "download_allowed": "False",
                "redistribution_allowed": "False",
                "derivative_allowed": "False",
                "local_path": "",
                "sha256": "",
                "width": "",
                "height": "",
                "notes": notes,
                "audit_date": today,
            }
        )
    return rows


def download_allowed_rows(rows: list[dict[str, str]], output_root: Path, max_downloads: int) -> None:
    download_dir = output_root / "downloaded"
    download_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        if row["download_allowed"] != "True" or not row["url"]:
            continue
        existing = next(download_dir.glob(f"{row['source_id']}.*"), None)
        if existing is not None:
            payload = existing.read_bytes()
            row["local_path"] = str(existing.as_posix())
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            continue
        if count >= max_downloads:
            break
        suffix = ".jpg"
        if ".png" in row["url"].lower():
            suffix = ".png"
        local_path = download_dir / f"{row['source_id']}{suffix}"
        request = Request(row["url"], headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
        except HTTPError as exc:
            row["notes"] = f"{row['notes']}; download skipped after HTTP {exc.code}"
            if exc.code == 429:
                break
            continue
        except URLError as exc:
            row["notes"] = f"{row['notes']}; download skipped after URL error {exc.reason}"
            continue
        local_path.write_bytes(payload)
        row["local_path"] = str(local_path.as_posix())
        row["sha256"] = hashlib.sha256(payload).hexdigest()
        count += 1
        time.sleep(0.2)


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_commons_rows(args) + build_search_rows(args) + reference_rows()
    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in rows:
        key = row["page_url"] or row["url"] or row["title"]
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(row)
    rows = deduped
    if args.download:
        download_allowed_rows(rows, args.output_root, args.max_downloads)
    manifest = args.output_root / "sources_manifest.csv"
    write_csv(rows, manifest)
    audit = {
        "manifest": str(manifest.as_posix()),
        "category": args.category,
        "rows": len(rows),
        "downloaded": sum(1 for row in rows if row["local_path"]),
        "url_only": sum(1 for row in rows if not row["local_path"]),
        "policy": "Only rows marked download_allowed=True are cached, and all local files live under ignored outputs/.",
    }
    (args.output_root / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
