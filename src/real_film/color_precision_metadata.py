"""Deterministic metadata-only audit helpers for Color Precision pages.

The parser inventories image URLs embedded in already-retained HTML.  It never
requests or decodes those image payloads.  URL-derived labels are weak source
metadata, not verified physical stock, exposure, process, roll, or pairing
truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


IMAGE_URL_PATTERN = re.compile(
    r"https://colorprecision-cdn\.s3\.us-east-1\.amazonaws\.com/"
    r"(?:Pictures-for-Slider-Comparison|Exterior-Film-Scans-Tool)/"
    r"[^\"'<>\\\s]+?\.(?:png|jpg)",
    flags=re.IGNORECASE,
)
SCANNER_SUFFIX_PATTERN = re.compile(
    r"-(Frontier|Noritsu)\.(png|jpg)$",
    flags=re.IGNORECASE,
)
SHOPIFY_DYNAMIC_META_PATTERN = re.compile(
    r'(<meta name="shopify-y" content=")[^"]+("\s*/?>)',
    flags=re.IGNORECASE,
)

# Ordered longest/specific aliases before shorter names.
STOCK_ALIASES: tuple[tuple[str, str], ...] = (
    ("ColorPlus", "colorplus"),
    ("Ektachrome", "ektachrome"),
    ("Ektar", "ektar"),
    ("Gold", "gold"),
    ("Portra160", "portra-160"),
    ("Portra400", "portra-400"),
    ("Portra800", "portra-800"),
    ("ProImage", "pro-image"),
    ("UltraMax", "ultramax"),
    ("C200", "c200"),
    ("Fuji200", "fuji-200"),
    ("Fuji400", "fuji-400"),
    ("NHGII800", "ngh-800-ii"),
    ("NHGII800", "ngh-ii-800"),
    ("NHGII800", "nhg-ii-800"),
    ("NPS160", "nps-160"),
    ("Natura1600", "natura-1600"),
    ("Pro160S", "pro-160s"),
    ("Pro400H", "pro-400h"),
    ("Provia100F", "provia-100f"),
    ("Superia400", "superia-400"),
    ("Velvia50", "velvia-50"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_dynamic_html(payload: bytes) -> bytes:
    """Remove only the observed per-request Shopify UUID token."""

    text = payload.decode("utf-8", errors="strict")
    return SHOPIFY_DYNAMIC_META_PATTERN.sub(
        r"\g<1><dynamic>\2",
        text,
    ).encode("utf-8")


def extract_comparison_image_urls(payload: bytes) -> list[str]:
    """Return a sorted, deduplicated embedded comparison-image inventory."""

    text = payload.decode("utf-8", errors="strict")
    return sorted(set(IMAGE_URL_PATTERN.findall(text)))


def _stock_id(filename: str) -> str:
    lowered = filename.casefold()
    for stock_id, prefix in STOCK_ALIASES:
        if lowered.startswith(prefix):
            return stock_id
    raise ValueError(f"unknown URL-derived stock label: {filename}")


def parse_comparison_url(url: str) -> dict[str, str]:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError(f"comparison URL is not HTTPS: {url}")
    if parsed.netloc.casefold() != (
        "colorprecision-cdn.s3.us-east-1.amazonaws.com"
    ):
        raise ValueError(f"unexpected comparison host: {url}")
    parts = unquote(parsed.path).strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(f"unexpected comparison path topology: {url}")
    collection, scene, filename = parts
    match = SCANNER_SUFFIX_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"missing scanner suffix: {url}")
    scanner = match.group(1).title()
    extension = match.group(2).lower()
    condition = filename[: match.start()]
    return {
        "url": url,
        "collection": collection,
        "scene": scene,
        "filename": filename,
        "extension": extension,
        "scanner": scanner,
        "stock_id": _stock_id(filename),
        "condition_key": condition.casefold(),
        "pair_key": f"{collection}/{scene}/{condition}".casefold(),
    }


def inventory_comparison_metadata(payload: bytes) -> dict[str, Any]:
    """Summarize embedded URLs without accessing their payloads."""

    records = [
        parse_comparison_url(url)
        for url in extract_comparison_image_urls(payload)
    ]
    pair_scanners: dict[str, set[str]] = defaultdict(set)
    pair_urls: dict[str, list[str]] = defaultdict(list)
    for record in records:
        pair_scanners[record["pair_key"]].add(record["scanner"])
        pair_urls[record["pair_key"]].append(record["url"])

    complete_pair_keys = sorted(
        key
        for key, scanners in pair_scanners.items()
        if scanners == {"Frontier", "Noritsu"}
    )
    incomplete_pair_keys = sorted(
        key
        for key, scanners in pair_scanners.items()
        if scanners != {"Frontier", "Noritsu"}
    )
    cells = {
        (record["collection"], record["scene"], record["stock_id"])
        for record in records
    }
    filenames = [record["filename"].casefold() for record in records]

    return {
        "embedded_image_url_count": len(records),
        "unique_url_count": len({record["url"] for record in records}),
        "collection_counts": _sorted_counter(
            record["collection"] for record in records
        ),
        "scene_counts": _sorted_counter(record["scene"] for record in records),
        "scanner_counts": _sorted_counter(
            record["scanner"] for record in records
        ),
        "extension_counts": _sorted_counter(
            record["extension"] for record in records
        ),
        "stock_label_counts": _sorted_counter(
            record["stock_id"] for record in records
        ),
        "stock_label_count": len({record["stock_id"] for record in records}),
        "metadata_cell_count": len(cells),
        "filename_implied_condition_count": len(pair_scanners),
        "filename_implied_complete_scanner_pair_count": len(complete_pair_keys),
        "filename_implied_incomplete_scanner_pair_count": len(
            incomplete_pair_keys
        ),
        "filename_implied_incomplete_pair_keys": incomplete_pair_keys,
        "pushed_url_count": sum("pushed" in name for name in filenames),
        "cross_processed_url_count": sum(
            "cross-processed" in name for name in filenames
        ),
        "tungsten_url_count": sum("tungsten" in name for name in filenames),
        "records": records,
    }


def _sorted_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def inspect_page_claims(
    comparison: bytes,
    product: bytes,
    scanner: bytes,
    terms: bytes,
) -> dict[str, bool]:
    """Check only exact source statements relevant to the audit boundary."""

    comparison_text = comparison.decode("utf-8", errors="strict")
    product_text = product.decode("utf-8", errors="strict")
    scanner_text = scanner.decode("utf-8", errors="strict")
    terms_text = terms.decode("utf-8", errors="strict")
    return {
        "comparison_claims_6k_35mm_scans": (
            "6K scans of 35mm film" in comparison_text
        ),
        "comparison_discloses_minor_exposure_adjustments": (
            "Only minor exposure adjustments were applied" in comparison_text
        ),
        "comparison_discloses_per_frame_scanner_auto_adjustment": (
            "film scanners automatically adjust color, white balance, and contrast"
            in comparison_text
        ),
        "product_claims_same_lens_and_lighting_pairing": (
            "The same lens and lighting situation is maintained during the digital "
            "and film shoot" in product_text
        ),
        "scanner_article_claims_same_negative_pair": (
            "They're from the same negative" in scanner_text
        ),
        "scanner_article_claims_6k_and_unedited_examples": (
            "scanned at 6K resolution" in scanner_text
            and "examples were not edited after scanning" in scanner_text
        ),
        "terms_updated_2026_07_15": (
            "Last updated: July 15, 2026" in terms_text
        ),
        "terms_personal_nontransferable_product_license": (
            "personal, non-transferable license" in terms_text
        ),
        "terms_forbid_profile_reverse_engineering_or_extraction": (
            "reverse engineer, or extract the emulation profiles" in terms_text
        ),
        "comparison_footer_all_rights_reserved": (
            "All rights reserved" in comparison_text
        ),
        "explicit_comparison_pixel_research_reuse_grant_found": False,
    }
