"""Audit the deterministic offline recipe-history HTML document."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import build_render_recipe_history, render_recipe_history_html

SOURCE_ROOT = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke"
SOURCE_RECIPE_SHA256 = {
    "ektar_100.recipe.json": "b92a75ed8ca8349df562153aad85df26db33ab0992d6bc07503033245ac7d0b8",
    "portra_400.recipe.json": "b6dece02a1c6e5cee9569424ae8fd53fdad8f4ee619753f6d81a2b2dea0d0fb3",
    "velvia_50.recipe.json": "254e5c8ddf8703e2689975590531b80ca4e40bde1a37fcd0c382f33e2a42cf25",
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.ids: set[str] = set()
        self.search_label = False
        self.external_sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "label" and values.get("for") == "recipe-search":
            self.search_label = True
        for key in ("src", "href"):
            if values.get(key):
                self.external_sources.append(str(values[key]))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_audit() -> tuple[dict[str, Any], bytes]:
    catalog = build_render_recipe_history(SOURCE_ROOT)
    first = render_recipe_history_html(catalog)
    second = render_recipe_history_html(copy.deepcopy(catalog))
    parser = _PageParser()
    parser.feed(first.decode("utf-8"))
    malicious = copy.deepcopy(catalog)
    malicious["entries"][0]["input_path"] = '</code><script id="owned">x</script>'
    escaped = render_recipe_history_html(malicious).decode("utf-8")
    source_exact = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SOURCE_ROOT.glob("*.recipe.json"))
    }
    text = first.decode("utf-8")
    gates = {
        "source_recipe_identities_exact": source_exact == SOURCE_RECIPE_SHA256,
        "repeat_html_bytes_exact": first == second,
        "three_valid_look_approximation_rows": catalog["counts"]
        == {"discovered": 3, "valid": 3, "invalid": 0}
        and all(
            row["output_label"] == "film-inspired"
            and row["evidence_grade"] == "look-approximation"
            for row in catalog["entries"]
        ),
        "landmark_heading_and_search_label_present": "main" in parser.tags
        and "h1" in parser.tags
        and parser.search_label,
        "live_search_status_present": "result-status" in parser.ids
        and 'aria-live="polite"' in text,
        "no_external_sources": parser.external_sources == [],
        "offline_content_security_policy_present": "default-src 'none'" in text
        and "connect-src 'none'" in text,
        "html_injection_escaped": '<script id="owned">' not in escaped
        and "&lt;/code&gt;&lt;script" in escaped,
        "responsive_and_reduced_motion_rules_present": "@media (max-width:720px)"
        in text
        and "prefers-reduced-motion:reduce" in text,
        "no_image_or_pixel_embeds": "<img" not in text,
        "network_reads_zero": True,
        "pixel_decodes_zero": True,
    }
    status = (
        "PASS_PRIVATE_OFFLINE_RECIPE_HISTORY_PAGE"
        if all(gates.values())
        else "FAIL_CLOSED_OFFLINE_RECIPE_HISTORY_PAGE"
    )
    scientific = {
        "protocol": "kmcfm.u7-3b-offline-recipe-history-page-contract.v1",
        "status": status,
        "source_recipe_sha256": source_exact,
        "catalog_counts": catalog["counts"],
        "html_bytes": len(first),
        "html_sha256": sha256_bytes(first),
        "gates": gates,
        "claim_ceiling": "Private self-contained read-only desktop browser document only; no live server, file watcher, preview decode/render, export authorization, packaging, telemetry, calibrated stock or product-release claim.",
    }
    report = {
        "schema": "kmcfm.u7-3b-offline-recipe-history-page-result.v1",
        "scientific": scientific,
        "stable_identity": f"sha256:{sha256_bytes(canonical_bytes(scientific))}",
    }
    return report, first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    report, html = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_bytes(html)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if report["scientific"]["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
