"""Audit official BALit metadata without requesting dataset payloads."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P250Error(RuntimeError):
    """Raised when the frozen P250 execution boundary is violated."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _plain_text(body: bytes) -> str:
    decoded = body.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(decoded)
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _fetch(url: str, maximum_bytes: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p250-source-audit/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise P250Error("official response exceeds frozen byte ceiling")
            return {
                "body": body,
                "final_url": response.geturl(),
                "http_status": int(response.status),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(maximum_bytes + 1)
        if len(body) > maximum_bytes:
            raise P250Error("official error response exceeds frozen byte ceiling")
        return {
            "body": body,
            "final_url": exc.geturl(),
            "http_status": int(exc.code),
        }


def _extract_facts(text: str, expected: dict[str, Any]) -> dict[str, Any]:
    lower = text.casefold()
    name_present = expected["dataset_name"].casefold() in lower
    doi_present = expected["doi"].casefold() in lower
    raw_count_present = bool(
        re.search(r"1[,.]?000[^.]{0,100}(?:raw|cr2)", lower)
        or re.search(r"(?:raw|cr2)[^.]{0,100}1[,.]?000", lower)
    )
    reference_count_present = bool(
        re.search(r"1[,.]?000[^.]{0,120}(?:expert[- ]retouched|reference)", lower)
        or re.search(r"(?:expert[- ]retouched|reference)[^.]{0,120}1[,.]?000", lower)
    )
    cr2_present = "cr2" in lower
    raw_size_present = "21.5 gb" in lower or "21.5gb" in lower
    reference_size_present = "26.84 gb" in lower or "26.84gb" in lower
    login_required = bool(re.search(r"log\s*in to access dataset files", lower))
    explicit_noncommercial = bool(
        re.search(r"non[- ]commercial|cc\s*by[- ]nc|by-nc", lower)
    )
    commercial_compatible_license = bool(
        re.search(
            r"creative commons attribution 4\.0|cc\s*by\s*4\.0|"
            r"apache license 2\.0|apache-2\.0|mit license",
            lower,
        )
    ) and not explicit_noncommercial
    exact_inventory = bool(
        re.search(r"sha-?256|sha-?1|md5|checksum", lower)
        and re.search(r"file[- ]?name|file list|manifest", lower)
    )
    group_identity = bool(
        re.search(r"scene[_ -]?id|group[_ -]?id|capture[_ -]?id", lower)
        and re.search(r"train(?:ing)?[^.]{0,100}validation[^.]{0,100}test", lower)
    )
    return {
        "commercial_compatible_license": commercial_compatible_license,
        "cr2_present": cr2_present,
        "doi_present": doi_present,
        "exact_inventory": exact_inventory,
        "explicit_noncommercial": explicit_noncommercial,
        "group_identity_and_roles": group_identity,
        "login_required": login_required,
        "name_present": name_present,
        "raw_count_present": raw_count_present,
        "raw_size_present": raw_size_present,
        "reference_count_present": reference_count_present,
        "reference_size_present": reference_size_present,
    }


def execute(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    maximum = int(sources["maximum_response_bytes_each"])
    responses = {
        "doi": _fetch(sources["doi_url"], maximum),
        "record": _fetch(sources["record_url"], maximum),
    }
    total_bytes = sum(len(value["body"]) for value in responses.values())
    if total_bytes > int(sources["maximum_total_network_bytes"]):
        raise P250Error("formal network bytes exceed frozen total ceiling")
    combined_text = " ".join(_plain_text(value["body"]) for value in responses.values())
    facts = _extract_facts(combined_text, config["expected_discovery_facts"])
    successful = all(value["http_status"] == 200 for value in responses.values())
    official_identity = successful and facts["name_present"] and facts["doi_present"]
    paired_observation = (
        facts["raw_count_present"]
        and facts["reference_count_present"]
        and facts["cr2_present"]
        and facts["raw_size_present"]
        and facts["reference_size_present"]
    )
    gates = {
        "anonymous_public_payload": not facts["login_required"],
        "commercial_rights": facts["commercial_compatible_license"],
        "group_isolation_facts": facts["group_identity_and_roles"],
        "official_identity": official_identity,
        "paired_observation": paired_observation,
        "public_exact_inventory": facts["exact_inventory"],
    }
    decision = (
        "PASS_PRIVATE_BALIT_PAIRED_SOURCE_ELIGIBILITY"
        if all(gates.values())
        else "NOT_READY_BALIT_SOURCE_RIGHTS_ACCESS_OR_GROUPING_GAP_NOT_SCIENTIFIC_RESULT"
    )
    transport = {
        key: {
            "body_bytes": len(value["body"]),
            "body_sha256": _sha256_bytes(value["body"]),
            "final_url": value["final_url"],
            "http_status": value["http_status"],
        }
        for key, value in responses.items()
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "dataset_file_requests": 0,
        "decision": decision,
        "experiment_id": "P250",
        "facts": facts,
        "gates": gates,
        "image_or_thumbnail_requests": 0,
        "network_bytes": total_bytes,
        "schema": "neuro-film.p250-balit-paired-retouch-source-eligibility-result.v1",
        "transport": transport,
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
