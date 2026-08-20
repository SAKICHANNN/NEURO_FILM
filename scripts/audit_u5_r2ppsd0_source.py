"""Bounded metadata and annotation-structure audit for official PPSD."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "K-MCFM-U5-R2PPSD0/1.0 (bounded source audit)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError(f"response exceeds frozen byte budget: {url}")
    return content


def _project_facts(html: str) -> dict[str, int]:
    normalized = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    patterns = {
        "retained_users": r"([\d,]+) users",
        "valid_preference_judgments_minimum": r"~([\d,]+) valid preference judgments",
        "unique_scenes": r"([\d,]+) unique scenes",
        "unique_image_style_pairs": r"([\d,]+) unique image style pairs",
        "source_categories": r"across ([\w-]+) source categories",
    }
    facts: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"missing official project fact: {key}")
        token = match.group(1).replace(",", "")
        facts[key] = 5 if token.lower() == "five" else int(token)
    return facts


def _drive_inventory(html: str, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = (
        html.replace(r"\x22", '"')
        .replace(r"\x5b", "[")
        .replace(r"\x5d", "]")
        .replace(r"\/", "/")
    )
    rows: list[dict[str, Any]] = []
    for item in expected:
        pattern = re.compile(
            rf'\[{{1,3}}"{re.escape(item["id"])}",\["[^"]+"\],'
            rf'"{re.escape(item["name"])}","application/zip",'
            rf'0,null,0,0,0,\d+,\d+,null,null,(\d+),'
        )
        match = pattern.search(normalized)
        if match is None:
            raise ValueError(f"missing exact Drive inventory row: {item['name']}")
        rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "mime_type": item["mime_type"],
                "size_bytes": int(match.group(1)),
            }
        )
    return rows


def _annotation_structure(archive: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        infos = sorted(bundle.infolist(), key=lambda item: item.filename)
        files = [item for item in infos if not item.is_dir()]
        role_counts: Counter[str] = Counter()
        suffix_counts: Counter[str] = Counter()
        device_counts: Counter[str] = Counter()
        processed_keys: set[str] = set()
        processed_types: Counter[str] = Counter()
        raw_jsonl: dict[str, dict[str, Any]] = {}
        inventory_rows: list[dict[str, Any]] = []
        for item in files:
            suffix_counts[Path(item.filename).suffix.lower() or "<none>"] += 1
            inventory_rows.append(
                {"path": item.filename, "size_bytes": item.file_size, "crc32": f"{item.CRC:08x}"}
            )
            if item.filename.startswith("__MACOSX/"):
                role_counts["macos_metadata"] += 1
            elif item.filename.endswith(".DS_Store"):
                role_counts["directory_metadata"] += 1
            elif item.filename in {
                "responses/raw/participants.jsonl",
                "responses/raw/votes_items.jsonl",
            }:
                role = Path(item.filename).stem
                role_counts[f"raw_{role}_jsonl"] += 1
                line_count = 0
                keys: set[str] = set()
                for line in bundle.read(item).splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError(f"non-object row in {role} JSONL")
                    line_count += 1
                    keys.update(str(key) for key in payload)
                raw_jsonl[role] = {"row_count": line_count, "top_level_keys": sorted(keys)}
            elif item.filename.startswith("responses/processed/") and item.filename.endswith(".json"):
                role_counts["processed_participant_json"] += 1
                device_match = re.search(r"_(desktop|laptop|mobile|tablet)\.json$", item.filename)
                device_counts[device_match.group(1) if device_match else "unknown"] += 1
                payload = json.loads(bundle.read(item))
                processed_types[type(payload).__name__] += 1
                if isinstance(payload, dict):
                    processed_keys.update(str(key) for key in payload)
            else:
                role_counts["unknown"] += 1
        inventory_identity = _sha256(
            json.dumps(inventory_rows, sort_keys=True, separators=(",", ":")).encode()
        )
        processed_key_rows = sorted(processed_keys)
        processed_key_identity = _sha256(
            json.dumps(processed_key_rows, separators=(",", ":")).encode()
        )
        processed_collection_counts = Counter(
            key.split("-", 1)[0] if "-" in key else "unknown" for key in processed_key_rows
        )
        participant_keys = set(raw_jsonl.get("participants", {}).get("top_level_keys", []))
        return {
            "archive_sha256": _sha256(archive),
            "file_count": len(files),
            "total_uncompressed_bytes": sum(item.file_size for item in files),
            "central_directory_identity_sha256": inventory_identity,
            "role_counts": dict(sorted(role_counts.items())),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "processed_device_counts": dict(sorted(device_counts.items())),
            "processed_top_level_types": dict(sorted(processed_types.items())),
            "processed_scene_key_count": len(processed_key_rows),
            "processed_scene_key_identity_sha256": processed_key_identity,
            "processed_collection_counts": dict(sorted(processed_collection_counts.items())),
            "raw_jsonl": raw_jsonl,
            "participant_sensitive_fields_present": bool(
                participant_keys
                & {"_id", "age", "gender", "job", "region", "user_id", "device_model"}
            ),
            "participant_identifiers_persisted": False,
        }


def evaluate(contract: dict[str, Any], *, fetch=_fetch) -> dict[str, Any]:
    bounded = contract["bounded_execution"]
    sources = contract["official_sources"]
    project = fetch(sources["project_page_url"], bounded["maximum_project_page_bytes"])
    drive = fetch(sources["drive_folder_url"], bounded["maximum_drive_listing_bytes"])
    facts = _project_facts(project.decode("utf-8"))
    inventory = _drive_inventory(drive.decode("utf-8"), sources["expected_drive_files"])
    responses = next(row for row in sources["expected_drive_files"] if row["name"] == "responses.zip")
    response_url = f"https://drive.usercontent.google.com/download?id={responses['id']}&export=download"
    archive = fetch(response_url, bounded["maximum_responses_archive_bytes"])
    structure = _annotation_structure(archive)
    project_license_observed = bool(
        re.search(r"\b(?:license|licence)\b", project.decode("utf-8"), re.IGNORECASE)
    )
    inventory_exact = inventory == [
        {key: row[key] for key in ("id", "name", "mime_type", "size_bytes")}
        for row in sources["expected_drive_files"]
    ]
    structure_auditable = (
        structure["role_counts"].get("unknown", 0) == 0
        and structure["role_counts"].get("raw_participants_jsonl", 0) == 1
        and structure["role_counts"].get("raw_votes_items_jsonl", 0) == 1
        and structure["role_counts"].get("processed_participant_json", 0) > 0
        and structure["participant_identifiers_persisted"] is False
    )
    gates = {
        "official_pages_reachable": True,
        "project_facts_exact": facts == sources["expected_project_facts"],
        "drive_inventory_exact": inventory_exact,
        "responses_archive_within_budget": len(archive) <= bounded["maximum_responses_archive_bytes"],
        "annotation_structure_auditable_without_persisting_participant_values": structure_auditable,
        "pixels_or_training_allowed": False,
    }
    automatic_pass = all(value for key, value in gates.items() if key != "pixels_or_training_allowed")
    decision = contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"]
    scientific = {
        "schema": "neuro_film.u5_r2ppsd0_source_audit_report.v1",
        "experiment_id": contract["experiment_id"],
        "project_page": {
            "canonical_facts": facts,
            "license_token_observed": project_license_observed,
        },
        "drive_inventory": inventory,
        "annotation_structure": structure,
        "requests": {
            "project_pages": 2,
            "annotation_archives": 1,
            "image_archives": 0,
            "image_members": 0,
            "image_decodes": 0,
            "operator_fits": 0,
        },
        "rights_scope": contract["license_gate"]["missing_license_fallback"],
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_evidence_id"] = f"sha256:{_sha256(json.dumps(scientific, sort_keys=True, separators=(',', ':')).encode())}"
    return scientific


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ppsd0_source_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "stable_evidence_id": report["stable_evidence_id"]}))


if __name__ == "__main__":
    main()
