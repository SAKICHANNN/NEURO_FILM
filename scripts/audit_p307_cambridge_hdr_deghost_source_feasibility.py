"""Range-only audit of fresh Cambridge HDR-deghost exposure-stack roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    locate_central_directory,
    parse_central_directory,
)


class P307Error(RuntimeError):
    """Raised when the frozen P307 source boundary is violated."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _verify_binding(bindings: dict[str, object], prefix: str) -> bool:
    body = (ROOT / str(bindings[f"{prefix}_path"])).read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256(body) == str(
        bindings[f"{prefix}_sha256"]
    )


def _fetch_exact(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    request = urllib.request.Request(
        str(source["url"]),
        headers={
            "Accept": "application/hal+json,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p307-formal/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(maximum_bytes + 1)
        status = int(response.status)
        final_url = response.geturl()
    if status != 200 or len(body) > maximum_bytes:
        raise P307Error("official metadata response violates frozen transport")
    if len(body) != int(source["bytes"]) or _sha256(body) != str(source["sha256"]):
        raise P307Error("official metadata identity differs from frozen source")
    return {
        "body": body,
        "bytes": len(body),
        "sha256": _sha256(body),
        "status": status,
        "final_url": final_url,
    }


def _fetch_range(url: str, start: int, end: int, total: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "neuro-film-p307-formal/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(end - start + 2)
        status = int(response.status)
        content_range = response.headers.get("Content-Range")
        final_url = response.geturl()
    expected = f"bytes {start}-{end}/{total}"
    if status != 206 or content_range != expected or len(body) != end - start + 1:
        raise P307Error("archive endpoint did not honor the exact frozen Range")
    return {
        "body": body,
        "bytes": len(body),
        "sha256": _sha256(body),
        "status": status,
        "content_range": content_range,
        "final_url": final_url,
    }


def _is_unsafe(member: ZipMember) -> bool:
    if "\\" in member.name or "\x00" in member.name:
        return True
    path = PurePosixPath(member.name)
    return path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)


def _is_symlink(member: ZipMember) -> bool:
    host = member.version_made_by >> 8
    mode = member.external_attributes >> 16
    return host == 3 and (mode & 0o170000) == 0o120000


def _member_record(member: ZipMember) -> dict[str, object]:
    return {
        "name": member.name,
        "flags": member.flags,
        "method": member.method,
        "crc32": member.crc32,
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "local_offset": member.local_offset,
    }


def analyze_members(members: list[ZipMember], expected_categories: list[str]) -> dict[str, object]:
    names = [member.name for member in members]
    unsafe = [member.name for member in members if _is_unsafe(member)]
    encrypted = [member.name for member in members if member.flags & 1]
    symlinks = [member.name for member in members if _is_symlink(member)]
    unsupported = [member.name for member in members if member.method not in (0, 8)]
    groups: dict[tuple[str, str], dict[tuple[str, str], list[ZipMember]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unexpected_files: list[str] = []
    for member in members:
        if member.name.endswith("/"):
            continue
        parts = PurePosixPath(member.name).parts
        if len(parts) == 2 and parts[-1].casefold() == "readme.txt":
            continue
        if len(parts) != 6:
            unexpected_files.append(member.name)
            continue
        _, category, image_set, role, file_format, filename = parts
        suffix = PurePosixPath(filename).suffix.casefold()
        expected_suffix = ".cr2" if file_format == "raw" else ".jpg"
        if (
            category not in expected_categories
            or not re.fullmatch(r"image_set[1-4]", image_set)
            or role not in ("ghosted", "ground_truth")
            or file_format not in ("jpg", "raw")
            or suffix != expected_suffix
        ):
            unexpected_files.append(member.name)
            continue
        groups[(category, image_set)][(role, file_format)].append(member)

    role_keys = [
        (role, file_format)
        for role in ("ghosted", "ground_truth")
        for file_format in ("jpg", "raw")
    ]
    incomplete: list[dict[str, object]] = []
    invalid_overlap: list[dict[str, object]] = []
    overlap_count = 0
    for (category, image_set), roles in sorted(groups.items()):
        counts = {f"{role}_{fmt}": len(roles[(role, fmt)]) for role, fmt in role_keys}
        if any(value != 5 for value in counts.values()):
            incomplete.append({"category": category, "image_set": image_set, "counts": counts})
        for file_format in ("jpg", "raw"):
            ground = defaultdict(list)
            for member in roles[("ground_truth", file_format)]:
                ground[(member.crc32, member.uncompressed_size)].append(member)
            overlaps: list[tuple[ZipMember, ZipMember]] = []
            for ghosted in roles[("ghosted", file_format)]:
                overlaps.extend(
                    (ghosted, reference)
                    for reference in ground[(ghosted.crc32, ghosted.uncompressed_size)]
                )
            overlap_count += len(overlaps)
            for ghosted, reference in overlaps:
                if "gt3" not in ghosted.name.casefold() or "gt3" not in reference.name.casefold():
                    invalid_overlap.append(
                        {
                            "ghosted": ghosted.name,
                            "ground_truth": reference.name,
                            "format": file_format,
                        }
                    )
            if len(overlaps) > 1:
                invalid_overlap.append(
                    {
                        "category": category,
                        "image_set": image_set,
                        "format": file_format,
                        "overlap_count": len(overlaps),
                    }
                )

    categories = sorted({category for category, _ in groups})
    expected_groups = {
        (category, f"image_set{index}")
        for category in expected_categories
        for index in range(1, 5)
    }
    return {
        "member_count": len(members),
        "member_names_sha256": _sha256(
            json.dumps(names, separators=(",", ":")).encode()
        ),
        "member_inventory_sha256": _canonical_sha256([_member_record(m) for m in members]),
        "categories": categories,
        "group_count": len(groups),
        "group_keys_sha256": _canonical_sha256(
            [[category, image_set] for category, image_set in sorted(groups)]
        ),
        "complete_group_count": len(groups) - len(incomplete),
        "incomplete_groups": incomplete,
        "missing_group_count": len(expected_groups - set(groups)),
        "extra_group_count": len(set(groups) - expected_groups),
        "allowed_gt3_overlap_count": overlap_count - len(invalid_overlap),
        "invalid_role_overlaps": invalid_overlap,
        "unexpected_file_count": len(unexpected_files),
        "unexpected_files": unexpected_files,
        "duplicate_name_count": len(names) - len(set(names)),
        "unsafe_member_count": len(unsafe),
        "encrypted_member_count": len(encrypted),
        "symlink_member_count": len(symlinks),
        "unsupported_method_member_count": len(unsupported),
    }


def _metadata_sources(config: dict[str, Any]) -> dict[str, dict[str, object]]:
    item = config["official_item"]
    meta = config["metadata_sources"]
    return {
        "item": {"url": item["api_url"], "bytes": item["api_bytes"], "sha256": item["api_sha256"]},
        "original_bundle": {
            "url": meta["original_bundle_url"],
            "bytes": meta["original_bundle_bytes"],
            "sha256": meta["original_bundle_sha256"],
        },
        "license_bundle": {
            "url": meta["license_bundle_url"],
            "bytes": meta["license_bundle_bytes"],
            "sha256": meta["license_bundle_sha256"],
        },
        "readme": {"url": meta["readme_url"], "bytes": meta["readme_bytes"], "sha256": meta["readme_sha256"]},
        "license": {"url": meta["license_url"], "bytes": meta["license_bytes"], "sha256": meta["license_sha256"]},
    }


def _metadata_facts(config: dict[str, Any], responses: dict[str, dict[str, object]]) -> dict[str, object]:
    item = json.loads(bytes(responses["item"]["body"]))
    original = json.loads(bytes(responses["original_bundle"]["body"]))
    licence_bundle = json.loads(bytes(responses["license_bundle"]["body"]))
    readme = bytes(responses["readme"]["body"]).decode("utf-8")
    licence = bytes(responses["license"]["body"]).decode("utf-8")
    normalized_readme = _normalized_text(readme)
    original_rows = {row["name"]: row for row in original["_embedded"]["bitstreams"]}
    licence_rows = {row["name"]: row for row in licence_bundle["_embedded"]["bitstreams"]}
    archive_rows_exact = all(
        archive["name"] in original_rows
        and original_rows[archive["name"]]["uuid"] == archive["uuid"]
        and int(original_rows[archive["name"]]["sizeBytes"]) == int(archive["bytes"])
        and original_rows[archive["name"]]["checkSum"]["checkSumAlgorithm"] == "MD5"
        and original_rows[archive["name"]]["checkSum"]["value"] == archive["md5"]
        for archive in config["archives"]
    )
    rights = [value["value"] for value in item["metadata"].get("dc.rights", [])]
    rights_uri = [value["value"] for value in item["metadata"].get("dc.rights.uri", [])]
    role_phrases = (
        "test image stacks, with motion and misalignment",
        "motion-free reference image stacks",
        "36 scenes",
        "9 categories of motion type",
        "both test and reference multi-exposure sequence were captured",
        "perfectly aligned (i.e. ground truth sequence)",
    )
    return {
        "official_item_identity": item["uuid"] == config["official_item"]["uuid"]
        and item["name"] == config["official_item"]["title"]
        and item["metadata"]["dc.identifier.doi"][0]["value"] == config["official_item"]["doi"],
        "archive_inventory_exact": archive_rows_exact,
        "readme_inventory_exact": config["metadata_sources"]["readme_uuid"]
        == original_rows["README.txt"]["uuid"],
        "license_inventory_exact": config["metadata_sources"]["license_uuid"]
        == licence_rows["license.txt"]["uuid"],
        "cc_by_4_0": "Attribution 4.0 International (CC BY 4.0)" in rights
        and "https://creativecommons.org/licenses/by/4.0/" in rights_uri
        and "Creative Commons Attribution license (CC BY)" in normalized_readme,
        "repository_deposit_license_present": licence.startswith(
            "University of Cambridge institutional repository DEPOSIT LICENCE AGREEMENT"
        ),
        "paired_role_semantics": all(
            phrase in normalized_readme for phrase in role_phrases
        ),
        "readme_part3_typo_observed": readme.count("* exposure_stacks_part1.zip") == 2
        and "* exposure_stacks_part3.zip" not in readme,
    }


def execute(config_path: Path, order: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_READY_FOR_FORMAL_EXECUTION":
        raise P307Error("P307 config is not execution-locked")
    bindings = {
        name: _verify_binding(config["bindings"], name)
        for name in ("contract", "runner", "test")
    }
    if not all(bindings.values()):
        raise P307Error("frozen local binding differs")

    metadata = {
        name: _fetch_exact(source, int(config["transport"]["maximum_response_bytes_metadata"]))
        for name, source in _metadata_sources(config).items()
    }
    metadata_facts = _metadata_facts(config, metadata)
    archives = list(config["archives"])
    if order == "reverse":
        archives.reverse()
    archive_facts: dict[str, dict[str, object]] = {}
    archive_network_bytes = 0
    for archive in archives:
        discovery = archive["discovery"]
        tail = _fetch_range(
            archive["url"], discovery["tail_start"], discovery["tail_end"], archive["bytes"]
        )
        if tail["sha256"] != discovery["tail_sha256"]:
            raise P307Error("frozen archive tail identity differs")
        offset, size, count = locate_central_directory(
            bytes(tail["body"]), archive_size=int(archive["bytes"])
        )
        if (offset, size, count) != (
            discovery["central_offset"],
            discovery["central_size"],
            discovery["entry_count"],
        ):
            raise P307Error("frozen EOCD facts differ")
        central = _fetch_range(archive["url"], offset, offset + size - 1, archive["bytes"])
        if central["sha256"] != discovery["central_sha256"]:
            raise P307Error("frozen central-directory identity differs")
        members = parse_central_directory(bytes(central["body"]))
        if len(members) != count:
            raise P307Error("central-directory entry count differs")
        analysis = analyze_members(members, list(discovery["categories"]))
        if analysis["member_names_sha256"] != discovery["member_names_sha256"]:
            raise P307Error("frozen member-name inventory differs")
        archive_network_bytes += int(tail["bytes"]) + int(central["bytes"])
        archive_facts[archive["name"]] = {
            "archive_bytes": archive["bytes"],
            "archive_md5_from_official_inventory": archive["md5"],
            "archive_uuid": archive["uuid"],
            "tail": {key: value for key, value in tail.items() if key != "body"},
            "central_directory": {
                key: value for key, value in central.items() if key != "body"
            },
            "analysis": analysis,
        }

    ordered_archive_facts = {name: archive_facts[name] for name in sorted(archive_facts)}
    all_analysis = [value["analysis"] for value in ordered_archive_facts.values()]
    categories = sorted({category for value in all_analysis for category in value["categories"]})
    total_groups = sum(int(value["group_count"]) for value in all_analysis)
    archive_safe = all(
        all(
            int(value[key]) == 0
            for key in (
                "duplicate_name_count",
                "unsafe_member_count",
                "encrypted_member_count",
                "symlink_member_count",
                "unsupported_method_member_count",
            )
        )
        for value in all_analysis
    )
    role_structure_complete = all(
        all(
            int(value[key]) == 0
            for key in (
                "unexpected_file_count",
                "missing_group_count",
                "extra_group_count",
            )
        )
        and not value["incomplete_groups"]
        and not value["invalid_role_overlaps"]
        for value in all_analysis
    )
    metadata_network_bytes = sum(int(value["bytes"]) for value in metadata.values())
    total_network_bytes = metadata_network_bytes + archive_network_bytes
    gates = {
        "exact_official_metadata": all(metadata_facts.values()),
        "cc_by_4_0": bool(metadata_facts["cc_by_4_0"]),
        "exact_range_transport": True,
        "safe_single_disk_archives": archive_safe,
        "exact_fresh_group_count": categories == config["fresh_roles"]["categories"]
        and total_groups == config["fresh_roles"]["expected_group_count"],
        "complete_paired_roles": role_structure_complete
        and all(
            int(value["complete_group_count"]) == int(value["group_count"])
            for value in all_analysis
        ),
        "forbidden_role_absence": not set(categories) & set(config["forbidden"]["categories"]),
        "zero_member_and_pixel_reads": True,
        "network_budget": total_network_bytes <= config["transport"]["maximum_formal_network_bytes"],
    }
    decision = (
        "PASS_PRIVATE_CAMBRIDGE_HDR_DEGHOST_SOURCE_ROLE_LOCK"
        if all(gates.values())
        else "FAIL_CLOSED_CAMBRIDGE_HDR_DEGHOST_SOURCE_STRUCTURE_GAP_NOT_SCIENTIFIC_RESULT"
    )
    report: dict[str, object] = {
        "schema": "neuro-film.p307-cambridge-hdr-deghost-source-feasibility-result.v1",
        "experiment_id": "P307",
        "decision": decision,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "bindings": bindings,
        "metadata": metadata_facts,
        "archives": ordered_archive_facts,
        "fresh_categories": categories,
        "fresh_group_count": total_groups,
        "gates": gates,
        "transport": {
            "metadata_requests": len(metadata),
            "archive_range_requests": 2 * len(archives),
            "metadata_network_bytes": metadata_network_bytes,
            "archive_network_bytes": archive_network_bytes,
            "total_network_bytes": total_network_bytes,
        },
        "reads": {
            "part1_requests": 0,
            "local_header_reads": 0,
            "member_payload_reads": 0,
            "image_or_raw_reads": 0,
            "pixel_decodes": 0,
            "model_or_metric_reads": 0,
        },
    }
    report["scientific_identity"] = "sha256:" + _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve(), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
