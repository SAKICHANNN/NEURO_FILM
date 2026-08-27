"""Audit official EyefulTower metadata without downloading image payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P292Error(RuntimeError):
    """Raised when a frozen P292 source or execution boundary differs."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_binding(bindings: dict[str, object], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    body = path.read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256(body) == str(
        bindings[f"{prefix}_sha256"]
    )


def _request(
    url: str,
    *,
    method: str = "GET",
    maximum_bytes: int,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/vnd.github+json,text/plain,application/xml",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p292-source-audit/1.0",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(maximum_bytes + 1)
                if len(body) > maximum_bytes:
                    raise P292Error("official response exceeds frozen byte ceiling")
                if int(response.status) != 200:
                    raise P292Error("official response did not return HTTP 200")
                headers = {key.casefold(): value for key, value in response.headers.items()}
                result = {
                    "body": body,
                    "body_bytes": len(body),
                    "body_sha256": _sha256(body),
                    "content_length": (
                        int(headers.get("content-length", "0"))
                        if method == "HEAD"
                        else len(body)
                    ),
                    "etag": headers.get("etag"),
                    "final_url": response.geturl(),
                    "http_status": int(response.status),
                    "method": method,
                }
            return result
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt == 2:
                raise P292Error("official source transport failed after bounded retries") from error
            time.sleep(0.25 * (attempt + 1))
    raise P292Error("official source transport failed") from last_error


def _fetch_exact(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    result = _request(str(source["url"]), maximum_bytes=maximum_bytes)
    body = bytes(result["body"])
    if len(body) != int(source["bytes"]) or _sha256(body) != source["sha256"]:
        raise P292Error("official response identity differs from frozen source")
    return result


def _parse_listing(body: bytes) -> list[dict[str, object]]:
    root = ET.fromstring(body)
    namespace_match = re.match(r"\{([^}]+)\}", root.tag)
    namespace = {"s3": namespace_match.group(1)} if namespace_match else {}
    prefix = "s3:" if namespace else ""
    items: list[dict[str, object]] = []
    for item in root.findall(f"{prefix}Contents", namespace):
        key = item.findtext(f"{prefix}Key", default="", namespaces=namespace)
        size = item.findtext(f"{prefix}Size", default="", namespaces=namespace)
        etag = item.findtext(f"{prefix}ETag", default="", namespaces=namespace)
        if not key or not size or not etag:
            raise P292Error("S3 listing contains an incomplete object row")
        items.append({"etag": etag, "key": key, "size": int(size)})
    return items


def _parse_checksums(body: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in body.decode("utf-8", errors="strict").splitlines():
        match = re.fullmatch(r"([0-9a-f]{32})  ([^\r\n]+\.exr)", line)
        if match is None:
            raise P292Error("checksum file contains a malformed row")
        checksum, path = match.groups()
        if path in checksums:
            raise P292Error("checksum file contains a duplicate path")
        checksums[path] = checksum
    return checksums


def _extract_facts(
    config: dict[str, Any], responses: dict[str, dict[str, object]]
) -> dict[str, object]:
    commit = json.loads(bytes(responses["commit"]["body"]))
    tree = json.loads(bytes(responses["tree"]["body"]))
    readme = bytes(responses["readme"]["body"]).decode("utf-8", errors="strict")
    license_text = bytes(responses["license"]["body"]).decode(
        "utf-8", errors="strict"
    )
    lower = re.sub(r"\s+", " ", readme.casefold()).strip()
    repository = config["official_repository"]
    expected = config["expected"]
    paths = {str(item["path"]): item for item in tree["tree"]}

    official_identity = all(
        (
            commit["sha"] == repository["commit"],
            commit["tree"]["sha"] == repository["tree"],
            [item["sha"] for item in commit["parents"]] == repository["parents"],
            tree["sha"] == repository["tree"],
            not bool(tree["truncated"]),
            len(tree["tree"]) == int(expected["repository_tree_entries"]),
            paths["LICENSE"]["sha"] == expected["license_blob"],
            paths["README.md"]["sha"] == expected["readme_blob"],
        )
    )
    mit_rights = all(
        (
            "permission is hereby granted, free of charge" in license_text.casefold(),
            "relicensed all content under the mit license" in lower,
        )
    )
    source_description = all(
        phrase in lower
        for phrase in (
            "high dynamic range images merged from 9-photo raw exposure brackets",
            "color space: dci-p3 (linear)",
            "stored as exr images with uncompressed 32-bit floating-point numbers",
        )
    )
    jpeg_independent_truth = not (
        "jpeg images are white-balanced and tone-mapped versions of the hdr images"
        in lower
    )

    objects = _parse_listing(bytes(responses["listing"]["body"]))
    exrs = sorted(
        (item for item in objects if str(item["key"]).endswith(".exr")),
        key=lambda item: str(item["key"]),
    )
    prefix = str(expected["s3_prefix"])
    cameras: set[str] = set()
    suffixes: set[str] = set()
    relative_paths: list[str] = []
    for item in exrs:
        key = str(item["key"])
        if not key.startswith(prefix):
            raise P292Error("listed EXR escapes frozen prefix")
        relative = key[len(prefix) :]
        match = re.fullmatch(r"([^/]+)/\1_(DSC\d+)\.exr", relative)
        if match is None:
            raise P292Error("listed EXR does not match frozen camera naming")
        cameras.add(match.group(1))
        suffixes.add(match.group(2))
        relative_paths.append(relative)
    checksums = _parse_checksums(bytes(responses["checksums"]["body"]))
    checksum_coverage = sorted(checksums) == sorted(relative_paths)
    selected = exrs[0] if exrs else {}
    selected_relative = str(selected.get("key", ""))[len(prefix) :]
    inventory_exact = all(
        (
            len(objects) == int(expected["listing_objects"]),
            len(exrs) == int(expected["exr_objects"]),
            sum(int(item["size"]) for item in exrs) == int(expected["exr_total_bytes"]),
            sorted(cameras) == expected["camera_names"],
            sorted(suffixes) == expected["capture_suffixes"],
            len(exrs) == len(cameras) * len(suffixes),
        )
    )
    selected_listing_exact = all(
        (
            selected.get("key") == expected["selected_key"],
            selected.get("size") == int(expected["selected_size"]),
            selected.get("etag") == expected["selected_etag"],
            checksums.get(selected_relative) == expected["selected_md5"],
        )
    )
    head = responses["selected_head"]
    selected_head_exact = all(
        (
            head["method"] == "HEAD",
            head["http_status"] == 200,
            head["body_bytes"] == 0,
            head["content_length"] == int(expected["selected_size"]),
            head["etag"] == expected["selected_etag"],
            selected_listing_exact,
        )
    )
    capture_group_identities = (
        sorted(suffixes) == expected["capture_suffixes"]
        and all(
            sum(relative.endswith(f"_{suffix}.exr") for relative in relative_paths)
            == len(cameras)
            for suffix in suffixes
        )
    )
    return {
        "camera_count": len(cameras),
        "camera_names": sorted(cameras),
        "capture_group_identities": capture_group_identities,
        "capture_suffix_count": len(suffixes),
        "capture_suffixes": sorted(suffixes),
        "checksum_coverage": checksum_coverage,
        "checksum_rows": len(checksums),
        "exr_objects": len(exrs),
        "exr_total_bytes": sum(int(item["size"]) for item in exrs),
        "inventory_exact": inventory_exact,
        "jpeg_independent_truth": jpeg_independent_truth,
        "linear_dci_p3_float32_exr": source_description,
        "listing_objects": len(objects),
        "mit_all_content_rights": mit_rights,
        "nine_raw_bracket_observation": source_description,
        "official_identity": official_identity,
        "selected_head_exact": selected_head_exact,
        "selected_key": selected.get("key"),
        "selected_md5": checksums.get(selected_relative),
        "selected_size": selected.get("size"),
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    binding_gates = {
        "contract": _verify_binding(bindings, "contract"),
        "runner": _verify_binding(bindings, "runner"),
        "test": _verify_binding(bindings, "test"),
    }
    if not all(binding_gates.values()):
        raise P292Error("frozen local binding differs")

    maximum = int(config["sources"]["maximum_response_bytes_each"])
    source_keys = ("commit", "tree", "readme", "license", "listing", "checksums")
    traversal = tuple(reversed(source_keys)) if reverse else source_keys
    responses = {key: _fetch_exact(config["sources"][key], maximum) for key in traversal}
    selected_url = (
        "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/"
        + str(config["expected"]["selected_key"])
    )
    responses["selected_head"] = _request(
        selected_url, method="HEAD", maximum_bytes=0
    )
    total_bytes = sum(int(item["body_bytes"]) for item in responses.values())
    if total_bytes > int(config["sources"]["maximum_total_network_bytes"]):
        raise P292Error("formal network bytes exceed frozen total ceiling")

    facts = _extract_facts(config, responses)
    gates = {
        "capture_group_identities": facts["capture_group_identities"],
        "checksum_coverage": facts["checksum_coverage"],
        "complete_bounded_inventory": facts["inventory_exact"],
        "linear_dci_p3_float32_exr": facts["linear_dci_p3_float32_exr"],
        "mit_all_content_rights": facts["mit_all_content_rights"],
        "nine_raw_bracket_observation": facts["nine_raw_bracket_observation"],
        "official_identity": facts["official_identity"],
        "selected_object_head_exact": facts["selected_head_exact"],
        "zero_payload_and_pixel_reads": True,
    }
    decision = (
        "PASS_PRIVATE_EYEFULTOWER_DCI_P3_EXR_SOURCE_READINESS"
        if all(gates.values())
        else "FAIL_CLOSED_EYEFULTOWER_DCI_P3_EXR_SOURCE_READINESS"
    )
    transport = {
        key: {name: value for name, value in item.items() if name != "body"}
        for key, item in responses.items()
    }
    report = {
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "experiment_id": "P292",
        "exr_or_jpeg_payload_requests": 0,
        "facts": facts,
        "fitting_training_inference_render_score_runs": 0,
        "gates": gates,
        "network_body_bytes": total_bytes,
        "pixel_decodes_or_reads": 0,
        "repository_clone_or_checkout": 0,
        "schema": "neuro-film.p292-eyefultower-dci-p3-exr-source-readiness-result.v1",
        "transport": transport,
    }
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
