"""Audit WildRelight metadata without requesting EXR or DNG payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P286Error(RuntimeError):
    """Raised when a frozen P286 source or execution boundary differs."""


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


def _request_bytes(url: str, maximum_bytes: int) -> tuple[bytes, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p286-source-audit/1.0",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(maximum_bytes + 1)
                if len(body) > maximum_bytes:
                    raise P286Error("official response exceeds frozen byte ceiling")
                if int(response.status) != 200:
                    raise P286Error("official response did not return HTTP 200")
                link = response.headers.get("Link")
            return body, link
        except (ConnectionError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt == 2:
                raise P286Error("official source transport failed after retries") from error
            time.sleep(0.25 * (attempt + 1))
    raise P286Error("official source transport failed") from last_error


def _fetch_exact(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    body, _ = _request_bytes(str(source["url"]), maximum_bytes)
    if len(body) != int(source["bytes"]) or _sha256(body) != source["sha256"]:
        raise P286Error("official text identity differs from frozen source")
    return {"body": body, "body_bytes": len(body), "body_sha256": _sha256(body)}


def _next_link(link: str | None) -> str | None:
    if not link:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return match.group(1) if match else None


def _fetch_manifest(config: dict[str, Any]) -> dict[str, object]:
    tree_config = config["sources"]["tree"]
    url: str | None = str(tree_config["url"])
    revision = str(config["official_dataset"]["revision"])
    maximum_bytes = int(tree_config["maximum_response_bytes_each"])
    maximum_pages = int(tree_config["maximum_pages"])
    items: list[dict[str, object]] = []
    page_bytes: list[int] = []
    while url is not None:
        if len(page_bytes) >= maximum_pages or revision not in url:
            raise P286Error("manifest pagination exceeded frozen boundary")
        body, link = _request_bytes(url, maximum_bytes)
        page = json.loads(body)
        if not isinstance(page, list):
            raise P286Error("manifest page is not a list")
        items.extend(page)
        page_bytes.append(len(body))
        url = _next_link(link)
    return {
        "items": items,
        "network_bytes": sum(page_bytes),
        "page_bytes": page_bytes,
        "page_count": len(page_bytes),
    }


def _manifest_facts(config: dict[str, Any], items: list[dict[str, object]]) -> dict[str, object]:
    paths = [str(item["path"]) for item in items]
    if len(paths) != len(set(paths)):
        raise P286Error("manifest contains duplicate paths")
    files = sorted(
        (item for item in items if item["type"] == "file"),
        key=lambda item: str(item["path"]),
    )
    directories = [item for item in items if item["type"] == "directory"]
    manifest_lines: list[str] = []
    lfs_count = 0
    missing_lfs_identity = 0
    for item in files:
        lfs = item.get("lfs")
        lfs_sha = ""
        if lfs is not None:
            lfs_count += 1
            lfs_sha = str(lfs.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", lfs_sha):
                missing_lfs_identity += 1
        manifest_lines.append(
            f"{item['path']}|{item['size']}|{item['oid']}|{lfs_sha}"
        )
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    file_paths = [str(item["path"]) for item in files]
    variants = sorted(
        {
            path.split("/", 1)[0]
            for path in file_paths
            if path.startswith(("ori-", "small-"))
        }
    )
    scenes = sorted(
        {
            match.group(1)
            for path in file_paths
            if (
                match := re.match(
                    r"^(?:ori|small)-(?:aligned|unaligned)/([^/]+)/", path
                )
            )
        }
    )
    photo_exr = [path for path in file_paths if re.search(r"/photo/time\d+_hdr\.exr$", path)]
    envmap_exr = [
        path for path in file_paths if re.search(r"/envmap/time\d+_envmap\.exr$", path)
    ]
    metadata = [path for path in file_paths if path.endswith("/meta.json")]
    dng_paths = [path for path in file_paths if path.casefold().endswith(".dng")]
    total_bytes = sum(int(item["size"]) for item in files)
    expected = config["expected"]
    exact_inventory = all(
        (
            len(items) == int(expected["entry_count"]),
            len(files) == int(expected["file_count"]),
            len(directories) == int(expected["directory_count"]),
            total_bytes == int(expected["total_file_bytes"]),
            lfs_count == int(expected["lfs_file_count"]),
            missing_lfs_identity == 0,
            len(manifest) == int(expected["manifest_bytes"]),
            _sha256(manifest) == expected["manifest_sha256"],
        )
    )
    roles_complete = all(
        (
            variants == expected["variants"],
            len(scenes) == int(expected["scene_count"]),
            len(photo_exr) == int(expected["photo_exr_count"]),
            len(envmap_exr) == int(expected["envmap_exr_count"]),
            len(metadata) == int(expected["metadata_count"]),
        )
    )
    return {
        "directory_count": len(directories),
        "dng_count": len(dng_paths),
        "entry_count": len(items),
        "envmap_exr_count": len(envmap_exr),
        "exact_inventory": exact_inventory,
        "file_count": len(files),
        "group_isolation": len(scenes) == int(expected["scene_count"]),
        "lfs_file_count": lfs_count,
        "manifest_bytes": len(manifest),
        "manifest_sha256": _sha256(manifest),
        "metadata_count": len(metadata),
        "missing_lfs_identity": missing_lfs_identity,
        "photo_exr_count": len(photo_exr),
        "roles_complete": roles_complete,
        "scene_count": len(scenes),
        "scenes": scenes,
        "total_file_bytes": total_bytes,
        "variants": variants,
    }


def _source_facts(
    config: dict[str, Any], api_body: bytes, readme_body: bytes, meta_body: bytes
) -> dict[str, object]:
    api = json.loads(api_body)
    readme = readme_body.decode("utf-8", errors="strict")
    meta = json.loads(meta_body)
    lower = re.sub(r"\s+", " ", readme.casefold()).strip()
    dataset = config["official_dataset"]
    official_identity = all(
        (
            api["id"] == dataset["id"],
            api["sha"] == dataset["revision"],
            not bool(api["private"]),
            not bool(api["gated"]),
            not bool(api["disabled"]),
            "wildrelight" in lower,
            config["expected"]["arxiv_id"] in readme,
        )
    )
    rights = all(
        phrase in lower
        for phrase in (
            "license: cc-by-4.0",
            "creative commons attribution 4.0 international",
            "cc by 4.0",
        )
    )
    required_meta_keys = {
        "scene",
        "ref_shutter",
        "hdr_method",
        "photos",
        "envmaps",
        "alignment_method",
    }
    meta_structure = required_meta_keys.issubset(meta)
    photo_times = [item["time"] for item in meta["photos"]]
    envmap_times = [item["time"] for item in meta["envmaps"]]
    photo_fields = all(
        {
            "time",
            "shooting_time",
            "iso",
            "source_files",
            "exposure_times",
            "ref_shutter",
        }.issubset(item)
        for item in meta["photos"]
    )
    envmap_fields = all(
        {
            "time",
            "shooting_time",
            "iso",
            "exposure_times",
            "ref_shutter",
        }.issubset(item)
        for item in meta["envmaps"]
    )
    metadata_roles = all(
        (
            meta_structure,
            meta["scene"] == config["expected"]["metadata_witness_scene"],
            photo_fields,
            envmap_fields,
            photo_times == envmap_times,
            len(photo_times) >= 3,
            any(len(item["source_files"]) >= 2 for item in meta["photos"]),
        )
    )
    return {
        "alignment_method": meta["alignment_method"],
        "commercial_compatible_dataset_rights": rights,
        "metadata_roles": metadata_roles,
        "metadata_time_count": len(photo_times),
        "official_identity": official_identity,
        "public_ungated": not bool(api["private"]) and not bool(api["gated"]),
        "raw_dng_names_present_in_metadata": any(
            name.casefold().endswith(".dng")
            for item in meta["photos"]
            for name in item["source_files"]
        ),
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
        raise P286Error("frozen local binding differs")

    maximum = int(config["sources"]["maximum_text_response_bytes_each"])
    source_order = ("api", "readme", "metadata")
    traversal = tuple(reversed(source_order)) if reverse else source_order
    responses: dict[str, dict[str, object]] = {}
    manifest: dict[str, object] | None = None
    if reverse:
        manifest = _fetch_manifest(config)
    for key in traversal:
        source = config["sources"][key]
        if key == "api":
            body, _ = _request_bytes(str(source["url"]), maximum)
            responses[key] = {"body": body, "body_bytes": len(body)}
        else:
            responses[key] = _fetch_exact(source, maximum)
    if manifest is None:
        manifest = _fetch_manifest(config)

    manifest_facts = _manifest_facts(config, manifest["items"])
    source_facts = _source_facts(
        config,
        bytes(responses["api"]["body"]),
        bytes(responses["readme"]["body"]),
        bytes(responses["metadata"]["body"]),
    )
    payload_bounded = int(manifest_facts["total_file_bytes"]) <= int(
        config["gates"]["maximum_public_payload_bytes"]
    )
    gates = {
        "commercial_compatible_dataset_rights": source_facts[
            "commercial_compatible_dataset_rights"
        ],
        "exact_public_inventory": manifest_facts["exact_inventory"],
        "group_isolation": manifest_facts["group_isolation"],
        "metadata_capture_roles": source_facts["metadata_roles"],
        "official_identity": source_facts["official_identity"],
        "paired_hdr_roles": manifest_facts["roles_complete"],
        "payload_bounded": payload_bounded,
        "public_ungated": source_facts["public_ungated"],
    }
    decision = (
        "PASS_PRIVATE_WILDRELIGHT_PAIRED_HDR_SOURCE_FEASIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_WILDRELIGHT_PAIRED_HDR_SOURCE_FEASIBILITY"
    )
    network_bytes = int(manifest["network_bytes"]) + sum(
        int(item["body_bytes"]) for item in responses.values()
    )
    report = {
        "archive_or_payload_requests": 0,
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "dng_payload_requests": 0,
        "experiment_id": "P286",
        "gates": gates,
        "manifest": manifest_facts,
        "manifest_network_bytes": manifest["network_bytes"],
        "manifest_page_bytes": manifest["page_bytes"],
        "manifest_page_count": manifest["page_count"],
        "model_code_training_inference_runs": 0,
        "network_bytes": network_bytes,
        "pixel_reads": 0,
        "schema": "neuro-film.p286-wildrelight-paired-hdr-source-feasibility-result.v1",
        "source": source_facts,
        "text_source_bytes": {
            key: int(item["body_bytes"]) for key, item in responses.items()
        },
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
