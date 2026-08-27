"""Audit the official RawHDR source without requesting dataset payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P304Error(RuntimeError):
    """Raised when the frozen P304 source boundary is violated."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _verify_binding(bindings: dict[str, object], prefix: str) -> bool:
    body = (ROOT / str(bindings[f"{prefix}_path"])).read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256(body) == str(
        bindings[f"{prefix}_sha256"]
    )


def _fetch(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    request = urllib.request.Request(
        str(source["url"]),
        headers={
            "Accept": "application/json,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p304-source-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(maximum_bytes + 1)
        status = int(response.status)
        final_url = response.geturl()
    if len(body) > maximum_bytes:
        raise P304Error("official response exceeds frozen byte ceiling")
    if status != 200:
        raise P304Error("official response did not return HTTP 200")
    if len(body) != int(source["bytes"]) or _sha256(body) != str(source["sha256"]):
        raise P304Error("official response identity differs from frozen source")
    return {
        "body": body,
        "body_bytes": len(body),
        "body_sha256": _sha256(body),
        "final_url": final_url,
        "http_status": status,
    }


def _extract_facts(
    config: dict[str, Any], responses: dict[str, dict[str, object]]
) -> dict[str, object]:
    commit = json.loads(bytes(responses["commit"]["body"]))
    tree = json.loads(bytes(responses["tree"]["body"]))
    readme = bytes(responses["readme"]["body"]).decode("utf-8")
    licence = bytes(responses["license"]["body"]).decode("utf-8")
    lower = readme.casefold()
    paths = sorted(str(item["path"]) for item in tree["tree"])
    expected = config["expected"]
    repo = config["official_repository"]

    physical_phrases = (
        "real paired raw-to-hdr dataset",
        "canon 5d mark iv",
        "-3ev, 0ev, and +3ev",
        "0ev raw images are served as input images",
        "ground truth images are fused by hdr merging method",
        "324 pairs of raw/hdr images",
    )
    payload_urls = re.findall(r"https?://[^)\s]+", readme)
    dataset_urls = [
        url
        for url in payload_urls
        if "1drv.ms/f/" in url.casefold() or "pan.baidu.com/s/" in url.casefold()
    ]
    exact_inventory_paths = [
        path
        for path in paths
        if any(token in path.casefold() for token in ("manifest", "checksum", "sha256"))
    ]
    explicit_data_rights = bool(
        re.search(
            r"dataset.{0,120}(?:cc\s*by|creative commons|mit license|commercial use)",
            lower,
        )
    )
    split_names_present = "mat_train" in lower and "mat_test" in lower
    public_scene_manifest = bool(
        re.search(r"(?:scene|split).{0,40}(?:manifest|checksum|sha-?256)", lower)
    ) or any("split" in path.casefold() for path in paths)
    official_identity = (
        commit["sha"] == repo["commit"]
        and commit["tree"]["sha"] == repo["tree"]
        and tree["sha"] == repo["tree"]
        and len(paths) == int(expected["tree_entry_count"])
        and expected["repository_title"].casefold() in lower
    )
    return {
        "anonymous_payload_locator": bool(dataset_urls),
        "code_license_is_mit": licence.startswith("MIT License"),
        "commercial_compatible_data_rights": explicit_data_rights,
        "dataset_locator_count": len(dataset_urls),
        "dataset_locators": dataset_urls,
        "exact_inventory_paths": exact_inventory_paths,
        "group_isolation": split_names_present and public_scene_manifest,
        "materially_distinct_physical_observation": all(
            phrase in lower for phrase in physical_phrases
        ),
        "official_identity": official_identity,
        "public_exact_inventory": bool(exact_inventory_paths),
        "repository_path_count": len(paths),
        "split_directory_names_only": split_names_present,
    }


def execute(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding_gates = {
        name: _verify_binding(config["bindings"], name)
        for name in ("contract", "runner", "test")
    }
    if not all(binding_gates.values()):
        raise P304Error("frozen local binding differs")
    maximum = int(config["sources"]["maximum_response_bytes_each"])
    responses = {
        key: _fetch(config["sources"][key], maximum)
        for key in ("commit", "tree", "readme", "license")
    }
    network_bytes = sum(int(item["body_bytes"]) for item in responses.values())
    if network_bytes > int(config["sources"]["maximum_total_network_bytes"]):
        raise P304Error("formal network bytes exceed frozen total ceiling")
    facts = _extract_facts(config, responses)
    gates = {
        "anonymous_payload_locator": facts["anonymous_payload_locator"],
        "commercial_compatible_code_rights": facts["code_license_is_mit"],
        "commercial_compatible_data_rights": facts[
            "commercial_compatible_data_rights"
        ],
        "group_isolation": facts["group_isolation"],
        "materially_distinct_physical_observation": facts[
            "materially_distinct_physical_observation"
        ],
        "official_identity": facts["official_identity"],
        "public_exact_inventory": facts["public_exact_inventory"],
    }
    decision = (
        "PASS_PRIVATE_RAWHDR_PAIRED_RAW_SOURCE_READINESS"
        if all(gates.values())
        else "FAIL_CLOSED_RAWHDR_SOURCE_RIGHTS_OR_MANIFEST_GAP_NOT_SCIENTIFIC_RESULT"
    )
    transport = {
        key: {name: value for name, value in item.items() if name != "body"}
        for key, item in responses.items()
    }
    report: dict[str, object] = {
        "archive_or_dataset_requests": 0,
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "experiment_id": "P304",
        "facts": facts,
        "gates": gates,
        "image_or_thumbnail_requests": 0,
        "model_or_checkpoint_requests": 0,
        "network_bytes": network_bytes,
        "pixel_decodes": 0,
        "schema": "neuro-film.p304-rawhdr-paired-raw-source-readiness-result.v1",
        "transport": transport,
    }
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
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
