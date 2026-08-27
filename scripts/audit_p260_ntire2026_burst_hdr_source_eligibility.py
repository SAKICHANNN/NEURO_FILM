"""Audit official NTIRE 2026 burst-HDR sources without payload access."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P260Error(RuntimeError):
    """Raised when the frozen P260 source boundary is violated."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _plain_html(body: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(body.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _verify_binding(bindings: dict[str, object], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    body = path.read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256(body) == str(
        bindings[f"{prefix}_sha256"]
    )


def _fetch(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    request = urllib.request.Request(
        str(source["url"]),
        headers={
            "Accept": "application/json,text/html,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p260-source-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(maximum_bytes + 1)
        if len(body) > maximum_bytes:
            raise P260Error("official response exceeds frozen byte ceiling")
        if int(response.status) != 200:
            raise P260Error("official response did not return HTTP 200")
        final_url = response.geturl()
        http_status = int(response.status)
    if len(body) != int(source["bytes"]) or _sha256(body) != source["sha256"]:
        raise P260Error("official response identity differs from frozen source")
    return {
        "body": body,
        "body_bytes": len(body),
        "body_sha256": _sha256(body),
        "final_url": final_url,
        "http_status": http_status,
    }


def _extract_facts(
    config: dict[str, Any], responses: dict[str, dict[str, object]]
) -> dict[str, object]:
    commit = json.loads(bytes(responses["commit"]["body"]))
    tree = json.loads(bytes(responses["tree"]["body"]))
    readme = bytes(responses["readme"]["body"]).decode("utf-8")
    cvf = _plain_html(bytes(responses["cvf"]["body"]))
    lower = readme.casefold()
    cvf_lower = cvf.casefold()
    paths = sorted(str(item["path"]) for item in tree["tree"])
    lower_paths = [path.casefold() for path in paths]
    licence_paths = [
        path
        for path in paths
        if Path(path).name.casefold()
        in {"license", "license.md", "license.txt", "copying"}
    ]
    dataset_payload_paths = [
        path
        for path in paths
        if re.search(r"scene[-_].*\.(?:tif|tiff|dng|raw)$", path, re.IGNORECASE)
    ]
    exact_inventory_paths = [
        path
        for path in paths
        if any(token in path.casefold() for token in ("manifest", "checksum", "sha256"))
    ]
    repo = config["official_repository"]
    expected = config["expected"]
    training_restriction = (
        "all training datasets and their images must not be shared with others or used for other purposes"
        in lower
    )
    public_payload_locator = bool(dataset_payload_paths) or bool(
        re.search(r"https?://\S+\.(?:zip|tar|7z|tif|tiff|dng)(?:\s|\)|$)", readme)
    )
    explicit_code_license = bool(licence_paths)
    explicit_commercial_data_rights = bool(
        re.search(r"(?:cc\s*by\s*4\.0|apache-2\.0|mit license)", lower)
    ) and not training_restriction
    physical_observation = all(
        phrase in lower
        for phrase in (
            "each scene consists of nine input raw frames",
            "scene-xxx-gt.tif",
            "reference frame",
            "aligned with gt image",
            "short exposure time",
            "middle exposure time",
            "high exposure time",
        )
    )
    group_isolation = all(
        phrase in lower
        for phrase in (
            "training, validation, and test datasets",
            "scene-xxx-in-0.tif",
            "scene-xxx-in-8.tif",
            "scene-xxx-gt.tif",
        )
    )
    official_identity = (
        commit["sha"] == repo["commit"]
        and commit["tree"]["sha"] == repo["tree"]
        and tree["sha"] == repo["tree"]
        and len(tree["tree"]) == int(expected["tree_entry_count"])
        and expected["paper_title"].casefold() in cvf_lower
        and expected["code_url"].casefold() in cvf_lower
    )
    return {
        "anonymous_public_payload": public_payload_locator,
        "code_license_paths": licence_paths,
        "commercial_compatible_code_rights": explicit_code_license,
        "commercial_compatible_data_rights": explicit_commercial_data_rights,
        "dataset_payload_paths": dataset_payload_paths,
        "documented_downloaded_training_scene_count_present": "200 scenes" in lower,
        "exact_inventory_paths": exact_inventory_paths,
        "group_isolation": group_isolation,
        "new_physical_observation": physical_observation,
        "official_identity": official_identity,
        "public_exact_inventory": bool(exact_inventory_paths),
        "repository_path_count": len(lower_paths),
        "training_scene_count_present": "300 scenes" in lower,
        "training_use_restriction_present": training_restriction,
        "validation_and_test_scene_counts_present": lower.count("20 scenes") >= 2,
    }


def execute(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    binding_gates = {
        "contract": _verify_binding(bindings, "contract"),
        "runner": _verify_binding(bindings, "runner"),
        "test": _verify_binding(bindings, "test"),
    }
    if not all(binding_gates.values()):
        raise P260Error("frozen local binding differs")
    maximum = int(config["sources"]["maximum_response_bytes_each"])
    responses = {
        key: _fetch(config["sources"][key], maximum)
        for key in ("commit", "tree", "readme", "cvf")
    }
    total_bytes = sum(int(item["body_bytes"]) for item in responses.values())
    if total_bytes > int(config["sources"]["maximum_total_network_bytes"]):
        raise P260Error("formal network bytes exceed frozen total ceiling")
    facts = _extract_facts(config, responses)
    gates = {
        "anonymous_public_payload": facts["anonymous_public_payload"],
        "commercial_compatible_code_rights": facts[
            "commercial_compatible_code_rights"
        ],
        "commercial_compatible_data_rights": facts[
            "commercial_compatible_data_rights"
        ],
        "group_isolation": facts["group_isolation"],
        "new_physical_observation": facts["new_physical_observation"],
        "official_identity": facts["official_identity"],
        "public_exact_inventory": facts["public_exact_inventory"],
    }
    decision = (
        "PASS_PRIVATE_NTIRE2026_BURST_HDR_SOURCE_ELIGIBILITY"
        if all(gates.values())
        else "NOT_READY_NTIRE2026_BURST_HDR_RIGHTS_OR_PAYLOAD_GAP_NOT_SCIENTIFIC_RESULT"
    )
    transport = {
        key: {name: value for name, value in item.items() if name != "body"}
        for key, item in responses.items()
    }
    report = {
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "dataset_file_requests": 0,
        "decision": decision,
        "experiment_id": "P260",
        "facts": facts,
        "gates": gates,
        "image_or_thumbnail_requests": 0,
        "model_or_checkpoint_requests": 0,
        "network_bytes": total_bytes,
        "schema": "neuro-film.p260-ntire2026-burst-hdr-source-eligibility-result.v1",
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
    args = parser.parse_args()
    report = execute(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
