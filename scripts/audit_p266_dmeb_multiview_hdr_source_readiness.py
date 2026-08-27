"""Audit exact official DMEB sources without requesting datasets or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P266Error(RuntimeError):
    """Raised when a frozen P266 source or execution boundary differs."""


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


def _fetch(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    request = urllib.request.Request(
        str(source["url"]),
        headers={
            "Accept": "application/json,text/html,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p266-source-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(maximum_bytes + 1)
        if len(body) > maximum_bytes:
            raise P266Error("official response exceeds frozen byte ceiling")
        if int(response.status) != 200:
            raise P266Error("official response did not return HTTP 200")
        final_url = response.geturl()
        http_status = int(response.status)
    if len(body) != int(source["bytes"]) or _sha256(body) != source["sha256"]:
        raise P266Error("official response identity differs from frozen source")
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
    texts = {
        key: bytes(responses[key]["body"]).decode("utf-8", errors="strict")
        for key in (
            "citation",
            "license",
            "dataset_card",
            "benchmark",
            "code_readme",
            "checkpoint",
            "project",
        )
    }
    lower = {
        key: re.sub(r"\s+", " ", value.casefold()).strip()
        for key, value in texts.items()
    }
    paths = sorted(str(item["path"]) for item in tree["tree"])
    expected = config["expected"]
    repository = config["official_repository"]

    official_identity = all(
        (
            commit["sha"] == repository["commit"],
            commit["tree"]["sha"] == repository["tree"],
            tree["sha"] == repository["tree"],
            len(paths) == int(expected["tree_entry_count"]),
            expected["citation_title"].casefold() in lower["citation"],
            expected["project_title"].casefold() in lower["project"],
        )
    )
    observation_phrases = (
        "multi-view ldr inputs",
        "hdr ground truth",
        "intrinsics / extrinsics",
        "exposure / gain metadata",
        "valid masks",
    )
    new_physical_observation = (
        all(phrase in lower["dataset_card"] for phrase in observation_phrases)
        and all(
            phrase in lower["benchmark"]
            for phrase in (
                "varying exposures",
                "linear hdr radiance",
                "reference-camera field of view",
            )
        )
        and all(
            phrase in lower["dataset_card"]
            for phrase in ("synchronized, calibrated", "depth")
        )
    )
    code_rights = all(
        phrase in lower["license"]
        for phrase in (
            "2. code (code/)",
            "mit license",
            "permission is hereby granted",
            "sell copies",
        )
    )
    dataset_noncommercial = all(
        phrase in lower["license"]
        for phrase in (
            "1. datasets",
            "creative commons attribution-noncommercial 4.0",
            "non-commercial research",
        )
    )
    dataset_rights = (
        "creative commons attribution 4.0 international" in lower["license"]
        and not dataset_noncommercial
    )

    count_placeholders = lower["dataset_card"].count("[n]")
    public_exact_counts = (
        "todo final" not in lower["dataset_card"] and count_placeholders == 0
    )
    inventory_paths = [
        path
        for path in paths
        if any(
            token in Path(path).name.casefold()
            for token in ("manifest", "checksum", "sha256", "md5")
        )
    ]
    public_exact_inventory = bool(inventory_paths) and public_exact_counts
    checkpoint_placeholders = sum(
        lower["checkpoint"].count(token) for token in ("[drive-link]", "[md5]")
    )
    exact_checkpoint = checkpoint_placeholders == 0 and bool(
        re.search(r"https?://\S+", texts["checkpoint"])
    )
    drive_locator_count = len(
        set(
            re.findall(
                r"https://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+",
                texts["dataset_card"] + texts["project"],
            )
        )
    )
    group_isolation = all(
        phrase in lower["dataset_card"]
        for phrase in (
            "robot test holdout",
            "robot train/val",
            "sessions without pseudo-gt are excluded",
        )
    )
    return {
        "checkpoint_placeholders": checkpoint_placeholders,
        "code_commercial_compatible": code_rights,
        "dataset_card_count_placeholders": count_placeholders,
        "dataset_commercial_compatible": dataset_rights,
        "dataset_noncommercial_terms_present": dataset_noncommercial,
        "drive_folder_locator_count": drive_locator_count,
        "exact_reference_checkpoint": exact_checkpoint,
        "group_isolation": group_isolation,
        "inventory_paths": inventory_paths,
        "new_physical_observation": new_physical_observation,
        "official_identity": official_identity,
        "public_exact_counts": public_exact_counts,
        "public_exact_inventory": public_exact_inventory,
        "repository_path_count": len(paths),
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
        raise P266Error("frozen local binding differs")
    maximum = int(config["sources"]["maximum_response_bytes_each"])
    source_keys = (
        "commit",
        "tree",
        "citation",
        "license",
        "dataset_card",
        "benchmark",
        "code_readme",
        "checkpoint",
        "project",
    )
    traversal = tuple(reversed(source_keys)) if reverse else source_keys
    responses = {key: _fetch(config["sources"][key], maximum) for key in traversal}
    total_bytes = sum(int(item["body_bytes"]) for item in responses.values())
    if total_bytes > int(config["sources"]["maximum_total_network_bytes"]):
        raise P266Error("formal network bytes exceed frozen total ceiling")
    facts = _extract_facts(config, responses)
    gates = {
        "commercial_compatible_code_rights": facts["code_commercial_compatible"],
        "commercial_compatible_dataset_rights": facts["dataset_commercial_compatible"],
        "exact_public_archive_inventory": facts["public_exact_inventory"],
        "exact_public_scene_frame_counts": facts["public_exact_counts"],
        "exact_reference_checkpoint": facts["exact_reference_checkpoint"],
        "group_isolation": facts["group_isolation"],
        "new_physical_observation": facts["new_physical_observation"],
        "official_identity": facts["official_identity"],
    }
    decision = (
        "PASS_PRIVATE_DMEB_MULTIVIEW_HDR_SOURCE_READINESS"
        if all(gates.values())
        else "NOT_READY_DMEB_RIGHTS_INVENTORY_OR_CHECKPOINT_GAP_NOT_SCIENTIFIC_RESULT"
    )
    transport = {
        key: {name: value for name, value in item.items() if name != "body"}
        for key, item in responses.items()
    }
    report = {
        "archive_or_member_requests": 0,
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "checkpoint_or_model_requests": 0,
        "claim_ceiling": config["claim_ceiling"],
        "dataset_drive_requests": 0,
        "decision": decision,
        "experiment_id": "P266",
        "facts": facts,
        "gates": gates,
        "network_bytes": total_bytes,
        "pixel_reads": 0,
        "schema": "neuro-film.p266-dmeb-multiview-hdr-source-readiness-result.v1",
        "training_or_inference_runs": 0,
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
