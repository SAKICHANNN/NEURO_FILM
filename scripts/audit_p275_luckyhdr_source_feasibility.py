#!/usr/bin/env python3
"""Audit exact LuckyHDR source objects without reading binary bodies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "neuro-film.p275-luckyhdr-source-feasibility.v1"


class P275Error(RuntimeError):
    """Raised when a frozen P275 source requirement differs."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request_json(url: str) -> tuple[dict[str, Any], int]:
    result = subprocess.run(
        [
            "curl.exe",
            "--http1.1",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--retry-all-errors",
            "-H",
            "User-Agent: Codex-NeuroFilm-P275",
            url,
        ],
        check=True,
        capture_output=True,
    )
    body = result.stdout
    return json.loads(body), len(body)


def _blob(api_root: str, binding: dict[str, Any]) -> tuple[bytes, int]:
    record, network_bytes = _request_json(f"{api_root}/git/blobs/{binding['git_blob']}")
    if (
        record.get("sha") != binding["git_blob"]
        or record.get("size") != binding["bytes"]
    ):
        raise P275Error(f"text blob identity differs: {binding['path']}")
    if record.get("encoding") != "base64":
        raise P275Error(f"text blob encoding differs: {binding['path']}")
    value = base64.b64decode(record["content"], validate=False)
    if len(value) != binding["bytes"]:
        raise P275Error(f"decoded text size differs: {binding['path']}")
    return value, network_bytes


def _tree_index(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    repo = config["repository"]
    commit, commit_bytes = _request_json(
        f"{repo['api_root']}/git/commits/{repo['commit']}"
    )
    if (
        commit.get("sha") != repo["commit"]
        or commit.get("tree", {}).get("sha") != repo["tree"]
    ):
        raise P275Error("repository commit or tree differs")
    tree, tree_bytes = _request_json(
        f"{repo['api_root']}/git/trees/{repo['tree']}?recursive=1"
    )
    if tree.get("sha") != repo["tree"] or tree.get("truncated"):
        raise P275Error("recursive tree is incomplete or differs")
    return {entry["path"]: entry for entry in tree["tree"]}, commit_bytes + tree_bytes


def execute(config_path: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P275Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_FORMAL_GIT_OBJECT_AUDIT":
        raise P275Error("P275 is not frozen")

    tree, network_bytes = _tree_index(config)
    all_bindings = {**config["text_objects"], **config["binary_objects"]}
    names = list(all_bindings)
    if order == "reverse":
        names.reverse()
    inventory: dict[str, dict[str, Any]] = {}
    text: dict[str, str] = {}
    for name in names:
        binding = all_bindings[name]
        observed = tree.get(binding["path"])
        if observed is None or observed.get("type") != "blob":
            raise P275Error(f"bound object is absent or not a blob: {binding['path']}")
        if (
            observed.get("sha") != binding["git_blob"]
            or observed.get("size") != binding["bytes"]
        ):
            raise P275Error(f"bound object differs: {binding['path']}")
        inventory[name] = {
            "bytes": observed["size"],
            "git_blob": observed["sha"],
            "path": binding["path"],
        }
        if name in config["text_objects"]:
            value, body_bytes = _blob(config["repository"]["api_root"], binding)
            network_bytes += body_bytes
            text[name] = value.decode("utf-8")

    dng_bytes = sum(inventory[name]["bytes"] for name in ("short", "mid", "long"))
    demo = text["demo_readme"]
    inference = text["inference"]
    datasets = text["datasets"]
    rights_gap = bool(
        "MIT License" in text["license"]
        and "We do not redistribute capture data" in datasets
        and "ship as full 12 MP RAW" in demo
        and "License: see upstream" in datasets
    )
    gates = {
        "repository_commit_and_tree_exact": True,
        "root_mit_licence_present": "MIT License" in text["license"],
        "checkpoint_regular_git_blob_and_bounded": inventory["checkpoint"]["bytes"]
        <= config["limits"]["checkpoint_bytes_max"],
        "three_ordered_dng_blobs_and_bounded": dng_bytes
        <= config["limits"]["three_dng_bytes_max"],
        "demo_roles_and_capture_metadata_explicit": all(
            token in demo
            for token in (
                "frame_short.dng",
                "frame_mid.dng",
                "frame_long.dng",
                "ExposureTime",
                "ISO",
                "short",
                "mid",
                "long",
            )
        ),
        "inference_uses_shutter_times_iso": all(
            token in inference for token in ("ExposureTime", "ISO", "exposure")
        ),
        "model_and_checkpoint_fixed_without_load": bool(
            text["model"] and inventory["checkpoint"]
        ),
        "binary_body_reads_zero": config["limits"]["binary_body_reads"] == 0,
        "pixel_model_target_and_score_reads_zero": all(
            config["limits"][key] == 0
            for key in (
                "pixel_decodes",
                "model_loads",
                "inference_runs",
                "target_reads",
                "score_runs",
            )
        ),
        "rights_and_upstream_dependency_gap_recorded": rights_gap,
        "product_rights_remain_false": not config["frozen_interpretation"][
            "product_rights_clear"
        ],
        "candidate3_not_consumed": not config["frozen_interpretation"][
            "candidate3_consumed"
        ],
    }
    status = (
        "PASS_PRIVATE_BOUNDED_SOURCE_FEASIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_SOURCE_FEASIBILITY"
    )
    scientific = {
        "binary_body_reads": 0,
        "candidate3_consumed": False,
        "checkpoint_bytes": inventory["checkpoint"]["bytes"],
        "dng_bytes": dng_bytes,
        "dng_count": 3,
        "ground_truth_present": False,
        "inference_runs": 0,
        "model_loads": 0,
        "pixel_decodes": 0,
        "private_runtime_d0_eligible": status.startswith("PASS_"),
        "product_rights_clear": False,
        "rights_gap": config["frozen_interpretation"]["rights_gap"],
        "score_runs": 0,
        "status": status,
        "target_reads": 0,
    }
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": "P275",
        "order": order,
        "config_sha256": _sha256(config_bytes),
        "repository": {
            "commit": config["repository"]["commit"],
            "tree": config["repository"]["tree"],
        },
        "inventory": dict(sorted(inventory.items())),
        "gates": gates,
        "scientific": scientific,
        "scientific_payload_sha256": _sha256(_canonical(scientific)),
        "network_response_bytes": network_bytes,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = execute(args.config, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))


if __name__ == "__main__":
    main()
