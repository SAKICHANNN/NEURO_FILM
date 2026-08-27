#!/usr/bin/env python3
"""Formal zero-pixel P301 Mono-HDR-3D source-readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "neuro-film-p301-source-audit"
SCHEMA = "neuro_film.p301_mono_hdr_3d_real_exposure_source_result.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _request(url: str, *, method: str = "GET") -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.status), response.read()


def _classify(readme: str, license_text: str, paths: list[str]) -> dict[str, bool]:
    lowered = readme.casefold()
    material = all(
        token in lowered
        for token in (
            "4 real scenes",
            "35 different poses",
            "5 different exposure time",
        )
    )
    dataset_context = lowered[lowered.find("download and organize the dataset") :]
    rights_explicit = (
        any(
            token in dataset_context
            for token in ("dataset license", "dataset is licensed", "data is licensed")
        )
        and "apache" in dataset_context
    )
    manifest_names = {
        path.casefold()
        for path in paths
        if any(token in path.casefold() for token in ("manifest", "checksum", "sha256"))
    }
    exact_manifest = bool(manifest_names) and any(
        token in lowered for token in ("sha-256", "sha256", "checksum")
    )
    declared_real_names = set(re.findall(r"real/([a-z0-9_-]+)", lowered))
    group_ready = len(declared_real_names) >= 4
    return {
        "material_capture_observation_explicit": material,
        "external_dataset_rights_explicit": rights_explicit,
        "exact_asset_manifest_with_sizes_and_checksums": exact_manifest,
        "group_identities_ready": group_ready,
        "apache_license_present": "apache license" in license_text.casefold(),
    }


def run(config_path: Path, output_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository = config["repository"]
    commit = repository["commit"]
    names = ["README.md", "LICENSE"]
    if reverse:
        names.reverse()
    bodies: dict[str, bytes] = {}
    statuses: dict[str, int] = {}
    for name in names:
        status, body = _request(
            f"https://raw.githubusercontent.com/prinasi/Mono-HDR-3D/{commit}/{name}"
        )
        statuses[name] = status
        bodies[name] = body
    commit_status, commit_body = _request(
        f"https://api.github.com/repos/prinasi/Mono-HDR-3D/git/commits/{commit}"
    )
    tree_status, tree_body = _request(
        f"https://api.github.com/repos/prinasi/Mono-HDR-3D/git/trees/{repository['tree']}?recursive=1"
    )
    drive_status, drive_body = _request(
        "https://drive.google.com/drive/folders/"
        + config["external_dataset"]["folder_id"]
        + "?usp=sharing",
        method="HEAD",
    )
    commit_json = json.loads(commit_body)
    tree_json = json.loads(tree_body)
    paths = sorted(str(item["path"]) for item in tree_json["tree"])
    readme = bodies["README.md"].decode("utf-8")
    license_text = bodies["LICENSE"].decode("utf-8")
    facts = _classify(readme, license_text, paths)

    identities_exact = all(
        (
            statuses == {"LICENSE": 200, "README.md": 200},
            commit_status == 200,
            tree_status == 200,
            drive_status == 200,
            drive_body == b"",
            commit_json["sha"] == commit,
            commit_json["tree"]["sha"] == repository["tree"],
            tree_json["sha"] == repository["tree"],
            len(bodies["README.md"]) == repository["readme_bytes"],
            _sha256(bodies["README.md"]) == repository["readme_sha256"],
            len(bodies["LICENSE"]) == repository["license_bytes"],
            _sha256(bodies["LICENSE"]) == repository["license_sha256"],
        )
    )
    gates = {
        "official_identities_exact": identities_exact,
        "material_capture_observation_explicit": facts[
            "material_capture_observation_explicit"
        ],
        "external_dataset_rights_explicit": facts["external_dataset_rights_explicit"],
        "exact_asset_manifest_with_sizes_and_checksums": facts[
            "exact_asset_manifest_with_sizes_and_checksums"
        ],
        "group_identities_ready": facts["group_identities_ready"],
        "payload_and_pixel_reads_zero": True,
    }
    status = (
        "PASS_PRIVATE_MONO_HDR_3D_REAL_EXPOSURE_SOURCE"
        if all(gates.values())
        else "FAIL_CLOSED_MONO_HDR_3D_SOURCE_RIGHTS_OR_MANIFEST_GAP_NOT_SCIENTIFIC_RESULT"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P301",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(
                ROOT / "scripts/audit_p301_mono_hdr_3d_real_exposure_source.py"
            ),
            "commit": commit,
            "tree": repository["tree"],
            "readme_sha256": _sha256(bodies["README.md"]),
            "license_sha256": _sha256(bodies["LICENSE"]),
        },
        "source_facts": {
            "repository_tree_entries": len(paths),
            "declared_real_scenes": config["external_dataset"]["declared_real_scenes"],
            "declared_poses_per_real_scene": config["external_dataset"][
                "declared_poses_per_real_scene"
            ],
            "declared_exposures_per_pose": config["external_dataset"][
                "declared_exposures_per_pose"
            ],
            "drive_folder_head_status": drive_status,
            "manifest_like_repository_paths": [
                path
                for path in paths
                if any(
                    token in path.casefold()
                    for token in ("manifest", "checksum", "sha256")
                )
            ],
            **facts,
        },
        "reads": {
            "official_text_bytes": len(bodies["README.md"]) + len(bodies["LICENSE"]),
            "github_metadata_requests": 2,
            "drive_head_requests": 1,
            "drive_body_bytes": len(drive_body),
            "dataset_file_requests": 0,
            "archive_bytes": 0,
            "image_or_pose_bytes": 0,
            "pixel_decodes": 0,
            "training_or_inference": 0,
        },
        "gates": gates,
        "decision": status,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, reverse=args.reverse)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
