"""Freeze P302 WildRelight member identities without requesting payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.audit_p286_wildrelight_paired_hdr_source_feasibility import (
    _fetch_manifest,
    _manifest_facts,
    _request_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


class P302SourceLockError(RuntimeError):
    """Raised when the frozen P302 source selection differs."""


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _rank_scenes(revision: str, scenes: list[str]) -> list[str]:
    return sorted(
        scenes,
        key=lambda scene: hashlib.sha256(
            f"{revision}:p302:{scene}".encode()
        ).hexdigest(),
    )


def _select_members(
    items: list[dict[str, object]], roles: dict[str, object]
) -> list[dict[str, object]]:
    by_path = {str(item["path"]): item for item in items if item.get("type") == "file"}
    selected: list[dict[str, object]] = []
    for role in ("training", "development", "confirmation", "reserve"):
        for scene in roles[role]:
            paths = [f"small-aligned/{scene}/meta.json"]
            for time_index in range(6):
                paths.extend(
                    (
                        f"small-aligned/{scene}/envmap/time{time_index}_envmap.exr",
                        f"small-aligned/{scene}/photo/time{time_index}_hdr.exr",
                    )
                )
            for path in paths:
                if path not in by_path:
                    raise P302SourceLockError(f"required member is absent: {path}")
                item = by_path[path]
                lfs = item.get("lfs")
                if isinstance(lfs, dict):
                    oid = str(lfs.get("oid", ""))
                    if not re.fullmatch(r"[0-9a-f]{64}", oid):
                        raise P302SourceLockError(
                            f"required member has invalid LFS SHA-256: {path}"
                        )
                    identity = {
                        "git_blob_sha1": None,
                        "identity_kind": "lfs_sha256",
                        "sha256": oid,
                    }
                else:
                    git_oid = str(item.get("oid", ""))
                    if not path.endswith("/meta.json") or not re.fullmatch(
                        r"[0-9a-f]{40}", git_oid
                    ):
                        raise P302SourceLockError(
                            f"required member has no exact identity: {path}"
                        )
                    identity = {
                        "git_blob_sha1": git_oid,
                        "identity_kind": "git_blob_sha1_and_body_sha256",
                        "sha256": None,
                    }
                selected.append(
                    {
                        "bytes": int(item["size"]),
                        **identity,
                        "path": path,
                        "role": role,
                        "scene": scene,
                    }
                )
    return selected


def _bind_metadata_bodies(
    members: list[dict[str, object]], dataset_id: str, revision: str
) -> tuple[int, int]:
    requests = 0
    body_bytes = 0
    for member in members:
        if member["identity_kind"] != "git_blob_sha1_and_body_sha256":
            continue
        url = (
            f"https://huggingface.co/datasets/{dataset_id}/resolve/"
            f"{revision}/{member['path']}?download=true"
        )
        body, _ = _request_bytes(url, 65536)
        if len(body) != int(member["bytes"]):
            raise P302SourceLockError("metadata body byte count differs")
        git_blob = hashlib.sha1(
            f"blob {len(body)}\0".encode() + body, usedforsecurity=False
        ).hexdigest()
        if git_blob != member["git_blob_sha1"]:
            raise P302SourceLockError("metadata Git-blob identity differs")
        member["sha256"] = _sha256(body)
        requests += 1
        body_bytes += len(body)
    return requests, body_bytes


def execute(config_path: Path, p286_config_path: Path) -> dict[str, object]:
    config_body = config_path.read_bytes()
    config = json.loads(config_body)
    p286_body = p286_config_path.read_bytes()
    p286 = json.loads(p286_body)
    if config["source"]["member_manifest"] is not None:
        raise P302SourceLockError(
            "source lock requires the preregistered null manifest"
        )
    if config["source"]["revision"] != p286["official_dataset"]["revision"]:
        raise P302SourceLockError("P286/P302 revisions differ")

    roles = config["roles"]
    role_scenes = [
        scene
        for role in ("training", "development", "confirmation", "reserve")
        for scene in roles[role]
    ]
    if len(role_scenes) != 29 or len(set(role_scenes)) != 29:
        raise P302SourceLockError("P302 scene roles are not disjoint and complete")
    if set(role_scenes).intersection(config["source"]["excluded_consumed_scenes"]):
        raise P302SourceLockError("P302 roles contain a consumed scene")

    manifest = _fetch_manifest(p286)
    facts = _manifest_facts(p286, manifest["items"])
    if not facts["exact_inventory"] or not facts["roles_complete"]:
        raise P302SourceLockError("P286 exact inventory no longer matches")
    eligible = [
        scene
        for scene in facts["scenes"]
        if scene not in config["source"]["excluded_consumed_scenes"]
    ]
    if _rank_scenes(config["source"]["revision"], eligible) != role_scenes:
        raise P302SourceLockError("P302 scene ranking differs")

    members = _select_members(manifest["items"], roles)
    metadata_requests, metadata_body_bytes = _bind_metadata_bodies(
        members, config["source"]["dataset_id"], config["source"]["revision"]
    )
    member_payload = {
        "dataset_id": config["source"]["dataset_id"],
        "experiment_id": "P302",
        "members": members,
        "revision": config["source"]["revision"],
        "schema": "neuro-film.p302-wildrelight-member-manifest.v1",
        "variant": config["source"]["variant"],
    }
    member_bytes = _canonical_bytes(member_payload)
    report: dict[str, Any] = {
        "archive_or_payload_requests": 0,
        "config_bytes": len(config_body),
        "config_sha256": _sha256(config_body),
        "dataset_id": config["source"]["dataset_id"],
        "experiment_id": "P302",
        "manifest_network_bytes": manifest["network_bytes"],
        "manifest_page_count": manifest["page_count"],
        "metadata_body_bytes": metadata_body_bytes,
        "metadata_body_requests": metadata_requests,
        "member_count": len(members),
        "member_manifest_bytes": len(member_bytes),
        "member_manifest_payload": member_payload,
        "member_manifest_sha256": _sha256(member_bytes),
        "pixel_decodes": 0,
        "p286_config_bytes": len(p286_body),
        "p286_config_sha256": _sha256(p286_body),
        "role_counts": {
            role: len(roles[role])
            for role in ("training", "development", "confirmation", "reserve")
        },
        "schema": "neuro-film.p302-wildrelight-source-lock.v1",
        "selected_bytes": sum(int(member["bytes"]) for member in members),
        "status": "PASS_PRIVATE_P302_METADATA_ONLY_SOURCE_LOCK",
    }
    scientific = dict(report)
    scientific.pop("manifest_network_bytes")
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p286-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve(), args.p286_config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
