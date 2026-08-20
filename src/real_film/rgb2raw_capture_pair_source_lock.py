"""Range-only source qualification for the NTIRE 2025 RGB2RAW capture pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from src.real_film.ppisp_capture_pair_source_lock import (
    PPISPSourceLockError,
    ZipMember,
    canonical_sha256,
    http_range_get,
    parse_central_directory,
    sha256_bytes,
)


class RGB2RAWSourceLockError(ValueError):
    """Raised when the frozen RGB2RAW source contract is invalid."""


def _read_url(url: str) -> bytes:
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroFilm-SF3-A0R/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _unsafe(member: ZipMember) -> bool:
    name = member.name
    path = PurePosixPath(name)
    host = member.version_made_by >> 8
    mode = member.external_attributes >> 16
    symlink = host == 3 and (mode & 0o170000) == 0o120000
    return (
        "\\" in name
        or "\x00" in name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or bool(member.flags & 1)
        or member.method not in (0, 8)
        or symlink
    )


def analyze_members(members: list[ZipMember]) -> dict[str, Any]:
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise RGB2RAWSourceLockError("duplicate ZIP member names")
    by_camera: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    metadata: dict[str, set[str]] = defaultdict(set)
    for member in members:
        path = PurePosixPath(member.name)
        if len(path.parts) != 3 or path.parts[0] != "train":
            continue
        camera = path.parts[1]
        suffix = path.suffix.lower()
        if suffix in (".png", ".npy"):
            by_camera[camera][suffix].add(path.stem)
        elif suffix == ".pkl":
            metadata[camera].add(path.stem)

    primary_cameras = ("iphone-x", "samsung-s9")
    camera_rows = []
    all_scene_groups: set[str] = set()
    all_pair_keys: list[str] = []
    metadata_covered = True
    for camera in primary_cameras:
        paired = sorted(by_camera[camera][".png"] & by_camera[camera][".npy"])
        unpaired = sorted(by_camera[camera][".png"] ^ by_camera[camera][".npy"])
        scene_groups = sorted({stem.rsplit("_", 1)[0] for stem in paired})
        missing_metadata = sorted(set(scene_groups) - metadata[camera])
        metadata_covered &= not missing_metadata
        all_scene_groups.update(f"{camera}/{scene}" for scene in scene_groups)
        all_pair_keys.extend(f"{camera}/{stem}" for stem in paired)
        camera_rows.append(
            {
                "camera_group": camera,
                "png_count": len(by_camera[camera][".png"]),
                "npy_count": len(by_camera[camera][".npy"]),
                "pair_count": len(paired),
                "scene_group_count": len(scene_groups),
                "metadata_count": len(metadata[camera]),
                "unpaired_count": len(unpaired),
                "missing_metadata_count": len(missing_metadata),
                "pair_keys_sha256": canonical_sha256(paired),
                "scene_groups_sha256": canonical_sha256(scene_groups),
            }
        )
    return {
        "member_count": len(members),
        "central_members_sha256": canonical_sha256(
            [
                {
                    "name": member.name,
                    "flags": member.flags,
                    "method": member.method,
                    "crc32": member.crc32,
                    "compressed_size": member.compressed_size,
                    "uncompressed_size": member.uncompressed_size,
                    "local_offset": member.local_offset,
                }
                for member in members
            ]
        ),
        "camera_rows": camera_rows,
        "primary_pair_count": len(all_pair_keys),
        "primary_scene_group_count": len(all_scene_groups),
        "metadata_member_count": sum(len(values) for values in metadata.values()),
        "primary_pair_keys_sha256": canonical_sha256(sorted(all_pair_keys)),
        "primary_scene_groups_sha256": canonical_sha256(sorted(all_scene_groups)),
        "metadata_covers_primary_scene_groups": metadata_covered,
        "unsafe_member_count": sum(_unsafe(member) for member in members),
    }


RangeReader = Callable[[str, int, int, int], bytes]


def run_source_lock(
    config_path: Path,
    *,
    range_reader: RangeReader = http_range_get,
    url_reader: Callable[[str], bytes] = _read_url,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != (
        "neuro-film.sf3-a0r-rgb2raw-capture-pair-source-lock-contract.v1"
    ):
        raise RGB2RAWSourceLockError("contract schema differs")
    source = config["source"]
    archive = source["archive"]
    readme = url_reader(source["readme_url"])
    url = (
        f"https://huggingface.co/datasets/{source['dataset_id']}/resolve/"
        f"{source['revision']}/{archive['path']}?download=true"
    )
    central = range_reader(
        url,
        archive["central_offset"],
        archive["central_offset"] + archive["central_size"] - 1,
        archive["size"],
    )
    if sha256_bytes(central) != archive["central_sha256"]:
        raise RGB2RAWSourceLockError("central-directory identity differs")
    try:
        members = parse_central_directory(central)
    except PPISPSourceLockError as exc:
        raise RGB2RAWSourceLockError(str(exc)) from exc
    facts = analyze_members(members)
    gates_config = config["gates"]
    camera_groups = [row["camera_group"] for row in facts["camera_rows"]]
    gates = {
        "readme_identity_exact": (
            len(readme) == source["readme_size"]
            and sha256_bytes(readme) == source["readme_sha256"]
        ),
        "publisher_declares_mit": (
            source["license_declaration"] == "mit"
            and readme.startswith(b"---\nlicense: mit\n")
        ),
        "primary_camera_groups_exact": (
            camera_groups == gates_config["required_primary_camera_groups"]
        ),
        "primary_pair_count_exact": (
            facts["primary_pair_count"] == gates_config["required_primary_pair_count"]
        ),
        "primary_scene_group_count_exact": (
            facts["primary_scene_group_count"]
            == gates_config["required_primary_scene_group_count"]
        ),
        "metadata_member_count_exact": (
            facts["metadata_member_count"]
            == gates_config["required_metadata_member_count"]
        ),
        "each_primary_pair_png_npy": all(
            row["unpaired_count"] == 0 for row in facts["camera_rows"]
        ),
        "metadata_for_each_primary_scene_group": facts[
            "metadata_covers_primary_scene_groups"
        ],
        "safe_unencrypted_supported_members": facts["unsafe_member_count"] == 0,
        "zero_member_payload_reads": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0r-rgb2raw-capture-pair-source-lock-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source": {
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "readme_size": len(readme),
            "readme_sha256": sha256_bytes(readme),
            "license_declaration": source["license_declaration"],
            "license_text_present": source["license_text_present"],
            "archive_path": archive["path"],
            "archive_size": archive["size"],
            "archive_lfs_sha256": archive["lfs_sha256"],
            "archive_xet_hash": archive["xet_hash"],
            "central_offset": archive["central_offset"],
            "central_size": archive["central_size"],
            "central_sha256": sha256_bytes(central),
        },
        **facts,
        "archive_bytes_read": len(central),
        "member_payload_reads": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "renders": 0,
        "scores": 0,
        "bounded_final_candidate_counter_before": 0,
        "bounded_final_candidate_counter_after": 0,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
