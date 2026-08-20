#!/usr/bin/env python3
"""Acquire only frozen SPCP2 fit/calibration winner-loser PNG pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    ROOT,
    _canonical_bytes,
    _extract_member,
    _fetch,
    _sha256,
    _stable_id,
)
from src.preprocess.output_encode import (
    normalized_icc_profile_sha256,
    srgb_icc_profile_fingerprint_sha256,
)

DEFAULT_CONTRACT = ROOT / "configs/u5_r2spcp2a_preference_pair_acquisition_v1.json"


def _bound(root: Path, row: dict[str, Any], label: str) -> tuple[Path, bytes]:
    relative = Path(row[f"{label}_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid {label} path")
    path = root / relative
    payload = path.read_bytes()
    if _sha256(payload) != row[f"{label}_sha256"]:
        raise ValueError(f"{label} hash drift")
    return path, payload


def _fetch_member(url: str, member: dict[str, Any]) -> bytes:
    offset = int(member["local_offset"])
    header, _ = _fetch(url, (offset, offset + 29))
    if header[:4] != b"PK\x03\x04":
        raise ValueError("local ZIP header missing")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    end = offset + 30 + name_len + extra_len + int(member["compressed_size"]) - 1
    payload, _ = _fetch(url, (offset, end))
    return _extract_member(payload, range_start=offset, row=member)


def _inspect_png(payload: bytes, contract: dict[str, Any]) -> dict[str, Any]:
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != contract["decode"]["required_format"]:
            raise ValueError("SPCP member is not PNG")
        if image.mode != contract["decode"]["required_mode"]:
            raise ValueError("SPCP PNG is not opaque RGB")
        width, height = image.size
        profile = image.info.get("icc_profile", b"")
        samples = np.asarray(image)
    if min(width, height) < int(contract["decode"]["minimum_width_height"]):
        raise ValueError("SPCP PNG is below minimum dimensions")
    if samples.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("SPCP PNG sample depth is unsupported")
    if profile and normalized_icc_profile_sha256(profile) != srgb_icc_profile_fingerprint_sha256():
        raise ValueError("SPCP PNG contains a non-sRGB ICC profile")
    return {
        "width": width,
        "height": height,
        "dtype": samples.dtype.name,
        "pixel_sha256": hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest(),
        "icc_sha256": _sha256(profile) if profile else None,
    }


def acquire(contract_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("schema") != "neuro-film.u5-r2spcp2a-preference-pair-acquisition-contract.v1":
        raise ValueError("unsupported SPCP2A contract")
    parent = contract["parent"]
    _, parent_bytes = _bound(root, parent, "contract")
    _, role_bytes = _bound(root, parent, "role_manifest")
    parent_contract = json.loads(parent_bytes)
    roles = json.loads(role_bytes)
    if roles["status"] != parent["required_status"] or roles["decision"] != parent["required_decision"]:
        raise ValueError("SPCP2 role admission drift")
    allowed_roles = set(contract["acquisition"]["roles"])
    selected = [row for row in roles["selected_rows"] if row["role"] in allowed_roles]
    members = [
        (row, endpoint, row[f"{endpoint}_member"])
        for row in selected
        for endpoint in ("loser", "winner")
    ]
    if (
        len(selected) != int(contract["acquisition"]["scene_count_exact"])
        or len(members) != int(contract["acquisition"]["member_count_exact"])
        or sum(int(member["compressed_size"]) for _, _, member in members)
        != int(contract["acquisition"]["compressed_bytes_exact"])
    ):
        raise ValueError("SPCP2 acquisition inventory drift")
    output_root = root / contract["acquisition"]["output_root"]
    if output_root.exists():
        raise FileExistsError("SPCP2A output root is create-only")
    output_root.mkdir(parents=True)
    try:
        url = parent_contract["source"]["zip_url"]
        workers = int(contract["acquisition"]["maximum_parallel_requests"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            payloads = tuple(executor.map(lambda item: _fetch_member(url, item[2]), members))
        facts: list[dict[str, Any]] = []
        pair_dimensions: dict[str, set[tuple[int, int]]] = {}
        for (row, endpoint, member), payload in zip(members, payloads, strict=True):
            inspection = _inspect_png(payload, contract)
            relative = Path(row["role"]) / row["scene_id"] / f"{endpoint}.png"
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            pair_dimensions.setdefault(row["scene_id"], set()).add(
                (inspection["width"], inspection["height"])
            )
            facts.append(
                {
                    "scene_id": row["scene_id"],
                    "role": row["role"],
                    "endpoint": endpoint,
                    "image_id": row[f"{endpoint}_image_id"],
                    "relative_path": relative.as_posix(),
                    "file_sha256": _sha256(payload),
                    "file_bytes": len(payload),
                    **inspection,
                }
            )
        mismatched = sorted(scene for scene, sizes in pair_dimensions.items() if len(sizes) != 1)
        gates = {
            "parent_and_roles_exact": True,
            "scene_count_exact": len(selected) == contract["acquisition"]["scene_count_exact"],
            "member_count_exact": len(facts) == contract["acquisition"]["member_count_exact"],
            "pair_dimensions_match": not mismatched,
            "all_members_unique": len({row["file_sha256"] for row in facts}) == len(facts),
            "all_png_rgb_supported": True,
            "sealed_payload_reads_zero": True,
            "full_zip_downloaded": False,
        }
        failed = sorted(key for key, value in gates.items() if not value)
        report = {
            "schema": "neuro-film.u5-r2spcp2a-preference-pair-acquisition-report.v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": _sha256(contract_bytes),
            "role_manifest_sha256": _sha256(role_bytes),
            "rows": facts,
            "aggregate": {
                "scenes": len(selected),
                "members": len(facts),
                "file_bytes": sum(row["file_bytes"] for row in facts),
                "mismatched_pair_scenes": mismatched,
            },
            "gates": gates,
            "failed_gates": failed,
            "status": "PASS_EXACT_PAIR_ACQUISITION" if not failed else "FAIL_CLOSED_PAIR_ACQUISITION",
            "decision": contract["decision_if_pass"] if not failed else contract["decision_if_fail"],
            "claim_ceiling": contract["claim_ceiling"],
        }
        report["stable_evidence_id"] = _stable_id(report)
        (output_root / "acquisition_manifest.json").write_bytes(_canonical_bytes(report))
        return report
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(json.dumps(acquire(args.contract.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
