"""Resumable exact REPID fit/calibration acquisition; sealed members stay unread."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_u5_r2repid2_shared_operator_roles import (
    ROOT,
    _fetch,
    _post_paths,
    _prior_scenes,
    eligible_rows,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_records(
    roles_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = roles_contract["source"]
    markup = _fetch(source["processed_markup_url"], source["processed_markup_size"])
    if (
        len(markup) != source["processed_markup_size"]
        or _sha256(markup) != source["processed_markup_sha256"]
    ):
        raise ValueError("aggregate annotation drift")
    prior, prior_identity = _prior_scenes(ROOT / source["prior_scene_root"])
    if prior_identity != source["prior_scene_identity_sha256"]:
        raise ValueError("prior scene inventory drift")
    rows = list(csv.DictReader(io.StringIO(markup.decode("utf-8-sig"))))
    selection = roles_contract["selection"]
    eligible, _ = eligible_rows(
        rows,
        geometry_fields=selection["geometry_fields"],
        minimum_margin=float(selection["minimum_direct_preference_margin"]),
        excluded_stems=prior,
    )
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"u5-r2repid2-v1|{row['scene_id']}".encode()
        ).digest()
    )
    counts = {
        role: int(selection[f"{role}_scenes"])
        for role in ("fit", "calibration", "sealed")
    }
    selected = eligible[: sum(counts.values())]
    cursor = 0
    for role, count in counts.items():
        for row in selected[cursor : cursor + count]:
            row["role"] = role
        cursor += count
    paths = [
        f"images/{endpoint}/{row['scene_id']}"
        for row in selected
        for endpoint in (row["loser"], row["winner"])
    ]
    info = {row["path"]: row for row in _post_paths(source["paths_info_url"], paths)}
    members = [
        {"path": path, "size": info[path]["size"], "sha256": info[path]["lfs"]["oid"]}
        for path in paths
    ]
    return selected, members


def _download(
    url: str, destination: Path, size: int, sha256: str, temporary_suffix: str
) -> None:
    if (
        destination.exists()
        and destination.stat().st_size == size
        and _file_sha256(destination) == sha256
    ):
        return
    temporary = destination.with_name(destination.name + temporary_suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "K-MCFM-U5-R2REPID3/1.0"}
    )
    digest = hashlib.sha256()
    written = 0
    with (
        urllib.request.urlopen(request, timeout=300) as response,
        temporary.open("wb") as output,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if written != size or digest.hexdigest() != sha256:
        raise ValueError(f"download identity mismatch: {destination}")
    os.replace(temporary, destination)


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = contract["parent"]
    roles_config_bytes = (ROOT / parent["roles_config_path"]).read_bytes()
    roles_builder_bytes = (ROOT / parent["roles_builder_path"]).read_bytes()
    roles_evidence_bytes = (ROOT / parent["roles_evidence_path"]).read_bytes()
    roles_evidence = json.loads(roles_evidence_bytes)
    bindings = {
        "roles_config": _sha256(roles_config_bytes) == parent["roles_config_sha256"],
        "roles_builder": _sha256(roles_builder_bytes) == parent["roles_builder_sha256"],
        "roles_evidence": _sha256(roles_evidence_bytes)
        == parent["roles_evidence_sha256"],
        "roles_decision": roles_evidence["decision"] == parent["required_decision"],
    }
    if not all(bindings.values()):
        raise ValueError("parent role lock binding drift")
    selected, members = _selected_records(json.loads(roles_config_bytes))
    selected_identity = _sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    )
    member_identity = _sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        selected_identity != parent["selected_identity_sha256"]
        or member_identity != parent["member_identity_sha256"]
    ):
        raise ValueError("reconstructed role identity drift")
    member_lookup = {row["path"]: row for row in members}
    initial = [
        row for row in selected if row["role"] in contract["acquisition"]["roles"]
    ]
    tasks = []
    root = ROOT / contract["acquisition"]["output_root"]
    for row in initial:
        for endpoint_name in ("loser", "winner"):
            endpoint = row[endpoint_name]
            remote = f"images/{endpoint}/{row['scene_id']}"
            member = member_lookup[remote]
            local = root / row["role"] / endpoint_name / row["scene_id"]
            url = f"{contract['acquisition']['resolve_base']}/{urllib.parse.quote(remote, safe='/')}"
            tasks.append((url, local, int(member["size"]), member["sha256"]))
    with ThreadPoolExecutor(
        max_workers=int(contract["acquisition"]["maximum_workers"])
    ) as pool:
        futures = [
            pool.submit(
                _download,
                url,
                local,
                size,
                sha,
                contract["acquisition"]["temporary_suffix"],
            )
            for url, local, size, sha in tasks
        ]
        for future in futures:
            future.result()
    records = []
    for row in initial:
        pair = []
        for endpoint_name in ("loser", "winner"):
            endpoint = row[endpoint_name]
            remote = f"images/{endpoint}/{row['scene_id']}"
            member = member_lookup[remote]
            local = root / row["role"] / endpoint_name / row["scene_id"]
            with Image.open(local) as image:
                image.load()
                if image.format != contract["decode_preflight"]["required_container"]:
                    raise ValueError("container drift")
                if image.mode != contract["decode_preflight"]["required_mode"]:
                    raise ValueError("mode drift")
                if min(image.size) < contract["decode_preflight"]["minimum_dimension"]:
                    raise ValueError("geometry below minimum")
                facts = {
                    "endpoint": endpoint_name,
                    "expert_role": endpoint,
                    "relative_path": local.relative_to(ROOT).as_posix(),
                    "size": local.stat().st_size,
                    "sha256": _file_sha256(local),
                    "width": image.width,
                    "height": image.height,
                    "icc_sha256": _sha256(image.info["icc_profile"])
                    if image.info.get("icc_profile")
                    else None,
                }
            pair.append(facts)
        if (pair[0]["width"], pair[0]["height"]) != (
            pair[1]["width"],
            pair[1]["height"],
        ):
            raise ValueError("paired dimensions differ")
        records.append(
            {
                "scene_id": row["scene_id"],
                "role": row["role"],
                "loser": row["loser"],
                "winner": row["winner"],
                "direct_margin": row["direct_margin"],
                "members": pair,
            }
        )
    scientific = {
        "schema": "neuro_film.u5_r2repid3_fit_calibration_acquisition_report.v1",
        "experiment_id": contract["experiment_id"],
        "parent_bindings": bindings,
        "selected_identity_sha256": selected_identity,
        "all_member_identity_sha256": member_identity,
        "role_counts": dict(sorted(Counter(row["role"] for row in records).items())),
        "scene_count": len(records),
        "member_count": sum(len(row["members"]) for row in records),
        "total_bytes": sum(
            member["size"] for row in records for member in row["members"]
        ),
        "records": records,
        "sealed_member_requests": 0,
        "automatic_pass": True,
        "decision": contract["decision_if_pass"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    scientific["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scientific, indent=2) + "\n", encoding="utf-8")
    return scientific


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid3_fit_calibration_acquisition_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
