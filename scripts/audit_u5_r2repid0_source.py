"""Bounded rights and aggregate-annotation source audit for official REPID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "K-MCFM-U5-R2REPID0/1.0 (bounded source audit)"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"response exceeds frozen byte budget: {url}")
    return payload


def _file_facts(data: bytes) -> dict[str, Any]:
    return {"size": len(data), "sha256": _sha256(data)}


def _license_names(data: bytes) -> list[str]:
    names = [
        line.strip() for line in data.decode("utf-8-sig").splitlines() if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError("image license list contains duplicate names")
    return names


def _annotation_structure(data: bytes) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required = {"name", "left", "right", "mos"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("processed annotation columns are incomplete")
    row_count = 0
    scenes: Counter[str] = Counter()
    pairs: set[tuple[str, str, str]] = set()
    roles: set[str] = set()
    mos_minimum = math.inf
    mos_maximum = -math.inf
    for row in reader:
        scene = row["name"]
        left = row["left"]
        right = row["right"]
        if not scene or not left or not right or left == right:
            raise ValueError("invalid annotation identity row")
        mos = float(row["mos"])
        if not math.isfinite(mos):
            raise ValueError("non-finite MOS")
        pair = (scene, *sorted((left, right)))
        if pair in pairs:
            raise ValueError("duplicate same-scene pair")
        pairs.add(pair)
        roles.update((left, right))
        scenes[scene] += 1
        row_count += 1
        mos_minimum = min(mos_minimum, mos)
        mos_maximum = max(mos_maximum, mos)
    header = list(reader.fieldnames)
    sensitive = {
        "participant_id",
        "worker_id",
        "user_id",
        "annotator_id",
        "email",
        "ip",
    }
    return {
        "file_sha256": _sha256(data),
        "file_size": len(data),
        "column_count": len(header),
        "header_identity_sha256": _sha256(",".join(header).encode()),
        "row_count": row_count,
        "scene_count": len(scenes),
        "unique_scene_pair_count": len(pairs),
        "roles": sorted(roles),
        "rows_per_scene_values": sorted(set(scenes.values())),
        "rows_per_scene_distribution": dict(sorted(Counter(scenes.values()).items())),
        "mos_minimum": mos_minimum,
        "mos_maximum": mos_maximum,
        "participant_identifier_columns_present": bool(set(header) & sensitive),
        "annotation_rows_persisted": False,
        "scene_names_persisted": False,
    }


def _tree_files(tree: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in tree:
        if row.get("type") != "file":
            continue
        lfs = row.get("lfs") or {}
        output[str(row["path"])] = {
            "size": int(row["size"]),
            "sha256": lfs.get("oid"),
        }
    return output


def evaluate(
    contract: dict[str, Any], *, fetch: Callable[[str, int], bytes] = _fetch
) -> dict[str, Any]:
    source = contract["official_sources"]
    bounded = contract["bounded_execution"]
    api = json.loads(fetch(source["api_url"], bounded["maximum_api_bytes"]))
    tree = json.loads(fetch(source["tree_url"], bounded["maximum_api_bytes"]))
    repository = {
        "sha": api["sha"],
        "last_modified": api["lastModified"],
        "private": bool(api["private"]),
        "gated": bool(api["gated"]),
        "card_license": (api.get("cardData") or {}).get("license"),
    }
    resolve = source["resolve_base"]
    small_names = (
        "README.md",
        "LICENSE.md",
        "LicenseAdobe.txt",
        "LicenseAdobeMIT.txt",
        "filesAdobe.txt",
        "filesAdobeMIT.txt",
    )
    small = {
        name: fetch(f"{resolve}/{name}", bounded["maximum_small_file_bytes"])
        for name in small_names
    }
    markup = fetch(
        f"{resolve}/processed_markup.csv",
        bounded["maximum_processed_markup_bytes"],
    )
    annotation = _annotation_structure(markup)
    adobe = set(_license_names(small["filesAdobe.txt"]))
    adobe_mit = set(_license_names(small["filesAdobeMIT.txt"]))
    license_lists = {
        "adobe_count": len(adobe),
        "adobe_mit_count": len(adobe_mit),
        "union_count": len(adobe | adobe_mit),
        "intersection_count": len(adobe & adobe_mit),
    }
    expected_files = source["expected_files"]
    observed_tree = _tree_files(tree)
    tree_exact = all(
        observed_tree.get(name) == facts for name, facts in expected_files.items()
    )
    small_exact = all(
        _file_facts(small[name]) == expected_files[name] for name in small_names
    )
    markup_exact = _file_facts(markup) == expected_files["processed_markup.csv"]
    structure_expected = source["expected_annotation_structure"]
    structure_exact = all(
        annotation[key] == value for key, value in structure_expected.items()
    )
    license_text = small["LICENSE.md"].decode("utf-8")
    rights_exact = (
        "research purposes only" in license_text
        and "Creative Commons Attribution 4.0" in license_text
        and "underlying images" in license_text
    )
    gates = {
        "exact_repository_revision": repository == source["expected_repository"],
        "exact_root_file_inventory": tree_exact,
        "exact_small_file_bytes": small_exact,
        "exact_processed_markup_bytes": markup_exact,
        "aggregate_annotation_structure_exact": structure_exact,
        "aggregate_annotation_has_no_participant_identifier_columns": not annotation[
            "participant_identifier_columns_present"
        ],
        "image_license_lists_exact_and_disjoint": license_lists
        == source["expected_image_license_lists"],
        "rights_text_explicitly_separates_annotations_from_research_only_images": rights_exact,
        "image_payload_reads_zero": True,
        "operator_fits_zero": True,
        "product_dependency_allowed": False,
    }
    automatic_pass = all(
        value for key, value in gates.items() if key != "product_dependency_allowed"
    )
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2repid0_source_audit_report.v1",
        "experiment_id": contract["experiment_id"],
        "repository": repository,
        "root_tree_file_facts": {
            name: observed_tree.get(name) for name in sorted(expected_files)
        },
        "small_file_facts": {name: _file_facts(small[name]) for name in small_names},
        "annotation_structure": annotation,
        "image_license_lists": license_lists,
        "requests": {
            "repository_api": 1,
            "root_tree_api": 1,
            "small_text_files": len(small_names),
            "aggregate_annotation_files": 1,
            "image_tree_requests": 0,
            "image_archives": 0,
            "image_members": 0,
            "image_decodes": 0,
            "operator_fits": 0,
        },
        "rights_scope": contract["rights_gate"],
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid0_source_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
