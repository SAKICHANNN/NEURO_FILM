"""Audit source-clean non-generative PPSD preference structure without pixels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _fetch(url: str, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "K-MCFM-U5-R2PPSD1/1.0 (bounded structure audit)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("responses archive exceeds frozen byte budget")
    return content


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256(path.read_bytes()) != expected_sha256:
        raise ValueError(f"hash-bound parent or primary material drift: {path}")


def _valid_vote(row: dict[str, Any]) -> bool:
    required = ("collection", "scene_id", "left_style", "right_style", "choice", "user_id")
    if any(key not in row for key in required):
        return False
    if any(row[key] in (None, "") for key in required):
        return False
    return row["left_style"] != row["right_style"]


def _audit_archive(archive: bytes, contract: dict[str, Any]) -> dict[str, Any]:
    roles = contract["source_roles"]
    included = tuple(roles["included_collections"])
    included_set = set(included)
    excluded_set = set(roles["excluded_collections"])
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        vote_rows = [
            json.loads(line)
            for line in bundle.read("responses/raw/votes_items.jsonl").splitlines()
            if line.strip()
        ]
        processed_scene_keys: set[str] = set()
        for name in sorted(bundle.namelist()):
            if not name.startswith("responses/processed/") or not name.endswith(".json"):
                continue
            payload = json.loads(bundle.read(name))
            if isinstance(payload, dict):
                processed_scene_keys.update(str(key) for key in payload)

    processed_counts = Counter(
        key.split("-", 1)[0] if "-" in key else "unknown" for key in processed_scene_keys
    )
    observed_vote_collections = {
        str(row.get("collection")) for row in vote_rows if isinstance(row, dict)
    }
    known_collections = included_set | excluded_set
    retained = [
        row
        for row in vote_rows
        if isinstance(row, dict) and str(row.get("collection")) in included_set
    ]
    invalid_rows = sum(not isinstance(row, dict) or not _valid_vote(row) for row in retained)
    vote_counts: Counter[str] = Counter()
    scene_sets: defaultdict[str, set[str]] = defaultdict(set)
    participant_sets: defaultdict[str, set[str]] = defaultdict(set)
    style_pair_sets: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    structural_rows: list[dict[str, str]] = []
    for row in retained:
        if not _valid_vote(row):
            continue
        collection = str(row["collection"])
        scene = str(row["scene_id"])
        participant = str(row["user_id"])
        left = str(row["left_style"])
        right = str(row["right_style"])
        vote_counts[collection] += 1
        scene_sets[collection].add(scene)
        participant_sets[collection].add(participant)
        style_pair_sets[collection].add((left, right))
        structural_rows.append(
            {
                "collection": collection,
                "scene_token_sha256": _sha256(f"ppsd1-scene:{scene}".encode()),
                "participant_token_sha256": _sha256(f"ppsd1-participant:{participant}".encode()),
                "style_pair_sha256": _sha256(f"ppsd1-style-pair:{left}:{right}".encode()),
                "choice_type": type(row["choice"]).__name__,
            }
        )
    included_scene_counts = {key: processed_counts.get(key, 0) for key in included}
    expected_counts = roles["expected_processed_scene_counts"]
    excluded_retained = sum(
        str(row.get("collection")) in excluded_set for row in retained if isinstance(row, dict)
    )
    collection_summary = {
        key: {
            "processed_scene_keys": included_scene_counts[key],
            "vote_rows": vote_counts[key],
            "unique_vote_scenes": len(scene_sets[key]),
            "unique_participants": len(participant_sets[key]),
            "ordered_style_pairs": len(style_pair_sets[key]),
        }
        for key in included
    }
    return {
        "archive_sha256": _sha256(archive),
        "total_raw_vote_rows": len(vote_rows),
        "retained_vote_rows": len(retained),
        "retained_collection_summary": collection_summary,
        "retained_processed_scene_keys": sum(included_scene_counts.values()),
        "retained_structure_identity_sha256": _canonical_sha256(
            sorted(structural_rows, key=lambda row: tuple(row.values()))
        ),
        "excluded_collection_vote_rows_persisted": excluded_retained,
        "excluded_raw_vote_rows": sum(
            str(row.get("collection")) in excluded_set
            for row in vote_rows
            if isinstance(row, dict)
        ),
        "unexpected_raw_vote_collection_count": len(observed_vote_collections - known_collections),
        "invalid_retained_vote_rows": invalid_rows,
        "raw_scene_or_user_ids_persisted": False,
        "raw_participant_values_persisted": False,
        "checks": {
            "included_scene_counts_exact": included_scene_counts == expected_counts,
            "included_scene_total_exact": sum(included_scene_counts.values())
            == roles["expected_included_scene_keys"],
            "excluded_collections_absent_from_retained_rows": excluded_retained == 0,
            "raw_vote_collection_partition_exact": observed_vote_collections == known_collections,
            "all_included_vote_rows_structurally_valid": invalid_rows == 0 and bool(retained),
            "each_included_collection_has_votes_and_participants": all(
                vote_counts[key] > 0 and participant_sets[key] for key in included
            ),
        },
    }


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path = ROOT,
    fetch: Callable[[str, int], bytes] = _fetch,
) -> dict[str, Any]:
    parents = contract["parents"]
    _verify_file(root / parents["ppsd0_evidence"]["path"], parents["ppsd0_evidence"]["sha256"])
    parent = json.loads((root / parents["ppsd0_evidence"]["path"]).read_text(encoding="utf-8"))
    if parent.get("decision") != parents["ppsd0_evidence"]["required_decision"]:
        raise ValueError("PPSD0 decision drift")
    _verify_file(root / parents["paper"]["path"], parents["paper"]["sha256"])
    _verify_file(root / parents["supplemental"]["path"], parents["supplemental"]["sha256"])
    source = contract["annotation_source"]
    archive = fetch(
        f"https://drive.usercontent.google.com/download?id={source['drive_file_id']}&export=download",
        source["maximum_archive_bytes"],
    )
    if _sha256(archive) != parents["responses_archive_sha256"]:
        raise ValueError("responses archive identity drift")
    structure = _audit_archive(archive, contract)
    checks = structure["checks"]
    gates = {
        "parent_and_primary_material_hashes_exact": True,
        "included_scene_counts_exact": checks["included_scene_counts_exact"]
        and checks["included_scene_total_exact"],
        "excluded_collections_absent_from_retained_rows": checks[
            "excluded_collections_absent_from_retained_rows"
        ]
        and checks["raw_vote_collection_partition_exact"],
        "all_included_vote_rows_structurally_valid": checks[
            "all_included_vote_rows_structurally_valid"
        ],
        "each_included_collection_has_votes_and_participants": checks[
            "each_included_collection_has_votes_and_participants"
        ],
        "no_raw_identifiers_or_participant_values_persisted": not structure[
            "raw_scene_or_user_ids_persisted"
        ]
        and not structure["raw_participant_values_persisted"],
        "pixels_or_training_allowed": False,
    }
    automatic_pass = all(value for key, value in gates.items() if key != "pixels_or_training_allowed")
    report = {
        "schema": "neuro_film.u5_r2ppsd1_source_clean_structure_report.v1",
        "experiment_id": contract["experiment_id"],
        "source_roles": contract["source_roles"],
        "annotation_structure": {key: value for key, value in structure.items() if key != "checks"},
        "requests": {
            "annotation_archives": 1,
            "image_archives": 0,
            "image_members": 0,
            "image_decodes": 0,
            "operator_fits": 0,
        },
        "rights_scope": contract["privacy_and_rights"]["rights_scope"],
        "explicit_dataset_license_observed": False,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = f"sha256:{_canonical_sha256(report)}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/u5_r2ppsd1_source_clean_structure_v1.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "stable_evidence_id": report["stable_evidence_id"]}))


if __name__ == "__main__":
    main()
