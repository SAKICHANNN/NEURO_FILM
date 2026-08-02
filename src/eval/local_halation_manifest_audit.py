"""Bounded metadata-only audit for P3Y-compatible local manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u6_p3z_local_halation_manifest_audit_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3z_local_halation_manifest_audit_report.v1"


class LocalManifestAuditError(ValueError):
    """Raised when the P3Z scope, parents or bounded inventory drift."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LocalManifestAuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise LocalManifestAuditError(f"non-finite JSON constant: {value}")

    return json.loads(
        text, object_pairs_hook=object_pairs, parse_constant=reject_constant
    )


def _verify_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = root / binding["path"]
    if _sha256_bytes(path.read_bytes()) != binding["sha256"]:
        raise LocalManifestAuditError(f"P3Z {label} hash drift")
    return path


def load_contract(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _strict_json_loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3Z"
    ):
        raise LocalManifestAuditError("unsupported P3Z contract")
    parents = contract["parents"]
    protocol_path = _verify_file(root, parents["p3y_protocol"], "parent protocol")
    decision_path = _verify_file(root, parents["p3y_decision"], "parent decision")
    _verify_file(root, parents["p3y_validator"], "parent validator")
    decision = _strict_json_loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != parents["p3y_decision"]["required_decision"]:
        raise LocalManifestAuditError("P3Z parent decision drift")
    protocol = _strict_json_loads(protocol_path.read_text(encoding="utf-8"))
    if contract["scope"]["pixel_reads_allowed"] is not False:
        raise LocalManifestAuditError("P3Z pixel-read policy drift")
    return contract, protocol


def _discover_files(data_root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(data_root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        for name in sorted(names):
            path = current_path / name
            if path.suffix.lower() in extensions and path.is_file():
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(data_root).as_posix())


def _dictionary_nodes(value: Any, maximum_nodes: int) -> list[dict[str, Any]]:
    stack = [value]
    dictionaries: list[dict[str, Any]] = []
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > maximum_nodes:
            raise LocalManifestAuditError("P3Z JSON node bound exceeded")
        if isinstance(current, dict):
            dictionaries.append(current)
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            stack.extend(reversed(current))
    return dictionaries


def _parse_nodes(path: Path, maximum_nodes: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        dictionaries: list[dict[str, Any]] = []
        remaining = maximum_nodes
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = _strict_json_loads(line)
            nodes = _dictionary_nodes(value, remaining)
            remaining -= len(nodes)
            if remaining < 0:
                raise LocalManifestAuditError(
                    f"P3Z JSON node bound exceeded at line {line_number}"
                )
            dictionaries.extend(nodes)
        return dictionaries
    return _dictionary_nodes(_strict_json_loads(text), maximum_nodes)


def _candidate_facts(
    dictionaries: list[dict[str, Any]], required_fields: set[str]
) -> dict[str, Any]:
    exact_indices: list[int] = []
    maximum_overlap = 0
    maximum_overlap_indices: list[int] = []
    for index, row in enumerate(dictionaries):
        overlap = len(required_fields & set(row))
        if overlap > maximum_overlap:
            maximum_overlap = overlap
            maximum_overlap_indices = [index]
        elif overlap == maximum_overlap and overlap > 0:
            maximum_overlap_indices.append(index)
        if required_fields.issubset(set(row)):
            exact_indices.append(index)
    return {
        "dictionary_node_count": len(dictionaries),
        "exact_candidate_node_indices": exact_indices,
        "exact_candidate_count": len(exact_indices),
        "maximum_required_field_overlap_count": maximum_overlap,
        "maximum_required_field_overlap_node_indices": maximum_overlap_indices[:16],
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, protocol = load_contract(root, contract_path)
    scope = contract["scope"]
    data_root = root / scope["root"]
    if not data_root.is_dir():
        raise LocalManifestAuditError("P3Z data root is unavailable")
    files = _discover_files(data_root, set(scope["extensions"]))
    total_bytes = sum(path.stat().st_size for path in files)
    inventory_within_bounds = (
        len(files) <= int(scope["maximum_files"])
        and total_bytes <= int(scope["maximum_total_bytes"])
        and all(
            path.stat().st_size <= int(scope["maximum_file_bytes"]) for path in files
        )
    )
    if not inventory_within_bounds:
        raise LocalManifestAuditError("P3Z metadata inventory exceeds frozen bounds")
    required_fields = set(protocol["required_row_fields"])
    inventory: list[dict[str, Any]] = []
    parse_failure_count = 0
    exact_candidates: list[dict[str, Any]] = []
    maximum_overlap = 0
    maximum_overlap_files: list[str] = []
    for path in files:
        relative = path.relative_to(data_root).as_posix()
        payload = path.read_bytes()
        try:
            dictionaries = _parse_nodes(path, int(scope["maximum_json_nodes_per_file"]))
            facts = _candidate_facts(dictionaries, required_fields)
            parse_status = "parsed"
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            LocalManifestAuditError,
        ) as exc:
            parse_failure_count += 1
            facts = {
                "dictionary_node_count": 0,
                "exact_candidate_node_indices": [],
                "exact_candidate_count": 0,
                "maximum_required_field_overlap_count": 0,
                "maximum_required_field_overlap_node_indices": [],
            }
            parse_status = f"parse-failed:{type(exc).__name__}"
        if facts["maximum_required_field_overlap_count"] > maximum_overlap:
            maximum_overlap = int(facts["maximum_required_field_overlap_count"])
            maximum_overlap_files = [relative]
        elif (
            facts["maximum_required_field_overlap_count"] == maximum_overlap
            and maximum_overlap > 0
        ):
            maximum_overlap_files.append(relative)
        if facts["exact_candidate_count"]:
            exact_candidates.append(
                {
                    "path": relative,
                    "node_indices": facts["exact_candidate_node_indices"],
                }
            )
        inventory.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "parse_status": parse_status,
                **facts,
            }
        )
    relative_paths = [row["path"] for row in inventory]
    duplicate_paths = len(relative_paths) - len(set(relative_paths))
    gates = contract["automatic_gates"]
    gate_results = {
        "inventory_within_bounds": inventory_within_bounds,
        "relative_paths_unique": duplicate_paths
        <= int(gates["duplicate_relative_path_count_max"]),
        "all_metadata_parse": parse_failure_count
        <= int(gates["parse_failure_count_max"]),
        "no_pixel_reads": 0 <= int(gates["pixel_read_count_max"]),
    }
    core = {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "protocol_sha256": contract["parents"]["p3y_protocol"]["sha256"],
        "scope": scope,
        "inventory": inventory,
        "summary": {
            "metadata_file_count": len(files),
            "metadata_total_bytes": total_bytes,
            "parse_failure_count": parse_failure_count,
            "dictionary_node_count": sum(
                int(row["dictionary_node_count"]) for row in inventory
            ),
            "exact_candidate_count": sum(
                int(row["exact_candidate_count"]) for row in inventory
            ),
            "exact_candidate_files": exact_candidates,
            "maximum_required_field_overlap_count": maximum_overlap,
            "maximum_required_field_overlap_files": maximum_overlap_files[:16],
            "required_field_count": len(required_fields),
            "pixel_read_count": 0,
        },
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "decision": (
            "candidate-manifest-found-requires-p3y-validation"
            if exact_candidates
            else "no-local-p3y-candidate-manifest-data-gap"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "scientific_stable_id": _sha256_bytes(_canonical_bytes(core))}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "LocalManifestAuditError",
    "_candidate_facts",
    "evaluate",
    "load_contract",
    "write_report",
]
