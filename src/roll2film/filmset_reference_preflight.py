"""Metadata-only FilmSet partition freeze for the W2F reference-look bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.roll2film.manifests import (
    FILMSET_ARCHIVE_VERSION,
    FILMSET_DATASET_ID,
    FILMSET_MANIFEST_SCHEMA,
    FilmSetManifestError,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise FilmSetManifestError(f"{path}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FilmSetManifestError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise FilmSetManifestError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def _validate_manifest_hash(path: Path, expected: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise FilmSetManifestError(
            f"manifest hash mismatch for {path}: expected {expected}, got {observed}"
        )


def _group_rows(
    rows: Iterable[dict[str, Any]],
    *,
    role: str,
    access: str,
    domains: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema_version") != FILMSET_MANIFEST_SCHEMA:
            raise FilmSetManifestError(f"{role}: manifest schema mismatch")
        if row.get("dataset_id") != FILMSET_DATASET_ID:
            raise FilmSetManifestError(f"{role}: dataset ID mismatch")
        if row.get("archive_version") != FILMSET_ARCHIVE_VERSION:
            raise FilmSetManifestError(f"{role}: archive version mismatch")
        if row.get("research_pool") != role:
            raise FilmSetManifestError(f"{role}: research role mismatch")
        if row.get("payload_access") != access:
            raise FilmSetManifestError(f"{role}: payload access mismatch")
        if row.get("allowed_use") != "internal_research_only":
            raise FilmSetManifestError(f"{role}: allowed-use mismatch")
        if row.get("redistributable") is not False:
            raise FilmSetManifestError(f"{role}: redistributable must be false")
        content_id = str(row.get("content_id", ""))
        domain = str(row.get("domain", ""))
        cluster = str(row.get("duplicate_cluster_id", ""))
        if not content_id or not cluster or domain not in domains:
            raise FilmSetManifestError(f"{role}: invalid content/domain/cluster row")
        by_domain = grouped.setdefault(content_id, {})
        if domain in by_domain:
            raise FilmSetManifestError(f"{role}: duplicate content/domain row")
        by_domain[domain] = row
    expected_domains = domains
    for content_id, by_domain in grouped.items():
        if set(by_domain) != expected_domains:
            raise FilmSetManifestError(
                f"{role}: incomplete domains for {content_id}: {sorted(by_domain)}"
            )
        clusters = {
            str(row["duplicate_cluster_id"]) for row in by_domain.values()
        }
        if len(clusters) != 1:
            raise FilmSetManifestError(
                f"{role}: inconsistent duplicate cluster for {content_id}"
            )
    return grouped


def _partition_ids(
    content_ids: Iterable[str], *, seed: int, pool: str
) -> list[str]:
    def key(content_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{seed}:{pool}:{content_id}".encode("utf-8")
        ).hexdigest()
        return digest, content_id

    return sorted(content_ids, key=key)


def _ids_sha256(content_ids: Iterable[str]) -> str:
    payload = "".join(f"{content_id}\n" for content_id in content_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _cluster_set(
    grouped: dict[str, dict[str, dict[str, Any]]]
) -> set[str]:
    return {
        str(next(iter(by_domain.values()))["duplicate_cluster_id"])
        for by_domain in grouped.values()
    }


def build_filmset_reference_preflight(
    config: dict[str, Any],
    *,
    root: Path,
    config_sha256: str = "unknown",
    software_commit: str = "unknown",
) -> dict[str, Any]:
    dataset = config["dataset"]
    if dataset.get("dataset_id") != FILMSET_DATASET_ID:
        raise FilmSetManifestError("FilmSet config dataset ID mismatch")
    if dataset.get("archive_version") != FILMSET_ARCHIVE_VERSION:
        raise FilmSetManifestError("FilmSet config archive version mismatch")
    dataset_root = root / dataset["root"]
    if not dataset_root.is_dir():
        raise FilmSetManifestError(f"missing FilmSet root: {dataset_root}")
    evidence_dir = root / dataset["evidence_dir"]
    paths = {
        "source": evidence_dir / dataset["source_manifest"],
        "target": evidence_dir / dataset["target_manifest"],
        "internal": evidence_dir / dataset["internal_dev_manifest"],
        "final": evidence_dir / dataset["forbidden_final_manifest"],
    }
    expected_hashes = {
        "source": dataset["source_manifest_sha256"],
        "target": dataset["target_manifest_sha256"],
        "internal": dataset["internal_dev_manifest_sha256"],
        "final": dataset["forbidden_final_manifest_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FilmSetManifestError(f"missing {name} manifest: {path}")
        _validate_manifest_hash(path, str(expected_hashes[name]))

    domains = set(dataset["domains"])
    source_rows = _load_jsonl(paths["source"])
    target_rows = _load_jsonl(paths["target"])
    internal_rows = _load_jsonl(paths["internal"])
    source = _group_rows(
        source_rows,
        role="source_train",
        access="training_source_only",
        domains={"input"},
    )
    target = _group_rows(
        target_rows,
        role="target_train",
        access="training_target_only",
        domains=domains,
    )
    internal = _group_rows(
        internal_rows,
        role="internal_dev_lockbox",
        access="internal_evaluator_only",
        domains={"input", *domains},
    )
    observed = {
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "internal_rows": len(internal_rows),
        "source_identities": len(source),
        "target_identities": len(target),
        "internal_identities": len(internal),
    }
    for key, value in observed.items():
        expected = int(dataset[f"expected_{key}"])
        if value != expected:
            raise FilmSetManifestError(
                f"{key}: expected {expected}, observed {value}"
            )

    identity_intersections = {
        "source_target": len(set(source) & set(target)),
        "source_internal": len(set(source) & set(internal)),
        "target_internal": len(set(target) & set(internal)),
    }
    clusters = {
        "source": _cluster_set(source),
        "target": _cluster_set(target),
        "internal": _cluster_set(internal),
    }
    cluster_intersections = {
        "source_target": len(clusters["source"] & clusters["target"]),
        "source_internal": len(clusters["source"] & clusters["internal"]),
        "target_internal": len(clusters["target"] & clusters["internal"]),
    }
    if any(identity_intersections.values()) or any(cluster_intersections.values()):
        raise FilmSetManifestError("FilmSet W2F pool leakage detected")

    partition = config["partition"]
    if (
        partition["algorithm"]
        != "sha256_seed_pool_content_id_ascending"
    ):
        raise FilmSetManifestError("unsupported W2F partition algorithm")
    seed = int(partition["seed"])
    target_order = _partition_ids(target, seed=seed, pool="target")
    source_order = _partition_ids(source, seed=seed, pool="source")
    internal_order = _partition_ids(internal, seed=seed, pool="internal")
    target_n = int(partition["target_reference_bank_identities"])
    source_n = int(partition["source_probe_identities"])
    development_n = int(partition["internal_development_identities"])
    confirmatory_n = int(partition["internal_confirmatory_identities"])
    stress_n = int(partition["internal_stress_identities"])
    if target_n <= 0 or target_n >= len(target_order):
        raise FilmSetManifestError("target reference-bank count is invalid")
    if source_n <= 0 or source_n >= len(source_order):
        raise FilmSetManifestError("source probe count is invalid")
    if development_n + confirmatory_n + stress_n != len(internal_order):
        raise FilmSetManifestError("internal W2F partition does not exhaust identities")

    partitions = {
        "target_reference_bank": target_order[:target_n],
        "target_reference_reserve": target_order[target_n:],
        "source_probe": source_order[:source_n],
        "source_reserve": source_order[source_n:],
        "internal_development": internal_order[:development_n],
        "internal_confirmatory": internal_order[
            development_n : development_n + confirmatory_n
        ],
        "internal_stress": internal_order[development_n + confirmatory_n :],
    }
    if len(set().union(*(set(ids) for ids in partitions.values()))) != (
        len(target) + len(source) + len(internal)
    ):
        raise FilmSetManifestError("W2F partition union is not identity exact")

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": config["status"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "dataset_root_exists": True,
        "manifest_sha256": expected_hashes,
        "observed": observed,
        "identity_intersections": identity_intersections,
        "duplicate_cluster_intersections": cluster_intersections,
        "partition": {
            "algorithm": partition["algorithm"],
            "seed": seed,
            "counts": {name: len(ids) for name, ids in partitions.items()},
            "content_id_sha256": {
                name: _ids_sha256(ids) for name, ids in partitions.items()
            },
        },
        "final_manifest_payload_rows_parsed": 0,
        "image_payloads_read": 0,
        "image_payloads_decoded": 0,
        "activation_gate": config["activation_gate"],
        "allowed_use": config["allowed_use"],
        "claim_ceiling": config["claim_ceiling"],
        "decision": "source_ready_execution_queued_behind_repeated_w1",
    }
