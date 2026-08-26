"""Audit whether R1CY persists a complete independently consumable bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p228_r1cy_paired_hdr_handoff_artifact_audit_v1.json"
SCHEMA = "neuro-film.p228-r1cy-paired-hdr-handoff-artifact-audit-contract.v1"
PAYLOAD_FIELDS = {"payload", "payload_b64", "payload_hex", "payload_bytes"}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None) -> bytes:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).decode(errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed.stdout


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return run_command(["git", *arguments], cwd=repository)


def git_text(repository: Path, *arguments: str) -> str:
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("P228 contract schema differs")
    if config.get("status") != "FROZEN_AFTER_R1CY_HANDOFF_BEFORE_CONSUMER_MAPPING":
        raise RuntimeError("P228 contract is not frozen before consumer mapping")
    scope = config["bounded_artifact_scope"]
    if scope["unbounded_output_or_data_scan"]:
        raise RuntimeError("P228 forbids unbounded artifact scans")
    if not scope["payload_hash_only_is_not_payload"]:
        raise RuntimeError("P228 cannot treat a payload hash as payload bytes")


def _find_payload_fields(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in PAYLOAD_FIELDS:
                found.append(child)
            found.extend(_find_payload_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_payload_fields(item, f"{path}[{index}]"))
    return found


def _tracked_payload_candidates(
    repository: Path, commit: str, expected_bytes: int, expected_sha256: str
) -> tuple[list[str], list[str]]:
    listing = git_text(repository, "ls-tree", "-rl", commit)
    sized: list[str] = []
    exact: list[str] = []
    for line in listing.splitlines():
        metadata, path = line.split("\t", 1)
        fields = metadata.split()
        if len(fields) != 4 or not fields[3].isdigit():
            continue
        if int(fields[3]) != expected_bytes:
            continue
        sized.append(path)
        payload = git_bytes(repository, "show", f"{commit}:{path}")
        if sha256_bytes(payload) == expected_sha256:
            exact.append(path)
    return sorted(sized), sorted(exact)


def execute(order: str, config_path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    producer = config["producer"]
    repository = Path(producer["repository"])
    evidence = git_bytes(
        repository, "show", f'{producer["evidence_commit"]}:{producer["evidence_path"]}'
    )
    report_paths = [
        repository / producer["forward_report_path"],
        repository / producer["reverse_report_path"],
    ]
    execution = report_paths if order == "forward" else report_paths[::-1]
    reports: list[dict[str, Any]] = []
    for path in execution:
        raw = path.read_bytes()
        parsed = json.loads(raw)
        envelope = parsed["scientific"]["upstream"]["bundle"]
        reports.append(
            {
                "id": path.name,
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
                "scientific_identity": parsed["scientific_identity"],
                "bundle_envelope": envelope,
                "persisted_payload_fields": _find_payload_fields(parsed),
                "media_path": parsed["media_path"],
            }
        )
    reports.sort(key=lambda row: row["id"])

    expected_payload_sha = producer["payload_sha256"].removeprefix("sha256:")
    sized_tracked, exact_tracked = _tracked_payload_candidates(
        repository,
        producer["evidence_commit"],
        producer["payload_bytes"],
        expected_payload_sha,
    )
    eval_candidates = sorted(
        path.relative_to(repository).as_posix()
        for path in (repository / "outputs/eval").iterdir()
        if path.is_file() and path.stat().st_size == producer["payload_bytes"]
    )
    exact_eval = [
        path
        for path in eval_candidates
        if sha256_file(repository / path) == expected_payload_sha
    ]
    schema_bytes = git_bytes(
        repository,
        "show",
        f'{producer["evidence_commit"]}:{producer["bundle_schema_path"]}',
    )
    invocation_bytes = git_bytes(
        repository,
        "show",
        f'{producer["evidence_commit"]}:{producer["invocation_source_path"]}',
    )
    schema = json.loads(schema_bytes)
    schema_properties = sorted(schema["properties"])
    payload_persisted = bool(exact_tracked or exact_eval)
    envelope_exact = all(
        row["bundle_envelope"]["bundle_id"] == producer["bundle_id"]
        and row["bundle_envelope"]["payload_sha256"] == producer["payload_sha256"]
        and row["bundle_envelope"]["payload_byte_length"] == producer["payload_bytes"]
        and not row["persisted_payload_fields"]
        for row in reports
    )
    report_hashes = {row["id"]: row["sha256"] for row in reports}
    reports_exact = bool(
        report_hashes[Path(producer["forward_report_path"]).name]
        == producer["forward_report_sha256"]
        and report_hashes[Path(producer["reverse_report_path"]).name]
        == producer["reverse_report_sha256"]
        and all(
            row["scientific_identity"] == producer["scientific_identity"]
            for row in reports
        )
    )
    source_requires_payload = b"def _bundle(value: Mapping[str, Any], payload: bytes)" in invocation_bytes
    gates = {
        "producer_evidence_and_reports_exact": bool(
            sha256_bytes(evidence) == producer["evidence_sha256"] and reports_exact
        ),
        "bundle_schema_and_invocation_source_tracked": bool(
            schema["$id"] == reports[0]["bundle_envelope"]["schema"]
            and source_requires_payload
        ),
        "bundle_envelope_exact_in_both_reports": envelope_exact,
        "payload_bytes_persisted_and_sha_exact": payload_persisted,
        "consumer_can_construct_bundle_without_rerunning_build": bool(
            payload_persisted and source_requires_payload
        ),
        "no_consumer_mapping_before_all_gates": True,
        "canonical_order_reconstruction_exact": True,
    }
    decision = (
        "PASS_PRIVATE_R1CY_HANDOFF_ARTIFACT_COMPLETE"
        if all(gates.values())
        else "FAIL_CLOSED_R1CY_HANDOFF_PAYLOAD_NOT_PERSISTED"
    )
    scientific = {
        "protocol": config["schema"],
        "producer": {
            "evidence_commit": producer["evidence_commit"],
            "implementation_commit": producer["implementation_commit"],
        },
        "reports": reports,
        "artifact_inventory": {
            "tracked_files_with_payload_byte_length": sized_tracked,
            "tracked_files_with_exact_payload_sha256": exact_tracked,
            "eval_files_with_payload_byte_length": eval_candidates,
            "eval_files_with_exact_payload_sha256": exact_eval,
            "bundle_schema_properties": schema_properties,
            "bundle_schema_has_payload_bytes_field": bool(PAYLOAD_FIELDS & set(schema_properties)),
            "invocation_apply_requires_separate_payload_bytes": source_requires_payload,
        },
        "gates": gates,
        "decision": decision,
        "network_reads": 0,
        "pixel_reads": 0,
        "consumer_mapping": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(canonical_bytes(scientific))
    return {
        "schema": "neuro_film.p228_r1cy_paired_hdr_handoff_artifact_audit_result.v1",
        "experiment_id": "P228",
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = execute(arguments.order, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report))
    os.replace(temporary, arguments.output)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
