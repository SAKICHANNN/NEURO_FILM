#!/usr/bin/env python3
"""Run two committed-head P302 reports and publish bounded formal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P302AuditError(RuntimeError):
    """Raised when a P302 formal execution boundary differs."""


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _json(path: Path) -> tuple[bytes, dict[str, Any]]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise P302AuditError(f"P302 JSON root is invalid: {path}")
    return body, value


def _committed_head() -> str:
    for arguments in (
        ("git", "diff", "--quiet"),
        ("git", "diff", "--cached", "--quiet"),
    ):
        if subprocess.run(arguments, cwd=ROOT, check=False).returncode != 0:
            raise P302AuditError("P302 formal execution requires a clean tracked tree")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_report(
    config: Path,
    destination: Path,
    *,
    phase: str,
    reverse: bool,
    development_report: Path | None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_p302_wildrelight_spatial_envmap_explicit_operator",
        "--config",
        str(config),
        "--output",
        str(destination),
        "--phase",
        phase,
    ]
    if reverse:
        command.append("--reverse")
    if development_report is not None:
        command.extend(("--development-report", str(development_report)))
    subprocess.run(command, cwd=ROOT, check=True)


def build_evidence(
    *,
    config_body: bytes,
    config: dict[str, Any],
    source_lock_body: bytes,
    source_lock: dict[str, Any],
    acquisition_body: bytes,
    acquisition: dict[str, Any],
    report_body: bytes,
    report: dict[str, Any],
    head: str,
    phase: str,
) -> dict[str, Any]:
    manifest_binding = config["source"]["member_manifest"]
    expected_roles = (
        ["training", "development"] if phase == "development" else ["confirmation"]
    )
    gates = {
        "acquisition_config_exact": acquisition.get("config_sha256")
        == _sha256(config_body),
        "acquisition_members_exact": acquisition.get("members_exact") is True,
        "acquisition_role_exact": acquisition.get("roles") == expected_roles,
        "candidate_count_unchanged": report.get("candidate_count") == "2/3",
        "confirmation_unread_during_development": phase != "development"
        or report.get("confirmation_member_reads") == 0,
        "member_manifest_source_lock_exact": source_lock.get("member_manifest_bytes")
        == manifest_binding["bytes"]
        and source_lock.get("member_manifest_sha256") == manifest_binding["sha256"],
        "report_gates_all_pass": all(report.get("gates", {}).values()),
        "report_phase_exact": report.get("phase") == phase,
        "reserve_unread": report.get("reserve_member_reads") == 0,
        "source_lock_no_payload_or_pixels": source_lock.get(
            "archive_or_payload_requests"
        )
        == 0
        and source_lock.get("pixel_decodes") == 0,
    }
    report_pass = report.get("decision") == f"PASS_PRIVATE_P302_{phase.upper()}"
    decision = (
        f"PASS_PRIVATE_P302_{phase.upper()}"
        if report_pass and all(gates.values())
        else f"FAIL_CLOSED_P302_{phase.upper()}"
    )
    return {
        "schema": "neuro-film.p302-wildrelight-spatial-envmap-formal-evidence.v1",
        "experiment_id": "P302",
        "phase": phase,
        "decision": decision,
        "candidate_count": "2/3",
        "execution_commit": head,
        "bindings": {
            "acquisition_report_bytes": len(acquisition_body),
            "acquisition_report_sha256": _sha256(acquisition_body),
            "config_bytes": len(config_body),
            "config_sha256": _sha256(config_body),
            "formal_report_bytes": len(report_body),
            "formal_report_sha256": _sha256(report_body),
            "member_manifest_sha256": manifest_binding["sha256"],
            "source_lock_report_bytes": len(source_lock_body),
            "source_lock_report_sha256": _sha256(source_lock_body),
        },
        "gates": gates,
        "scientific_identity": report.get("scientific_identity"),
        "summary": report.get("summary"),
        "model_bytes": report.get("model_bytes"),
        "model_sha256": report.get("model_sha256"),
        "claim_ceiling": config["claim_ceiling"],
        "boundary": {
            "confirmation_member_reads": report.get("confirmation_member_reads"),
            "reserve_member_reads": report.get("reserve_member_reads"),
            "target_reads_before_model_lock": report.get(
                "target_reads_before_model_lock"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--development-report", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    source_lock_path = args.source_lock.resolve()
    acquisition_path = args.acquisition_report.resolve()
    development_report = (
        args.development_report.resolve() if args.development_report else None
    )
    head = _committed_head()
    config_body, config = _json(config_path)
    source_lock_body, source_lock = _json(source_lock_path)
    acquisition_body, acquisition = _json(acquisition_path)
    raw_directory = args.raw_directory.resolve()
    raw_directory.mkdir(parents=True, exist_ok=True)
    if any(raw_directory.iterdir()) or args.output.exists():
        raise P302AuditError("P302 formal destinations must be create-only and empty")
    forward = raw_directory / f"{args.phase}_forward.json"
    reverse = raw_directory / f"{args.phase}_reverse.json"
    _run_report(
        config_path,
        forward,
        phase=args.phase,
        reverse=False,
        development_report=development_report,
    )
    _run_report(
        config_path,
        reverse,
        phase=args.phase,
        reverse=True,
        development_report=development_report,
    )
    forward_body, report = _json(forward)
    reverse_body = reverse.read_bytes()
    if forward_body != reverse_body:
        raise P302AuditError("P302 forward and reverse reports differ")
    evidence = build_evidence(
        config_body=config_body,
        config=config,
        source_lock_body=source_lock_body,
        source_lock=source_lock,
        acquisition_body=acquisition_body,
        acquisition=acquisition,
        report_body=forward_body,
        report=report,
        head=head,
        phase=args.phase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(_canonical(evidence))


if __name__ == "__main__":
    main()
