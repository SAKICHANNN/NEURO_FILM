"""Acquire only source-locked P302 members without decoding pixels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.film_physics.create_only_file import publish_create_only

ROOT = Path(__file__).resolve().parents[1]


class P302AcquisitionError(RuntimeError):
    """Raised when a P302 acquisition boundary differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _report_identity_valid(report: dict[str, Any]) -> bool:
    claimed = report.get("scientific_identity")
    payload = dict(report)
    payload.pop("scientific_identity", None)
    observed = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return claimed == observed


def _load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    binding = config["source"]["member_manifest"]
    if not isinstance(binding, dict):
        raise P302AcquisitionError("P302 member manifest is not source-locked")
    relative = Path(str(binding["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise P302AcquisitionError("P302 member manifest path is invalid")
    path = ROOT / relative
    body = path.read_bytes()
    if len(body) != int(binding["bytes"]) or hashlib.sha256(body).hexdigest() != str(
        binding["sha256"]
    ):
        raise P302AcquisitionError("P302 member manifest identity differs")
    return json.loads(body)


def _local_root(config: dict[str, Any]) -> Path:
    relative = Path(str(config["source"]["local_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise P302AcquisitionError("P302 local source root is invalid")
    return ROOT / relative


def _local_path(root: Path, remote_path: str) -> Path:
    prefix = "small-aligned/"
    if not remote_path.startswith(prefix):
        raise P302AcquisitionError("P302 member is outside small-aligned")
    relative = Path(remote_path.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise P302AcquisitionError("P302 member path is invalid")
    return root / relative


def _member_state(root: Path, member: dict[str, Any]) -> dict[str, object]:
    path = _local_path(root, str(member["path"]))
    exists = path.is_file()
    size = path.stat().st_size if exists else -1
    digest = _sha256_file(path) if exists and size == int(member["bytes"]) else ""
    return {
        "bytes": size,
        "exists": exists,
        "path": str(member["path"]),
        "role": str(member["role"]),
        "sha256": digest,
        "valid": bool(
            exists and size == int(member["bytes"]) and digest == str(member["sha256"])
        ),
    }


def _acquire_member(
    destination: Path, *, url: str, expected_bytes: int, expected_sha256: str
) -> int:
    """Acquire one frozen member with bounded retries and resumable staging."""
    if destination.exists():
        if (
            destination.is_file()
            and destination.stat().st_size == expected_bytes
            and _sha256_file(destination) == expected_sha256
        ):
            return 0
        raise P302AcquisitionError("existing P302 source member differs")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise P302AcquisitionError("bounded P302 curl transport is unavailable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.p302.partial"
    if stage.exists() and (
        not stage.is_file() or stage.stat().st_size > expected_bytes
    ):
        raise P302AcquisitionError("P302 resumable stage is invalid")
    initial_bytes = stage.stat().st_size if stage.is_file() else 0
    command = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "4",
        "--retry-all-errors",
        "--retry-delay",
        "1",
        "--connect-timeout",
        "20",
        "--speed-limit",
        "1024",
        "--speed-time",
        "30",
        "--max-time",
        "120",
        "--continue-at",
        "-",
        "--user-agent",
        "neuro-film-p302-source-acquisition/2.0",
        "--output",
        str(stage),
        url,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise P302AcquisitionError(
            f"bounded P302 curl transport failed with exit {completed.returncode}"
        )
    if (
        not stage.is_file()
        or stage.stat().st_size != expected_bytes
        or _sha256_file(stage) != expected_sha256
    ):
        raise P302AcquisitionError("downloaded P302 source member differs")
    final_bytes = stage.stat().st_size
    publish_create_only(stage, destination)
    return max(0, final_bytes - initial_bytes)


def _confirmation_allowed(report_path: Path | None, config_sha256: str) -> bool:
    if report_path is None or not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        report.get("decision") == "PASS_PRIVATE_P302_DEVELOPMENT"
        and report.get("config_sha256") == config_sha256
        and _report_identity_valid(report)
        and report.get("phase") == "development"
        and report.get("development_model_match") is True
        and all(report.get("gates", {}).values())
    )


def acquire(
    config_path: Path,
    *,
    roles: tuple[str, ...],
    development_report: Path | None,
) -> dict[str, object]:
    config_body = config_path.read_bytes()
    config = json.loads(config_body)
    config_sha256 = hashlib.sha256(config_body).hexdigest()
    allowed = {"training", "development", "confirmation"}
    if not roles or not set(roles).issubset(allowed) or len(set(roles)) != len(roles):
        raise P302AcquisitionError("P302 acquisition roles are invalid")
    if "confirmation" in roles and not _confirmation_allowed(
        development_report, config_sha256
    ):
        raise P302AcquisitionError(
            "confirmation acquisition is closed before development pass"
        )
    manifest = _load_manifest(config)
    members = [member for member in manifest["members"] if member["role"] in roles]
    root = _local_root(config)
    dataset = str(config["source"]["dataset_id"])
    revision = str(config["source"]["revision"])

    def transfer(member: dict[str, Any]) -> int:
        url = (
            f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
            f"{member['path']}?download=true"
        )
        try:
            return _acquire_member(
                _local_path(root, str(member["path"])),
                url=url,
                expected_bytes=int(member["bytes"]),
                expected_sha256=str(member["sha256"]),
            )
        except Exception as error:
            raise P302AcquisitionError(
                f"P302 source transfer failed: {member['path']}"
            ) from error

    transferred = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for count in executor.map(transfer, members):
            transferred += count
    states = [_member_state(root, member) for member in members]
    report: dict[str, object] = {
        "config_bytes": len(config_body),
        "config_sha256": config_sha256,
        "experiment_id": "P302",
        "member_count": len(members),
        "members_exact": all(state["valid"] for state in states),
        "network_bytes": transferred,
        "pixel_decodes": 0,
        "roles": list(roles),
        "schema": "neuro-film.p302-wildrelight-source-acquisition.v1",
        "states": states,
        "status": "PASS_PRIVATE_P302_SOURCE_ACQUISITION",
    }
    report["source_identity"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", action="append", required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = acquire(
        args.config.resolve(),
        roles=tuple(args.role),
        development_report=(
            args.development_report.resolve() if args.development_report else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
