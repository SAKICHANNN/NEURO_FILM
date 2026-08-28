#!/usr/bin/env python3
"""Audit the exact source-locked R1FS DNG ImageSequenceInfo handoff."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "neuro-film.p309-r1fs-dng-image-sequence-no-copy-intake.v1"


class P309Error(RuntimeError):
    """Raised when a frozen P309 identity or execution gate fails."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_artifact(repo: Path, commit: str, binding: dict[str, Any]) -> bytes:
    value = _git_bytes(repo, commit, binding["path"])
    observed = {
        "bytes": len(value),
        "git_blob": _git_blob(repo, commit, binding["path"]),
        "sha256": _sha256_bytes(value),
    }
    if observed != {key: binding[key] for key in observed}:
        raise P309Error(f"producer artifact differs: {binding['path']}")
    return value


def _load_isolated_module(site: Path, module_bytes: bytes) -> Any:
    package = site / "zhuise"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "dng_image_sequence.py").write_bytes(module_bytes)
    sys.path.insert(0, str(site))
    try:
        return importlib.import_module("zhuise.dng_image_sequence")
    finally:
        sys.path.remove(str(site))


def _expect_value_error(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _frame_summary(frame: Any) -> dict[str, Any]:
    return {
        "source_bytes": frame.source_bytes,
        "source_sha256": frame.source_sha256,
        "payload_sha256": frame.payload_sha256,
        "sequence_id": frame.sequence_id,
        "sequence_type": frame.sequence_type,
        "frame_info": frame.frame_info,
        "index": frame.index,
        "count": frame.count,
        "is_final": frame.is_final,
    }


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P309Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_PRODUCER_OBJECT_IMPORT_OR_SOURCE_METADATA_READ":
        raise P309Error("P309 is not source locked")

    producer = config["producer"]
    artifact_bytes: dict[str, bytes] = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, binding in config["artifacts"].items():
        commit = producer[binding["commit_role"]]
        value = _verify_artifact(producer_repo, commit, binding)
        artifact_bytes[name] = value
        verified[name] = {
            "path": binding["path"],
            "bytes": len(value),
            "git_blob": binding["git_blob"],
            "sha256": _sha256_bytes(value),
        }

    evidence = json.loads(artifact_bytes["evidence"])
    source_paths = [producer_repo / row["path"] for row in config["sources"]]
    source_before = []
    for path, binding in zip(source_paths, config["sources"], strict=True):
        observed = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        expected = {key: binding[key] for key in observed}
        if observed != expected:
            raise P309Error(f"producer source differs: {binding['path']}")
        source_before.append(observed)

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p309-"))
    module_name = "zhuise.dng_image_sequence"
    report: dict[str, Any] | None = None
    try:
        module = _load_isolated_module(temporary / "site", artifact_bytes["module"])
        isolated_import = Path(module.__file__).resolve().is_relative_to(
            (temporary / "site").resolve()
        )
        names = [
            "main",
            "reverse-source-order",
            "empty",
            "missing",
            "duplicate",
            "non-contiguous",
            "cross-sequence",
            "wrong-final",
            "malformed-short",
            "malformed-unterminated",
            "malformed-trailing",
            "malformed-final",
        ]
        if order == "reverse":
            names.reverse()
        controls: dict[str, bool] = {}
        main = None
        parsed_frames = [module.read_dng_image_sequence_info(path) for path in source_paths]
        for name in names:
            if name == "main":
                main = module.assemble_dng_image_sequence(parsed_frames)
                controls[name] = [frame.index for frame in main.frames] == [1, 2, 3]
            elif name == "reverse-source-order":
                reversed_sequence = module.load_dng_image_sequence(reversed(source_paths))
                controls[name] = [frame.index for frame in reversed_sequence.frames] == [1, 2, 3]
            elif name == "empty":
                controls[name] = _expect_value_error(
                    lambda: module.assemble_dng_image_sequence([])
                )
            elif name == "missing":
                controls[name] = _expect_value_error(
                    lambda: module.assemble_dng_image_sequence(parsed_frames[:2])
                )
            elif name == "duplicate":
                controls[name] = _expect_value_error(
                    lambda: module.assemble_dng_image_sequence(
                        [parsed_frames[0], parsed_frames[0], parsed_frames[2]]
                    )
                )
            elif name == "non-contiguous":
                bad = dataclasses.replace(parsed_frames[1], index=3)
                controls[name] = _expect_value_error(
                    lambda bad=bad: module.assemble_dng_image_sequence(
                        [parsed_frames[0], bad, parsed_frames[2]]
                    )
                )
            elif name == "cross-sequence":
                bad = dataclasses.replace(parsed_frames[1], sequence_id="other-sequence")
                controls[name] = _expect_value_error(
                    lambda bad=bad: module.assemble_dng_image_sequence(
                        [parsed_frames[0], bad, parsed_frames[2]]
                    )
                )
            elif name == "wrong-final":
                bad = dataclasses.replace(parsed_frames[2], is_final=False)
                controls[name] = _expect_value_error(
                    lambda bad=bad: module.assemble_dng_image_sequence(
                        [parsed_frames[0], parsed_frames[1], bad]
                    )
                )
            elif name == "malformed-short":
                controls[name] = _expect_value_error(
                    lambda: module.parse_image_sequence_info_payload(b"short")
                )
            elif name == "malformed-unterminated":
                controls[name] = _expect_value_error(
                    lambda: module.parse_image_sequence_info_payload(b"A" * 32)
                )
            elif name == "malformed-trailing":
                payload = b"sequence-id\0type\0info\0" + bytes(10)
                controls[name] = _expect_value_error(
                    lambda payload=payload: module.parse_image_sequence_info_payload(payload)
                )
            elif name == "malformed-final":
                payload = b"sequence-id\0type\0info\0" + (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\x02"
                controls[name] = _expect_value_error(
                    lambda payload=payload: module.parse_image_sequence_info_payload(payload)
                )
        if main is None:
            raise AssertionError("main control did not execute")

        frames = [_frame_summary(frame) for frame in main.frames]
        expected_frames = [
            {
                key: row[key]
                for key in (
                    "source_bytes",
                    "source_sha256",
                    "payload_sha256",
                    "sequence_id",
                    "sequence_type",
                    "frame_info",
                    "index",
                    "count",
                    "is_final",
                )
            }
            for row in [
                {
                    **source,
                    "source_bytes": source["bytes"],
                    "source_sha256": source["sha256"],
                    "sequence_id": config["expected"]["sequence_id"],
                    "sequence_type": config["expected"]["sequence_type"],
                }
                for source in config["sources"]
            ]
        ]
        source_after = [
            {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in source_paths
        ]
        gates = {
            "all-controls-pass": all(controls.values()),
            "artifact-identities-exact": len(verified) == len(config["artifacts"]),
            "canonical-frame-facts-exact": frames == expected_frames,
            "evidence-decision-exact": evidence["decision"]
            == "PASS_PRIVATE_DNG_IMAGE_SEQUENCE_INFO_GROUP",
            "evidence-protocol-exact": evidence["protocol"] == config["protocol"],
            "isolated-git-object-import": isolated_import,
            "immutable-frozen-results": all(
                value.__dataclass_params__.frozen
                for value in (module.DngImageSequenceFrame, module.DngImageSequence)
            ),
            "producer-report-identity-exact": evidence["execution"]["forward_report"]["sha256"]
            == config["expected"]["report_sha256"],
            "producer-scientific-identity-exact": evidence["execution"]["scientific_stable_identity"]
            == f"sha256:{config['expected']['scientific_identity']}",
            "source-files-immutable": source_after == source_before,
            "zero-image-decoder-imports": not any(
                name in sys.modules for name in ("rawpy", "PIL", "cv2", "imageio")
            ),
        }
        scientific = {
            "controls": controls,
            "frames": frames,
            "gates": gates,
            "protocol": config["protocol"],
            "sequence": {
                "count": main.count,
                "sequence_id": main.sequence_id,
                "sequence_type": main.sequence_type,
            },
        }
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "P309",
            "status": "PASS_PRIVATE_R1FS_DNG_IMAGE_SEQUENCE_NO_COPY_INTAKE"
            if all(gates.values())
            else "FAIL_CLOSED_R1FS_DNG_IMAGE_SEQUENCE_NO_COPY_INTAKE",
            "bindings": {
                "artifacts": verified,
                "config": {"bytes": len(config_bytes), "sha256": _sha256_bytes(config_bytes)},
                "producer_head": producer["repo_head"],
            },
            "scientific": scientific,
            "scientific_identity": _sha256_bytes(_canonical(scientific)),
            "rights_and_product": {
                "candidate_3": False,
                "consumer_core_copied": False,
                "image_pixels_decoded": 0,
                "product_mapping": False,
                "public_capability": False,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        if not all(gates.values()):
            raise P309Error("one or more P309 gates failed")
        return report
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop("zhuise", None)
        shutil.rmtree(temporary, ignore_errors=False)
        if temporary.exists():
            raise P309Error("temporary residue remains")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config, args.producer_repo, args.order)
    payload = _canonical(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({"bytes": len(payload), "sha256": _sha256_bytes(payload), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
