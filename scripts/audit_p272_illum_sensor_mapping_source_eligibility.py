#!/usr/bin/env python3
"""Audit the official controlled illumination/sensor RAW source before data access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

REPORT_SCHEMA = "neuro-film.p272-illum-sensor-mapping-source-eligibility.v1"


class P272Error(RuntimeError):
    """Raised when the frozen audit cannot be executed exactly."""


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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(repo: Path, revision_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", revision_path],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _remove_clone(path: Path) -> None:
    """Remove a Windows clone whose pack files may retain read-only attributes."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            os.chmod(child, stat.S_IWRITE)
        except OSError:
            pass
    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path, ignore_errors=False)


def execute(config_path: Path, order: str) -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise P272Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_DATASET_OBJECT_REQUEST":
        raise P272Error("P272 is not frozen before data access")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p272-"))
    source = temporary / "source"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                config["default_branch"],
                config["official_repo"],
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        head = _git(source, "rev-parse", "HEAD")
        tree = _git(source, "rev-parse", "HEAD^{tree}")
        paths = _git(source, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        names = list(config["artifacts"])
        if order == "reverse":
            names.reverse()
        artifacts: dict[str, dict[str, object]] = {}
        texts: dict[str, str] = {}
        for name in names:
            binding = config["artifacts"][name]
            value = _git_bytes(source, f"HEAD:{binding['path']}")
            observed = {
                "bytes": len(value),
                "git_blob": _git_blob_sha1(value),
                "sha256": _sha256(value),
            }
            if observed != {key: binding[key] for key in observed}:
                raise P272Error(f"artifact identity differs: {binding['path']}")
            artifacts[name] = {"path": binding["path"], **observed}
            texts[name] = value.decode("utf-8")

        readme = texts["readme"]
        license_text = texts["license"]
        declared = config["declared_dataset"]
        structure_checks = {
            "camera-count-exact": all(camera in readme for camera in declared["cameras"]),
            "dataset-link-exact": declared["link"] in readme,
            "illuminant-counts-exact": all(
                f"- {declared['illuminants'][role]} for {role}" in readme
                for role in ("train", "validation", "test")
            ),
            "scene-counts-exact": all(
                phrase in readme
                for phrase in (
                    "- 1 training scene",
                    "- 12 testing scenes",
                    "- 1 X-Rite color chart scene",
                    "- 4 custom color chart scenes",
                )
            ),
        }
        rights = {
            "cc-by-nc-sa-4.0-exact": "Attribution-NonCommercial-ShareAlike 4.0 International"
            in license_text,
            "commercial-use-authorized": False,
            "noncommercial-restriction-explicit": "for NonCommercial purposes only"
            in license_text,
        }
        reproducibility = {
            "dataset-archive-size-published": False,
            "dataset-asset-manifest-published": False,
            "dataset-checksums-published": False,
            "repository-tree-only-readme-license": sorted(paths)
            == ["LICENSE.md", "README.md"],
        }
        gates = {
            "official-identity": head == config["head_commit"]
            and tree == config["tree"],
            "materially-new-controlled-physical-observation": all(
                structure_checks.values()
            ),
            "explicit-group-roles": structure_checks["illuminant-counts-exact"]
            and structure_checks["scene-counts-exact"],
            "exact-dataset-manifest-sizes-checksums": all(
                reproducibility[key]
                for key in (
                    "dataset-archive-size-published",
                    "dataset-asset-manifest-published",
                    "dataset-checksums-published",
                )
            ),
            "anonymous-reproducible-access": False,
            "commercial-product-research-rights": rights[
                "commercial-use-authorized"
            ],
            "zero-dataset-pixel-model-requests-before-admission": True,
        }
        scientific = {
            "artifacts": dict(sorted(artifacts.items())),
            "consumer_config_sha256": _sha256(config_bytes),
            "dataset_object_requests": 0,
            "fit_train_inference_score_runs": 0,
            "gates": gates,
            "official_head": head,
            "official_tree": tree,
            "pixel_reads": 0,
            "reproducibility": reproducibility,
            "rights": rights,
            "source_repository_requests": 1,
            "structure_checks": structure_checks,
        }
    finally:
        _remove_clone(temporary)

    scientific["zero_temporary_residue"] = not temporary.exists()
    status = (
        "PASS_SOURCE_ELIGIBILITY"
        if all(scientific["gates"].values())
        else "FAIL_CLOSED_SOURCE_RIGHTS_AND_MANIFEST"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scientific": scientific,
        "stable_identity": f"sha256:{_sha256(_canonical(scientific))}",
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve(), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
