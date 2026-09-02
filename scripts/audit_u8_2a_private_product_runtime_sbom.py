#!/usr/bin/env python3
"""Formal U8.2A private installed-runtime SBOM audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_runtime_sbom import (
    SbomValidationError,
    build_sbom_documents,
    collect_installed_inventory,
    inventory_digest,
    publish_sbom_documents,
    validate_sbom_documents,
)

CONFIG = ROOT / "configs/u8_2a_private_product_runtime_sbom_v1.json"
SOURCE_PATHS = (
    Path("configs/u8_2a_private_product_runtime_sbom_v1.json"),
    Path("docs/planning/U8_2A_PRIVATE_PRODUCT_RUNTIME_SBOM_CONTRACT.md"),
    Path("scripts/audit_u8_2a_private_product_runtime_sbom.py"),
    Path("scripts/build_private_product_runtime_sbom.py"),
    Path("src/inference/product_runtime_sbom.py"),
    Path("tests/test_u8_2a_private_product_runtime_sbom.py"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def _tracked_clean() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no")


def _git_objects(commit: str) -> dict[str, str]:
    return {
        path.as_posix(): _git("rev-parse", f"{commit}:{path.as_posix()}")
        for path in SOURCE_PATHS
    }


def _fetch_official_schema(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = urllib.request.Request(
        row["url"],
        headers={"User-Agent": "K-MCFM-SBOM-Audit/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    identity = {
        "url": row["url"],
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }
    if identity != row:
        raise RuntimeError("official schema identity drift")
    schema = json.loads(payload)
    jsonschema.validators.validator_for(schema).check_schema(schema)
    return schema, identity


def _schema_validate(instance: bytes, schema: dict[str, Any]) -> None:
    jsonschema.validate(json.loads(instance), schema)


def _remove_owned_directory(path: Path, names: tuple[str, ...]) -> None:
    for name in names:
        candidate = path / name
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def _create_only_controls(
    *,
    root: Path,
    cdx_name: str,
    spdx_name: str,
    cdx: bytes,
    spdx: bytes,
) -> dict[str, bool]:
    normal = root / "normal"
    foreign = root / "foreign"
    normal.mkdir(parents=True)
    foreign.mkdir()
    foreign_spdx = foreign / spdx_name
    foreign_spdx.write_bytes(b"foreign-u8-2a")
    try:
        paths = publish_sbom_documents(
            output_directory=normal,
            cyclonedx_name=cdx_name,
            spdx_name=spdx_name,
            cyclonedx_bytes=cdx,
            spdx_bytes=spdx,
        )
        normal_exact = paths[0].read_bytes() == cdx and paths[1].read_bytes() == spdx
        repeat_rejected = False
        try:
            publish_sbom_documents(
                output_directory=normal,
                cyclonedx_name=cdx_name,
                spdx_name=spdx_name,
                cyclonedx_bytes=cdx,
                spdx_bytes=spdx,
            )
        except FileExistsError:
            repeat_rejected = paths[0].read_bytes() == cdx and paths[1].read_bytes() == spdx
        foreign_rejected = False
        try:
            publish_sbom_documents(
                output_directory=foreign,
                cyclonedx_name=cdx_name,
                spdx_name=spdx_name,
                cyclonedx_bytes=cdx,
                spdx_bytes=spdx,
            )
        except FileExistsError:
            foreign_rejected = (
                foreign_spdx.read_bytes() == b"foreign-u8-2a"
                and not (foreign / cdx_name).exists()
            )
        stage_zero = not list(root.rglob(".u8-2a-*.stage"))
        return {
            "normal_create_only_exact": normal_exact,
            "repeat_destination_preserved": repeat_rejected,
            "late_foreign_destination_preserved": foreign_rejected,
            "stage_residue_zero": stage_zero,
        }
    finally:
        _remove_owned_directory(normal, (cdx_name, spdx_name))
        _remove_owned_directory(foreign, (cdx_name, spdx_name))
        try:
            root.rmdir()
        except FileNotFoundError:
            pass


def _report(order: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text("utf-8"))
    start_commit = _git("rev-parse", "HEAD")
    if not _tracked_clean():
        raise RuntimeError("formal audit requires a tracked-clean worktree")
    source_objects = _git_objects(start_commit)
    receipt_path = ROOT / config["input"]["receipt"]
    receipt, receipt_sha256, first = collect_installed_inventory(
        receipt_path,
        expected_sha256=config["input"]["receipt_sha256"],
    )
    _, second_receipt_sha256, second = collect_installed_inventory(
        receipt_path,
        expected_sha256=config["input"]["receipt_sha256"],
    )
    probe_exact = receipt_sha256 == second_receipt_sha256 and first == second
    ordered = first if order == "forward" else list(reversed(first))
    cdx, spdx = build_sbom_documents(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        inventory=ordered,
        creation_time=config["canonical_creation_time"],
    )
    validate_sbom_documents(cdx, spdx)
    cdx_schema, cdx_schema_identity = _fetch_official_schema(
        config["official_schemas"]["cyclonedx"]
    )
    spdx_schema, spdx_schema_identity = _fetch_official_schema(
        config["official_schemas"]["spdx"]
    )
    _schema_validate(cdx, cdx_schema)
    _schema_validate(spdx, spdx_schema)
    control_root = ROOT / "tmp" / f"u8_2a_formal_{order}"
    if control_root.exists():
        raise RuntimeError("owned formal control root already exists")
    create_only = _create_only_controls(
        root=control_root,
        cdx_name=config["output"]["cyclonedx"],
        spdx_name=config["output"]["spdx"],
        cdx=cdx,
        spdx=spdx,
    )
    end_commit = _git("rev-parse", "HEAD")
    source_stable = (
        end_commit == start_commit
        and _tracked_clean()
        and _git_objects(end_commit) == source_objects
    )
    installed = {row["normalized_name"]: row["version"] for row in first}
    receipt_versions = {
        name.replace("_", "-").replace(".", "-").lower(): version
        for name, version in receipt["distributions"].items()
    }
    gates = {
        "receipt_identity_exact": receipt_sha256
        == config["input"]["receipt_sha256"],
        "receipt_schema_exact": receipt["schema"]
        == config["input"]["receipt_schema"],
        "source_and_requirements_exact": receipt["source_commit"]
        == config["input"]["source_commit"]
        and receipt["requirements"]["sha256"]
        == config["input"]["requirements_sha256"],
        "fresh_inventory_probe_exact": probe_exact,
        "installed_inventory_complete": len(first) == 13,
        "receipt_versions_exact": all(
            installed.get(name) == version for name, version in receipt_versions.items()
        ),
        "official_cyclonedx_schema_exact": cdx_schema_identity
        == config["official_schemas"]["cyclonedx"],
        "official_spdx_schema_exact": spdx_schema_identity
        == config["official_schemas"]["spdx"],
        "official_schema_validation_pass": True,
        "cross_format_structural_validation_pass": True,
        "project_license_noassertion": (
            next(
                package
                for package in json.loads(spdx)["packages"]
                if package["name"] == "K-MCFM"
            )["licenseDeclared"]
            == "NOASSERTION"
        ),
        "private_look_approximation_claim_exact": receipt["claim"]
        == {
            "calibrated_stock_response": False,
            "evidence_grade": "look-approximation",
            "mode": "film-inspired",
            "physical_film_reproduction": False,
            "public_release": False,
        },
        **create_only,
        "source_commit_and_objects_stable": source_stable,
    }
    if not all(gates.values()):
        raise SbomValidationError("formal U8.2A gate failed")
    return {
        "schema": "kmcfm.u8-2a-private-product-runtime-sbom-formal-report.v1",
        "status": config["claim_ceiling"]["status"],
        "source_commit": start_commit,
        "source_git_objects": source_objects,
        "input": {
            "receipt": config["input"]["receipt"],
            "receipt_sha256": receipt_sha256,
            "runtime_source_commit": receipt["source_commit"],
            "requirements_sha256": receipt["requirements"]["sha256"],
        },
        "inventory": {
            "count": len(first),
            "sha256": inventory_digest(first),
            "distributions": [
                {
                    "name": row["name"],
                    "normalized_name": row["normalized_name"],
                    "version": row["version"],
                    "metadata_sha256": row["metadata_sha256"],
                    "license_expression": row["license_expression"],
                    "license_file_count": len(row["license_files"]),
                }
                for row in first
            ],
        },
        "documents": {
            "cyclonedx": {"bytes": len(cdx), "sha256": _sha256_bytes(cdx)},
            "spdx": {"bytes": len(spdx), "sha256": _sha256_bytes(spdx)},
        },
        "official_schemas": {
            "cyclonedx": cdx_schema_identity,
            "spdx": spdx_schema_identity,
        },
        "gates": gates,
        "claim": config["claim_ceiling"],
        "reads": {
            "official_schema_requests": 2,
            "media": 0,
            "pixels": 0,
            "models": 0,
        },
    }


def main() -> int:
    arguments = _arguments()
    report = _report(arguments.order)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
