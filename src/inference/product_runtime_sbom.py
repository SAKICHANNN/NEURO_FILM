"""Deterministic SBOM generation for one installed private product runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    publish_create_only,
    remove_if_published,
)

_RECEIPT_SCHEMA = "kmcfm.private-product-runtime-receipt.v3"
_CDX_SPEC = "1.7"
_SPDX_SPEC = "SPDX-2.3"
_APPLICATION_NAME = "K-MCFM"
_TOOL_NAME = "kmcfm-u8.2a-private-runtime-sbom"
_DRIVE_PATH = re.compile(r"(?i)\b[a-z]:\\")
_NORMALIZE_NAME = re.compile(r"[-_.]+")


class SbomValidationError(ValueError):
    """Raised when an input or generated SBOM violates the frozen contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _normalized_name(value: str) -> str:
    name = _NORMALIZE_NAME.sub("-", value).lower()
    if not name or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise SbomValidationError(f"invalid distribution name: {value!r}")
    return name


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


_INVENTORY_PROBE = r'''
import hashlib
import importlib.metadata as metadata
import json
import re
from pathlib import Path

from packaging.requirements import Requirement

normalize = re.compile(r"[-_.]+")

def normalized(value):
    return normalize.sub("-", value).lower()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

rows = []
for dist in metadata.distributions():
    meta = dist.metadata
    name = meta.get("Name")
    if not name:
        raise RuntimeError("installed distribution has no Name")
    base = Path(dist._path).resolve(strict=True)
    metadata_path = base / "METADATA"
    if not metadata_path.is_file():
        raise RuntimeError(f"missing METADATA for {name}")
    license_paths = set()
    licenses_root = base / "licenses"
    if licenses_root.is_dir():
        license_paths.update(path for path in licenses_root.rglob("*") if path.is_file())
    for relative in meta.get_all("License-File") or []:
        candidate = (licenses_root / relative).resolve(strict=False)
        if candidate.is_file() and candidate.is_relative_to(base):
            license_paths.add(candidate)
    dependencies = []
    for raw in meta.get_all("Requires-Dist") or []:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependencies.append(normalized(requirement.name))
    rows.append({
        "name": name,
        "normalized_name": normalized(name),
        "version": dist.version,
        "metadata_version": meta.get("Metadata-Version"),
        "license_expression": meta.get("License-Expression"),
        "license_raw": meta.get("License"),
        "classifiers": sorted(meta.get_all("Classifier") or []),
        "metadata_sha256": sha256(metadata_path),
        "license_files": [
            {
                "path": path.relative_to(base).as_posix(),
                "sha256": sha256(path),
            }
            for path in sorted(license_paths)
        ],
        "dependencies": sorted(set(dependencies)),
    })
print(json.dumps(sorted(rows, key=lambda row: row["normalized_name"]), sort_keys=True))
'''


def _validated_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str, Path, Path]:
    receipt_path = receipt_path.resolve(strict=True)
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha256 = _sha256_bytes(receipt_bytes)
    if expected_sha256 is not None and receipt_sha256 != expected_sha256.lower():
        raise SbomValidationError("runtime receipt SHA-256 mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise SbomValidationError("unsupported runtime receipt schema")
    installation = receipt_path.parent.resolve(strict=True)
    runtime_python = Path(receipt["python"]["executable"])
    if not _contained(runtime_python, installation):
        raise SbomValidationError("runtime Python is outside the installation")
    return receipt, receipt_sha256, installation, runtime_python.resolve(strict=True)


def _validate_inventory(
    inventory: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required_keys = {
        "name",
        "normalized_name",
        "version",
        "metadata_version",
        "license_expression",
        "license_raw",
        "classifiers",
        "metadata_sha256",
        "license_files",
        "dependencies",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in inventory:
        if set(source) != required_keys:
            raise SbomValidationError("installed inventory field mismatch")
        row = dict(source)
        normalized = _normalized_name(str(row["name"]))
        if row["normalized_name"] != normalized or normalized in seen:
            raise SbomValidationError("duplicate or non-canonical distribution name")
        seen.add(normalized)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["metadata_sha256"])):
            raise SbomValidationError("invalid METADATA SHA-256")
        for item in row["license_files"]:
            if set(item) != {"path", "sha256"}:
                raise SbomValidationError("invalid licence-file identity")
            if Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise SbomValidationError("unsafe licence-file path")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise SbomValidationError("invalid licence-file SHA-256")
        row["classifiers"] = sorted(set(map(str, row["classifiers"])))
        row["dependencies"] = sorted(set(map(str, row["dependencies"])))
        row["license_files"] = sorted(
            (dict(item) for item in row["license_files"]),
            key=lambda item: item["path"],
        )
        rows.append(row)
    rows.sort(key=lambda row: row["normalized_name"])
    installed = {row["normalized_name"]: row["version"] for row in rows}
    expected = {
        _normalized_name(name): version
        for name, version in receipt["distributions"].items()
    }
    for name, version in expected.items():
        if installed.get(name) != version:
            raise SbomValidationError(f"receipt distribution mismatch: {name}")
    for row in rows:
        missing = sorted(set(row["dependencies"]) - set(installed))
        if missing:
            raise SbomValidationError(
                f"active dependency missing for {row['normalized_name']}: {missing}"
            )
    return rows


def collect_installed_inventory(
    receipt_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Probe the exact receipt-bound runtime and return its installed inventory."""

    receipt, receipt_sha256, installation, runtime_python = _validated_receipt(
        receipt_path,
        expected_sha256=expected_sha256,
    )
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.upper().startswith("PYTHON"):
            environment.pop(name)
    completed = subprocess.run(
        [str(runtime_python), "-I", "-c", _INVENTORY_PROBE],
        cwd=installation,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if completed.returncode:
        raise SbomValidationError(
            f"installed-runtime inventory probe failed: {completed.stderr.strip()}"
        )
    inventory = _validate_inventory(json.loads(completed.stdout), receipt)
    return receipt, receipt_sha256, inventory


def _component_properties(row: Mapping[str, Any]) -> list[dict[str, str]]:
    properties = [
        {"name": "kmcfm:metadata-version", "value": str(row["metadata_version"])},
        {"name": "kmcfm:metadata-sha256", "value": str(row["metadata_sha256"])},
    ]
    if row["license_raw"]:
        properties.append(
            {"name": "kmcfm:license-raw", "value": str(row["license_raw"])}
        )
    for classifier in row["classifiers"]:
        if str(classifier).startswith("License ::"):
            properties.append(
                {"name": "kmcfm:license-classifier", "value": str(classifier)}
            )
    for item in row["license_files"]:
        properties.append(
            {
                "name": f"kmcfm:license-file-sha256:{item['path']}",
                "value": item["sha256"],
            }
        )
    return sorted(properties, key=lambda item: (item["name"], item["value"]))


def _package_ref(normalized_name: str, version: str) -> str:
    return f"pkg:pypi/{normalized_name}@{version}"


def _spdx_id(normalized_name: str) -> str:
    return "SPDXRef-Package-" + normalized_name


def build_sbom_documents(
    *,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    inventory: Sequence[Mapping[str, Any]],
    creation_time: str,
) -> tuple[bytes, bytes]:
    """Build canonical CycloneDX 1.7 and SPDX 2.3 JSON documents."""

    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise SbomValidationError("unsupported runtime receipt schema")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
        raise SbomValidationError("invalid receipt SHA-256")
    claim = receipt.get("claim", {})
    expected_claim = {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
        "public_release": False,
    }
    if any(claim.get(key) != value for key, value in expected_claim.items()):
        raise SbomValidationError("runtime claim exceeds private Look Approximation")
    rows = _validate_inventory(inventory, receipt)
    source_commit = str(receipt["source_commit"])
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise SbomValidationError("invalid source commit")
    python_version = str(receipt["python"]["version"])
    app_ref = f"urn:kmcfm:private-runtime:{receipt_sha256}"
    python_ref = f"pkg:generic/cpython@{python_version}"
    required_names = sorted(
        _normalized_name(name) for name in receipt["distributions"]
    )
    row_by_name = {row["normalized_name"]: row for row in rows}

    app_component = {
        "type": "application",
        "bom-ref": app_ref,
        "name": _APPLICATION_NAME,
        "version": source_commit,
        "licenses": [],
        "properties": [
            {"name": "kmcfm:calibrated-stock-response", "value": "false"},
            {"name": "kmcfm:evidence-grade", "value": "look-approximation"},
            {"name": "kmcfm:license-status", "value": "unresolved"},
            {"name": "kmcfm:mode", "value": "film-inspired"},
            {"name": "kmcfm:physical-film-reproduction", "value": "false"},
            {"name": "kmcfm:public-release", "value": "false"},
            {"name": "kmcfm:runtime-receipt-sha256", "value": receipt_sha256},
        ],
    }
    components: list[dict[str, Any]] = [
        {
            "type": "framework",
            "bom-ref": python_ref,
            "name": "CPython",
            "version": python_version,
            "licenses": [],
            "properties": [
                {"name": "kmcfm:license-status", "value": "NOASSERTION"}
            ],
        }
    ]
    for row in rows:
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": _package_ref(row["normalized_name"], row["version"]),
            "name": row["name"],
            "version": row["version"],
            "purl": _package_ref(row["normalized_name"], row["version"]),
            "licenses": (
                [{"expression": row["license_expression"]}]
                if row["license_expression"]
                else []
            ),
            "properties": _component_properties(row),
        }
        components.append(component)
    dependencies = [
        {
            "ref": app_ref,
            "dependsOn": [python_ref]
            + [
                _package_ref(name, row_by_name[name]["version"])
                for name in required_names
            ],
        },
        {"ref": python_ref, "dependsOn": []},
    ]
    for row in rows:
        dependencies.append(
            {
                "ref": _package_ref(row["normalized_name"], row["version"]),
                "dependsOn": [
                    _package_ref(name, row_by_name[name]["version"])
                    for name in row["dependencies"]
                ],
            }
        )
    cdx = {
        "bomFormat": "CycloneDX",
        "specVersion": _CDX_SPEC,
        "serialNumber": "urn:uuid:"
        + str(uuid.uuid5(uuid.NAMESPACE_URL, "kmcfm-u8.2a:" + receipt_sha256)),
        "version": 1,
        "metadata": {
            "timestamp": creation_time,
            "component": app_component,
        },
        "components": sorted(components, key=lambda item: item["bom-ref"]),
        "dependencies": sorted(dependencies, key=lambda item: item["ref"]),
    }

    packages: list[dict[str, Any]] = [
        {
            "name": _APPLICATION_NAME,
            "SPDXID": "SPDXRef-Package-K-MCFM",
            "versionInfo": source_commit,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "comment": json.dumps(
                {
                    "mode": "film-inspired",
                    "evidence_grade": "look-approximation",
                    "public_release": False,
                    "license_status": "unresolved",
                    "runtime_receipt_sha256": receipt_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "name": "CPython",
            "SPDXID": "SPDXRef-Package-CPython",
            "versionInfo": python_version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        },
    ]
    for row in rows:
        package = {
            "name": row["name"],
            "SPDXID": _spdx_id(row["normalized_name"]),
            "versionInfo": row["version"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": row["license_expression"] or "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": _package_ref(
                        row["normalized_name"], row["version"]
                    ),
                }
            ],
            "comment": json.dumps(
                {
                    "metadata_version": row["metadata_version"],
                    "metadata_sha256": row["metadata_sha256"],
                    "license_raw": row["license_raw"],
                    "license_classifiers": [
                        value
                        for value in row["classifiers"]
                        if value.startswith("License ::")
                    ],
                    "license_files": row["license_files"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        packages.append(package)
    relationships = [
        {
            "spdxElementId": "SPDXRef-Package-K-MCFM",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": "SPDXRef-Package-CPython",
        }
    ]
    for name in required_names:
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package-K-MCFM",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": _spdx_id(name),
            }
        )
    for row in rows:
        for dependency in row["dependencies"]:
            relationships.append(
                {
                    "spdxElementId": _spdx_id(row["normalized_name"]),
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": _spdx_id(dependency),
                }
            )
    spdx = {
        "spdxVersion": _SPDX_SPEC,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"K-MCFM-private-runtime-{receipt_sha256[:12]}",
        "documentNamespace": (
            "https://kmcfm.invalid/spdx/u8-2a/" + receipt_sha256
        ),
        "creationInfo": {
            "created": creation_time,
            "creators": [f"Tool: {_TOOL_NAME}/1"],
        },
        "documentDescribes": ["SPDXRef-Package-K-MCFM"],
        "packages": sorted(packages, key=lambda item: item["SPDXID"]),
        "relationships": sorted(
            relationships,
            key=lambda item: (
                item["spdxElementId"],
                item["relationshipType"],
                item["relatedSpdxElement"],
            ),
        ),
    }
    documents = (_canonical_json(cdx), _canonical_json(spdx))
    validate_sbom_documents(*documents)
    return documents


def _parsed_canonical(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SbomValidationError("SBOM is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise SbomValidationError("SBOM canonical JSON mismatch")
    if _DRIVE_PATH.search(payload.decode("utf-8")):
        raise SbomValidationError("SBOM leaks a host drive path")
    return value


def validate_sbom_documents(cyclonedx_bytes: bytes, spdx_bytes: bytes) -> None:
    """Validate frozen structural and cross-format U8.2A invariants."""

    cdx = _parsed_canonical(cyclonedx_bytes)
    spdx = _parsed_canonical(spdx_bytes)
    if cdx.get("bomFormat") != "CycloneDX" or cdx.get("specVersion") != _CDX_SPEC:
        raise SbomValidationError("CycloneDX identity mismatch")
    if cdx.get("version") != 1 or not str(cdx.get("serialNumber", "")).startswith(
        "urn:uuid:"
    ):
        raise SbomValidationError("CycloneDX document identity invalid")
    app = cdx.get("metadata", {}).get("component", {})
    if app.get("name") != _APPLICATION_NAME or app.get("licenses") != []:
        raise SbomValidationError("CycloneDX application licence is not unresolved")
    app_properties = {
        item["name"]: item["value"] for item in app.get("properties", [])
    }
    if app_properties.get("kmcfm:public-release") != "false" or app_properties.get(
        "kmcfm:evidence-grade"
    ) != "look-approximation":
        raise SbomValidationError("CycloneDX private claim mismatch")
    components = cdx.get("components")
    dependencies = cdx.get("dependencies")
    if not isinstance(components, list) or not isinstance(dependencies, list):
        raise SbomValidationError("CycloneDX inventory missing")
    cdx_refs = {app["bom-ref"]} | {item["bom-ref"] for item in components}
    if len(cdx_refs) != len(components) + 1:
        raise SbomValidationError("CycloneDX duplicate component reference")
    if {item["ref"] for item in dependencies} != cdx_refs:
        raise SbomValidationError("CycloneDX dependency subjects incomplete")
    if any(set(item["dependsOn"]) - cdx_refs for item in dependencies):
        raise SbomValidationError("CycloneDX dependency target missing")

    required_spdx = {
        "spdxVersion",
        "dataLicense",
        "SPDXID",
        "name",
        "documentNamespace",
        "creationInfo",
        "documentDescribes",
        "packages",
        "relationships",
    }
    if set(spdx) != required_spdx:
        raise SbomValidationError("SPDX top-level field mismatch")
    if (
        spdx["spdxVersion"] != _SPDX_SPEC
        or spdx["dataLicense"] != "CC0-1.0"
        or spdx["SPDXID"] != "SPDXRef-DOCUMENT"
    ):
        raise SbomValidationError("SPDX document identity mismatch")
    packages = spdx["packages"]
    spdx_ids = {item["SPDXID"] for item in packages}
    if len(spdx_ids) != len(packages):
        raise SbomValidationError("SPDX duplicate package identifier")
    root = next(
        (item for item in packages if item["SPDXID"] == "SPDXRef-Package-K-MCFM"),
        None,
    )
    if root is None or root["licenseDeclared"] != "NOASSERTION":
        raise SbomValidationError("SPDX project licence must remain NOASSERTION")
    for package in packages:
        if package.get("filesAnalyzed") is not False:
            raise SbomValidationError("SPDX package filesAnalyzed drift")
        for key in (
            "name",
            "SPDXID",
            "versionInfo",
            "downloadLocation",
            "licenseConcluded",
            "licenseDeclared",
            "copyrightText",
        ):
            if key not in package:
                raise SbomValidationError(f"SPDX package field missing: {key}")
    for relationship in spdx["relationships"]:
        if relationship["relationshipType"] != "DEPENDS_ON":
            raise SbomValidationError("SPDX relationship type drift")
        if relationship["spdxElementId"] not in spdx_ids:
            raise SbomValidationError("SPDX dependency subject missing")
        if relationship["relatedSpdxElement"] not in spdx_ids:
            raise SbomValidationError("SPDX dependency target missing")

    cdx_identities = {(app["name"], app["version"])} | {
        (item["name"], item["version"]) for item in components
    }
    spdx_identities = {(item["name"], item["versionInfo"]) for item in packages}
    if cdx_identities != spdx_identities:
        raise SbomValidationError("cross-format component identity mismatch")


def publish_sbom_documents(
    *,
    output_directory: Path,
    cyclonedx_name: str,
    spdx_name: str,
    cyclonedx_bytes: bytes,
    spdx_bytes: bytes,
) -> tuple[Path, Path]:
    """Publish both documents create-only and roll back only owned output."""

    validate_sbom_documents(cyclonedx_bytes, spdx_bytes)
    output_directory.mkdir(parents=True, exist_ok=True)
    if Path(cyclonedx_name).name != cyclonedx_name or Path(spdx_name).name != spdx_name:
        raise SbomValidationError("SBOM output names must be basenames")
    destinations = (
        output_directory / cyclonedx_name,
        output_directory / spdx_name,
    )
    stages = (
        output_directory / f".u8-2a-{os.getpid()}-cyclonedx.stage",
        output_directory / f".u8-2a-{os.getpid()}-spdx.stage",
    )
    published: list[PublishedFileIdentity] = []
    try:
        for stage, payload in zip(stages, (cyclonedx_bytes, spdx_bytes), strict=True):
            with stage.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for stage, destination in zip(stages, destinations, strict=True):
            published.append(publish_create_only(stage, destination))
    except BaseException:
        for identity in reversed(published):
            remove_if_published(identity)
        raise
    finally:
        for stage in stages:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass
    return destinations


def inventory_digest(inventory: Iterable[Mapping[str, Any]]) -> str:
    """Return the canonical SHA-256 of an installed inventory."""

    rows = sorted((dict(row) for row in inventory), key=lambda row: row["normalized_name"])
    return _sha256_bytes(_canonical_json({"inventory": rows}))


__all__ = [
    "SbomValidationError",
    "build_sbom_documents",
    "collect_installed_inventory",
    "inventory_digest",
    "publish_sbom_documents",
    "validate_sbom_documents",
]
