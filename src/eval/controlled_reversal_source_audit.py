"""U6.P6Y controlled reversal-film public-deposit source audit."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from zipfile import ZipFile, ZipInfo

from pypdf import PdfReader

SCHEMA = "neuro_film.u6_p6y_controlled_reversal_source_audit_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6y_controlled_reversal_source_audit_report.v1"
DECISION = "FAIL_CLOSED_PUBLIC_DEPOSIT_PAPERS_ONLY_NO_CONTROLLED_PIXEL_PAYLOADS"
_COPYRIGHT_ACT = "https://nva.sikt.no/license/copyright-act/1.0"
_PIXEL_SUFFIXES = {
    ".arw",
    ".csv",
    ".dng",
    ".envi",
    ".exr",
    ".h5",
    ".hdf5",
    ".hdr",
    ".jpg",
    ".jpeg",
    ".mat",
    ".npy",
    ".npz",
    ".png",
    ".raw",
    ".tif",
    ".tiff",
}


class ControlledReversalSourceAuditError(ValueError):
    """Raised when the frozen P6Y contract or a retained source drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ControlledReversalSourceAuditError(
            "P6Y source destination must be a bounded relative path"
        )
    return path


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size:
        raise ControlledReversalSourceAuditError(f"P6Y source size drift: {path}")
    if sha256_file(path) != sha256:
        raise ControlledReversalSourceAuditError(f"P6Y source hash drift: {path}")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    amendment = payload.get("execution_amendment", {})
    sources = payload.get("primary_sources", [])
    expected_roles = {
        "2026 open-access journal article",
        "NTNU public thesis record",
        "NTNU thesis PDF",
        "NTNU submission attachments ZIP",
    }
    if (
        payload.get("schema") != SCHEMA
        or {row.get("role") for row in sources} != expected_roles
        or payload.get("acquisition", {}).get("maximum_total_bytes") != 500_000_000
        or not payload.get("acquisition", {}).get("no_other_payloads")
        or amendment.get("thesis_pdf_sha256")
        != "e231da44fc8163559d8ae6d9b4f41cf825b4106a0316aeeb078efecdb43e8a68"
        or amendment.get("thesis_page_count") != 109
        or amendment.get("pdftotext_layout_utf8_sha256")
        != "18849216863a6bfd28aead8eb67c69ae2841ae28d3acfa3b02f2c74dda2859c9"
        or amendment.get("thresholds_or_gates_changed") is not False
        or not payload.get("source_gate", {}).get("two_independent_audits")
        or payload.get("source_gate", {}).get(
            "pixel_decode_operator_fit_render_and_visual_review_allowed"
        )
        is not False
    ):
        raise ControlledReversalSourceAuditError("P6Y frozen contract drift")
    _relative_path(payload["acquisition"]["destination"])
    return payload


def _source_by_role(config: dict[str, Any], role: str) -> dict[str, Any]:
    rows = [row for row in config["primary_sources"] if row["role"] == role]
    if len(rows) != 1:
        raise ControlledReversalSourceAuditError(f"P6Y source role drift: {role}")
    return rows[0]


def audit_record(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = _source_by_role(config, "NTNU public thesis record")
    _verify_file(
        path,
        size=int(source["expected_record_bytes"]),
        sha256=str(source["expected_record_sha256"]),
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    title = record.get("entityDescription", {}).get("mainTitle")
    handles = [
        row.get("value")
        for row in record.get("additionalIdentifiers", [])
        if row.get("type") == "HandleIdentifier"
    ]
    artifacts = []
    for row in record.get("associatedArtifacts", []):
        if row.get("type") != "OpenFile":
            continue
        artifacts.append(
            {
                "identifier": row.get("identifier"),
                "mime_type": row.get("mimeType"),
                "bytes": row.get("size"),
                "license": row.get("license"),
                "allowed_operations": sorted(row.get("allowedOperations", [])),
            }
        )
    artifacts.sort(key=lambda row: str(row["identifier"]))
    declared = {
        (
            row["file_identifier"],
            row["mime_type"],
            int(row["expected_bytes"]),
            row["deposit_license"],
        )
        for row in config["primary_sources"]
        if "file_identifier" in row
    }
    observed = {
        (
            row["identifier"],
            row["mime_type"],
            int(row["bytes"]),
            row["license"],
        )
        for row in artifacts
    }
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "title": title,
        "title_exact": title == source["expected_title"],
        "handle_exact": handles == [source["handle"]],
        "open_artifacts": artifacts,
        "open_artifact_count": len(artifacts),
        "declared_inventory_exact": observed == declared,
        "only_pdf_and_zip": {row["mime_type"] for row in artifacts}
        == {"application/pdf", "application/zip"},
        "explicit_pixel_payload_artifacts": [
            row
            for row in artifacts
            if str(row["mime_type"]).startswith("image/")
            or str(row["mime_type"])
            in {
                "application/x-hdf5",
                "application/x-matlab-data",
                "application/x-netcdf",
            }
        ],
    }


def _safe_member(info: ZipInfo) -> bool:
    path = PurePosixPath(info.filename.replace("\\", "/"))
    mode = info.external_attr >> 16
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
    )


def audit_support_zip(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = _source_by_role(config, "NTNU submission attachments ZIP")
    _verify_file(
        path,
        size=int(source["expected_bytes"]),
        sha256=str(source["expected_sha256"]),
    )
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        members = [
            {
                "name": info.filename,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "safe_relative_non_symlink": _safe_member(info),
            }
            for info in infos
        ]
    pixel_members = [
        row
        for row in members
        if PurePosixPath(str(row["name"])).suffix.casefold() in _PIXEL_SUFFIXES
    ]
    code_members = [
        row
        for row in members
        if PurePosixPath(str(row["name"])).suffix.casefold()
        in {".c", ".cpp", ".h", ".ipynb", ".m", ".py", ".r"}
    ]
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(members),
        "members": members,
        "member_names_unique": len(names) == len(set(names)),
        "all_member_paths_safe": all(_safe_member(info) for info in infos),
        "pixel_payload_members": pixel_members,
        "code_members": code_members,
        "pdf_member_count": sum(
            PurePosixPath(name).suffix.casefold() == ".pdf" for name in names
        ),
        "substantive_pdf_member_count": sum(
            PurePosixPath(name).suffix.casefold() == ".pdf"
            and not name.startswith("__MACOSX/")
            and not PurePosixPath(name).name.startswith("._")
            for name in names
        ),
    }


def _tool_identity(name: str) -> tuple[str, dict[str, str]]:
    executable = shutil.which(name)
    if executable is None:
        raise ControlledReversalSourceAuditError(f"missing PDF tool: {name}")
    version = subprocess.run([executable, "-v"], capture_output=True, check=False)
    banner = (
        (version.stdout + b"\n" + version.stderr)
        .decode("utf-8", errors="replace")
        .strip()
        .splitlines()
    )
    if version.returncode != 0 or not banner:
        raise ControlledReversalSourceAuditError(f"cannot identify PDF tool: {name}")
    return executable, {
        "name": name,
        "version": banner[0].strip(),
        "sha256": sha256_file(Path(executable)),
    }


def audit_thesis_pdf(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = _source_by_role(config, "NTNU thesis PDF")
    amendment = config["execution_amendment"]
    _verify_file(
        path,
        size=int(source["expected_bytes"]),
        sha256=str(amendment["thesis_pdf_sha256"]),
    )
    pdftotext, text_tool = _tool_identity("pdftotext.exe")
    pdfinfo, info_tool = _tool_identity("pdfinfo.exe")
    text_result = subprocess.run(
        [pdftotext, "-layout", "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        check=True,
    )
    try:
        text = text_result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ControlledReversalSourceAuditError(
            "P6Y extracted PDF text is not UTF-8"
        ) from exc
    info_result = subprocess.run(
        [pdfinfo, str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    page_match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", info_result.stdout)
    if page_match is None:
        raise ControlledReversalSourceAuditError("P6Y PDF page count unavailable")
    page_count = int(page_match.group(1))
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    folded = " ".join(text.split()).casefold()
    method_anchors = {
        "controlled_dataset": "we created a new dataset" in folded,
        "two_mock_paintings": "hyperspectral scans of two paintings" in folded,
        "two_stocks": all(
            phrase in folded
            for phrase in (
                "kodak ektachrome e100",
                "fujifilm fujichrome velvia 50",
            )
        ),
        "two_illuminants": all(
            phrase in folded for phrase in ("halogen lamps", "led panels")
        ),
        "multiple_exposures": "multiple exposure times" in folded,
        "120_reversal_film": "120-format colour reversal film" in folded,
        "e6_process": "e-6" in folded or "e6 process" in folded,
        "multispectral_film_scan": all(
            phrase in folded
            for phrase in (
                "multispectral imaging (msi)",
                "ten led spectral bands",
            )
        ),
        "painting_hyperspectral_ground_truth": "hyperspectral ground-truth" in folded,
    }
    reader = PdfReader(path)
    uris: set[str] = set()
    for page in reader.pages:
        for annotation in page.annotations or []:
            obj = annotation.get_object()
            action = obj.get("/A")
            if action and action.get("/URI"):
                uris.add(str(action.get("/URI")))
    repository_hosts = {
        "data.mendeley.com",
        "dataverse.harvard.edu",
        "figshare.com",
        "github.com",
        "osf.io",
        "perceive-data.iesl.forth.gr",
        "zenodo.org",
    }
    repository_uris = [
        uri for uri in sorted(uris) if urlparse(uri).hostname in repository_hosts
    ]
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "page_count": page_count,
        "page_count_exact": page_count == int(amendment["thesis_page_count"]),
        "text_sha256": text_sha256,
        "text_sha256_exact": text_sha256 == amendment["pdftotext_layout_utf8_sha256"],
        "tools": [text_tool, info_tool],
        "method_anchors": method_anchors,
        "external_uri_count": len(uris),
        "repository_uris": repository_uris,
        "separate_data_repository_linked": bool(repository_uris),
    }


def evaluate_controlled_reversal_source_audit(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    source_root = root / _relative_path(config["acquisition"]["destination"])
    record = audit_record(source_root / "ntnu_publication_record.json", config)
    support_zip = audit_support_zip(source_root / "submission_attachments.zip", config)
    thesis = audit_thesis_pdf(source_root / "thesis.pdf", config)
    explicit_pixel_artifacts = record["explicit_pixel_payload_artifacts"]
    machine_readable_pixels = (
        explicit_pixel_artifacts or support_zip["pixel_payload_members"]
    )
    observations = {
        "record_title_and_handle_exact": record["title_exact"]
        and record["handle_exact"],
        "record_contains_only_declared_open_files": record["declared_inventory_exact"],
        "support_zip_safe_inventory": support_zip["member_names_unique"]
        and support_zip["all_member_paths_safe"],
        "thesis_text_confirms_two_stock_two_illuminant_multiple_exposure_control": all(
            thesis["method_anchors"][key]
            for key in (
                "controlled_dataset",
                "two_mock_paintings",
                "two_stocks",
                "two_illuminants",
                "multiple_exposures",
            )
        ),
        "thesis_text_confirms_120_reversal_film_e6_process_and_multispectral_scan": all(
            thesis["method_anchors"][key]
            for key in ("120_reversal_film", "e6_process", "multispectral_film_scan")
        ),
        "separate_machine_readable_film_msi_payloads_publicly_listed": bool(
            machine_readable_pixels
        ),
        "separate_machine_readable_painting_hsi_payloads_publicly_listed": bool(
            machine_readable_pixels
        ),
        "pixel_payload_reuse_rights_explicit": any(
            row["license"] != _COPYRIGHT_ACT for row in explicit_pixel_artifacts
        ),
    }
    checks = {
        "retained_payload_integrity": True,
        "pdf_page_count_and_text_identity": thesis["page_count_exact"]
        and thesis["text_sha256_exact"],
        "zip_member_paths_relative_unique_non_symlink": support_zip[
            "member_names_unique"
        ]
        and support_zip["all_member_paths_safe"],
        "required_observations": all(observations.values()),
        "machine_readable_controlled_pixel_payloads": bool(machine_readable_pixels),
        "pixel_payload_rights_explicit": observations[
            "pixel_payload_reuse_rights_explicit"
        ],
        "zero_pixel_decode_operator_fit_render_visual_review": True,
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "record": record,
        "support_zip": support_zip,
        "thesis": thesis,
        "observations": observations,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "OPEN_SEPARATE_CONTROLLED_PIXEL_INTEGRITY_RIGHTS_AUDIT"
            if automatic_pass
            else DECISION
        ),
        "pixel_payload_count": len(machine_readable_pixels),
        "pixel_decode_count": 0,
        "operator_fit_count": 0,
        "render_count": 0,
        "visual_review_count": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DECISION",
    "REPORT_SCHEMA",
    "SCHEMA",
    "ControlledReversalSourceAuditError",
    "audit_record",
    "audit_support_zip",
    "audit_thesis_pdf",
    "canonical_json",
    "evaluate_controlled_reversal_source_audit",
    "load_contract",
    "sha256_file",
    "write_report",
]
