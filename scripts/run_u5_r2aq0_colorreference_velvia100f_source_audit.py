#!/usr/bin/env python
"""Audit the frozen ColorReference Velvia 100F recorder/measurement source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.request
import zipfile
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)


CONFIG_SHA256 = "b67cf59e5881a26fce3dc31f2cae61e89bbb7eaea94542cac7a297154dc19dee"
REPORT_SCHEMA = "neuro-film.u5.r2aq0.colorreference-source-audit.v1"
USER_AGENT = "NeuroFilm-SourceAudit/1.0 (bounded research integrity audit)"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AQ0 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ0 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2aq0-colorreference-velvia100f-source-audit-v1"
        or config["acquisition"]["expected_total_bytes_per_pass"] != 3147395
        or config["acquisition"]["maximum_total_bytes_per_pass"] != 8388608
        or config["rights_contract"]["formal_open_data_license_observed"]
        or not config["rights_contract"][
            "internal_research_download_and_audit_allowed"
        ]
        or config["fit_allowed"]
        or config["training_allowed"]
        or config["image_render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AQ0 frozen contract mismatch")
    return config


def _fetch(url: str, *, maximum_bytes: int) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        status = int(response.status)
        payload = response.read(maximum_bytes + 1)
    if status != 200:
        raise ValueError(f"AQ0 expected HTTP 200 for {url}, got {status}")
    if len(payload) > maximum_bytes:
        raise ValueError(f"AQ0 response exceeds bounded size for {url}")
    return status, payload


def _persist_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"AQ0 retained file drift at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise ValueError(f"AQ0 unexpected temporary file at {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_zip_member_names(payload: bytes) -> list[str]:
    """Return unique safe regular-file members without extracting the archive."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names: list[str] = []
        seen: set[str] = set()
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if (
                info.is_dir()
                or pure.is_absolute()
                or not pure.parts
                or any(part in ("", ".", "..") for part in pure.parts)
                or ":" in pure.parts[0]
                or name in seen
            ):
                raise ValueError("AQ0 ZIP contains unsafe or duplicate member")
            seen.add(name)
            names.append(name)
    return names


def _decode_reference_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("AQ0 reference member is not decodable text")


def reference_member_record(
    archive_payload: bytes, member_name: str
) -> dict[str, Any]:
    """Inventory one IT8/CGATS member and its declared table structure."""

    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        payload = archive.read(member_name)
    text = _decode_reference_text(payload)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper = [line.upper() for line in lines]
    number_of_fields = None
    number_of_sets = None
    for line in lines:
        match = re.match(r"^NUMBER_OF_FIELDS\s+\"?(\d+)\"?$", line, re.I)
        if match:
            number_of_fields = int(match.group(1))
        match = re.match(r"^NUMBER_OF_SETS\s+\"?(\d+)\"?$", line, re.I)
        if match:
            number_of_sets = int(match.group(1))
    if "BEGIN_DATA" not in upper or "END_DATA" not in upper:
        raise ValueError("AQ0 reference member lacks a CGATS data block")
    begin_data = upper.index("BEGIN_DATA")
    end_data = upper.index("END_DATA")
    data_rows = lines[begin_data + 1 : end_data]
    format_fields: list[str] = []
    if "BEGIN_DATA_FORMAT" in upper and "END_DATA_FORMAT" in upper:
        begin_format = upper.index("BEGIN_DATA_FORMAT")
        end_format = upper.index("END_DATA_FORMAT")
        format_fields = " ".join(
            lines[begin_format + 1 : end_format]
        ).split()
    if (
        number_of_sets is None
        or number_of_sets != len(data_rows)
        or number_of_fields is None
        or (format_fields and number_of_fields != len(format_fields))
    ):
        raise ValueError("AQ0 reference member declares inconsistent fields")
    return {
        "member": member_name,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "number_of_fields": number_of_fields,
        "number_of_sets": number_of_sets,
        "format_fields": format_fields,
        "data_row_count": len(data_rows),
    }


def source_tiff_record(payload: bytes, *, slide_index: int) -> dict[str, Any]:
    """Decode TIFF samples directly, without profile conversion."""

    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.mode != "RGB" or image.n_frames != 1:
            raise ValueError("AQ0 source TIFF must be one RGB frame")
        samples = np.asarray(image)
        info_keys = sorted(str(key) for key in image.info)
        width, height = image.size
    if (
        samples.shape != (height, width, 3)
        or samples.dtype != np.uint8
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("AQ0 source TIFF sample contract mismatch")
    return {
        "slide_index": slide_index,
        "width": width,
        "height": height,
        "mode": "RGB",
        "dtype": "uint8",
        "minimum_sample": int(samples.min()),
        "maximum_sample": int(samples.max()),
        "sample_bytes_sha256": _sha256(samples.tobytes(order="C")),
        "image_info_keys": info_keys,
    }


def _pair_key(member_name: str) -> tuple[int, int] | None:
    match = re.search(
        r"testscan(?P<set>\d+)[_-](?P<slide>[1-5])(?:\D|$)",
        member_name,
        re.I,
    )
    if match is None:
        return None
    return int(match.group("set")), int(match.group("slide"))


def _audit_pass(
    config: dict[str, Any], *, pass_name: str, output_root: Path
) -> dict[str, Any]:
    pass_root = output_root / pass_name
    page_status, page_payload = _fetch(
        str(config["official_page"]["url"]), maximum_bytes=1048576
    )
    page_text = _decode_reference_text(page_payload)
    normalized_page = " ".join(page_text.split()).lower()
    required = [
        {
            "text": fact,
            "present": " ".join(str(fact).split()).lower()
            in normalized_page,
        }
        for fact in config["official_page"]["required_text_facts"]
    ]
    if not all(row["present"] for row in required):
        raise ValueError("AQ0 official page no longer contains frozen facts")
    _persist_exact(pass_root / "official_page.html", page_payload)
    file_records: list[dict[str, Any]] = []
    tiff_records: list[dict[str, Any]] = []
    reference_archives: list[dict[str, Any]] = []
    pair_keys: set[tuple[int, int]] = set()
    member_hash_to_locations: dict[str, list[str]] = {}
    total_bytes = 0
    rows = (
        list(config["acquisition"]["source_tiffs"])
        + list(config["acquisition"]["measured_reference_archives"])
    )
    for row in rows:
        url = str(row["url"])
        expected_bytes = int(row["expected_bytes"])
        status, payload = _fetch(url, maximum_bytes=expected_bytes)
        if len(payload) != expected_bytes:
            raise ValueError(f"AQ0 byte count drift for {url}")
        total_bytes += len(payload)
        filename = url.rsplit("/", 1)[-1]
        _persist_exact(pass_root / "downloads" / filename, payload)
        record = {
            "url": url,
            "filename": filename,
            "http_status": status,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
        file_records.append(record)
        if filename.lower().endswith(".tif"):
            tiff_records.append(
                source_tiff_record(
                    payload, slide_index=int(row["slide_index"])
                )
            )
            continue
        names = safe_zip_member_names(payload)
        member_records = [
            reference_member_record(payload, name) for name in names
        ]
        archive_pair_keys = []
        for member in member_records:
            key = _pair_key(str(member["member"]))
            if key is None:
                raise ValueError("AQ0 cannot derive pair key from member")
            if key[0] != int(row["test_set"]):
                raise ValueError("AQ0 member test-set key mismatch")
            pair_keys.add(key)
            archive_pair_keys.append(list(key))
            member_hash_to_locations.setdefault(
                str(member["sha256"]), []
            ).append(f"{filename}:{member['member']}")
        reference_archives.append(
            {
                **record,
                "declared_format": row["format"],
                "test_set": int(row["test_set"]),
                "member_count": len(member_records),
                "members": member_records,
                "pair_keys": archive_pair_keys,
            }
        )
    expected_total = int(
        config["acquisition"]["expected_total_bytes_per_pass"]
    )
    if (
        total_bytes != expected_total
        or total_bytes
        > int(config["acquisition"]["maximum_total_bytes_per_pass"])
    ):
        raise ValueError("AQ0 total download budget mismatch")
    expected_pairs = {
        (test_set, slide)
        for test_set in (1, 2, 3, 4, 5, 9)
        for slide in range(1, 6)
    }
    if pair_keys != expected_pairs:
        raise ValueError("AQ0 reference pair coverage mismatch")
    duplicate_members = {
        digest: locations
        for digest, locations in member_hash_to_locations.items()
        if len(locations) > 1
    }
    return {
        "pass_name": pass_name,
        "page": {
            "http_status": page_status,
            "bytes": len(page_payload),
            "sha256": _sha256(page_payload),
            "required_text_facts": required,
        },
        "download_total_bytes": total_bytes,
        "files": file_records,
        "source_tiffs": tiff_records,
        "reference_archives": reference_archives,
        "pair_keys": [list(key) for key in sorted(pair_keys)],
        "duplicate_reference_member_payloads": duplicate_members,
    }


def _stable_pass_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "pass_name"}


def run_audit(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, expected_sha256=CONFIG_SHA256)
    passes = [
        _audit_pass(config, pass_name=name, output_root=output_root)
        for name in ("run_a", "run_b")
    ]
    passes_identical = _stable_pass_projection(
        passes[0]
    ) == _stable_pass_projection(passes[1])
    automatic_checks = [
        {"name": "two_download_passes_identical", "passed": passes_identical},
        {
            "name": "all_required_page_facts_present",
            "passed": all(
                row["present"]
                for row in passes[0]["page"]["required_text_facts"]
            ),
        },
        {
            "name": "expected_total_bytes_exact",
            "passed": passes[0]["download_total_bytes"]
            == int(
                config["acquisition"]["expected_total_bytes_per_pass"]
            ),
        },
        {
            "name": "five_source_tiffs_decode",
            "passed": len(passes[0]["source_tiffs"]) == 5,
        },
        {
            "name": "twelve_reference_archives_parse",
            "passed": len(passes[0]["reference_archives"]) == 12,
        },
        {
            "name": "thirty_set_slide_pair_keys",
            "passed": len(passes[0]["pair_keys"]) == 30,
        },
    ]
    automatic_pass = all(
        bool(check["passed"]) for check in automatic_checks
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "passes": passes,
        "passes_identical": passes_identical,
        "automatic_checks": automatic_checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "open_semantics_and_identifiability_design_only"
            if automatic_pass
            else "close_colorreference_source"
        ),
        "rights": config["rights_contract"],
        "epistemic_contract": config["epistemic_contract"],
        "fit_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": automatic_pass,
                "decision": report["decision"],
                "report_sha256": _sha256(report_bytes),
                "download_bytes_per_pass": passes[0][
                    "download_total_bytes"
                ],
                "pair_key_count": len(passes[0]["pair_keys"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq0_colorreference_velvia100f_source_audit_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/source_recon/colorreference_velvia100f_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    run_audit(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
