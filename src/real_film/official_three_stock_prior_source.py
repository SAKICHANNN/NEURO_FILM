"""Acquire and audit current first-party sources for the RF3.D2 stock matrix."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pypdf import PdfReader

SCHEMA = "neuro-film.rf3-three-stock-official-source-matrix-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-three-stock-official-source-matrix-report.v1"


class OfficialStockSourceError(RuntimeError):
    """Raised when the frozen source contract or an official PDF drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise OfficialStockSourceError("RF3.D2 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stocks = payload.get("stocks", [])
    gates = payload.get("gates", {})
    required_ids = [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "RF3.D2"
        or [row.get("stock_id") for row in stocks] != required_ids
        or gates.get("required_stock_ids") != required_ids
        or gates.get("minimum_common_measurement_domains") != 3
        or not all(
            gates.get(key) is True
            for key in (
                "require_https",
                "require_first_party_host",
                "require_pdf_header",
                "require_exact_page_count",
                "require_all_text_anchors",
                "require_all_declared_measurement_domains",
                "require_renderable_rgb_target_false",
                "exact_replay_required",
            )
        )
    ):
        raise OfficialStockSourceError("RF3.D2 frozen contract drift")
    if len(stocks) != len({row["stock_id"] for row in stocks}):
        raise OfficialStockSourceError("RF3.D2 duplicate stock id")
    for row in stocks:
        parsed = urlparse(str(row.get("url", "")))
        if parsed.scheme != "https" or parsed.hostname != row.get("allowed_host"):
            raise OfficialStockSourceError("RF3.D2 source URL/host drift")
        _relative_path(str(row.get("path", "")))
        if (
            not isinstance(row.get("maximum_bytes"), int)
            or row["maximum_bytes"] <= 0
            or not isinstance(row.get("expected_pages"), int)
            or row["expected_pages"] <= 0
            or not row.get("required_text_anchors")
            or not row.get("available_measurement_domains")
            or row.get("renderable_rgb_target") is not False
        ):
            raise OfficialStockSourceError("RF3.D2 invalid stock source row")
    return payload


def _download_exact(row: Mapping[str, Any], target: Path) -> bytes:
    request = urllib.request.Request(
        str(row["url"]),
        headers={"User-Agent": "K-MCFM-RF3-D2/1.0 (bounded first-party source audit)"},
    )
    maximum = int(row["maximum_bytes"])
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != row["allowed_host"]:
            raise OfficialStockSourceError("RF3.D2 redirect left first-party host")
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise OfficialStockSourceError("RF3.D2 source byte limit exceeded")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return data


def _audit_pdf(row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or len(data) > int(row["maximum_bytes"]):
        raise OfficialStockSourceError(f"RF3.D2 invalid PDF bytes: {row['stock_id']}")
    try:
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pragma: no cover - library-specific parse failures
        raise OfficialStockSourceError(
            f"RF3.D2 PDF decode failed: {row['stock_id']}"
        ) from exc
    folded = text.casefold()
    anchors = {
        anchor: anchor.casefold() in folded for anchor in row["required_text_anchors"]
    }
    return {
        "stock_id": row["stock_id"],
        "manufacturer": row["manufacturer"],
        "process_family": row["process_family"],
        "url": row["url"],
        "path": row["path"],
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "pages": len(reader.pages),
        "expected_pages": row["expected_pages"],
        "page_count_exact": len(reader.pages) == int(row["expected_pages"]),
        "text_anchors": anchors,
        "all_text_anchors_present": all(anchors.values()),
        "available_measurement_domains": row["available_measurement_domains"],
        "interpretation": row["interpretation"],
        "renderable_rgb_target": False,
    }


def evaluate(
    contract: Mapping[str, Any], root: Path, *, acquire_missing: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in contract["stocks"]:
        path = root / _relative_path(str(source["path"]))
        if not path.is_file():
            if not acquire_missing:
                raise OfficialStockSourceError(
                    f"RF3.D2 source missing: {source['stock_id']}"
                )
            _download_exact(source, path)
        rows.append(_audit_pdf(source, path))

    common_domains = sorted(
        set.intersection(
            *(set(row["available_measurement_domains"]) for row in rows)
        )
    )
    gates = contract["gates"]
    gate_results = {
        "stock_universe_exact": [row["stock_id"] for row in rows]
        == gates["required_stock_ids"],
        "page_counts_exact": all(row["page_count_exact"] for row in rows),
        "all_text_anchors_present": all(
            row["all_text_anchors_present"] for row in rows
        ),
        "all_sources_are_nonrenderable_priors": all(
            row["renderable_rgb_target"] is False for row in rows
        ),
        "minimum_common_measurement_domains": len(common_domains)
        >= int(gates["minimum_common_measurement_domains"]),
    }
    automatic_pass = all(gate_results.values())
    scientific = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "source_rows": rows,
        "common_measurement_domains": common_domains,
        "common_domain_count": len(common_domains),
        "incomparable_granularity_semantics": {
            "fujifilm_velvia_50": "diffuse_rms_granularity",
            "kodak_portra_400": "print_grain_index",
            "kodak_ektar_100": "print_grain_index",
            "direct_numeric_cross_manufacturer_comparison_allowed": False,
        },
        "incomparable_dye_density_semantics": {
            "fujifilm_velvia_50": "separated_spectral_dye_density",
            "kodak_portra_400": "aggregate_midscale_and_dmin_spectral_density",
            "kodak_ektar_100": "aggregate_midscale_and_dmin_spectral_density",
            "direct_common_basis_comparison_allowed": False,
        },
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["decisions"]["pass" if automatic_pass else "fail"],
        "next_experiment": (
            "nonrenderable_common_characteristic_spectral_mtf_comparison"
            if automatic_pass
            else "new_first_party_source_acquisition"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_evidence_id"] = _sha256_bytes(_canonical_json(scientific))
    return scientific


def write_report(report: Mapping[str, Any], path: Path) -> str:
    data = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)
