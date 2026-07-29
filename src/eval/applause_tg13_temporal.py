"""Bounded APPLAUSE TG13 scanner-response regime confirmation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlparse

import numpy as np
import requests


SCHEMA = "neuro_film.u6_p6k_applause_tg13_temporal_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6k_applause_tg13_temporal_report.v1"


class ApplauseTg13Error(RuntimeError):
    """Raised when the source or measurement contract fails closed."""


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _row_path(root: Path, config: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    return root / str(config["data_root"]) / f"scan_{int(row['scan_id']):06d}.fits"


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ApplauseTg13Error("unsupported contract schema")
    rows = config.get("rows")
    if not isinstance(rows, list) or len(rows) != 14:
        raise ApplauseTg13Error("contract must freeze exactly fourteen rows")
    scan_ids = [int(row["scan_id"]) for row in rows]
    if len(set(scan_ids)) != len(scan_ids):
        raise ApplauseTg13Error("scan identifiers must be unique")
    role_counts = {
        role: sum(row.get("role") == role for row in rows)
        for role in (
            "development_early",
            "development_late",
            "confirmatory_early",
            "confirmatory_late",
        )
    }
    if role_counts != {
        "development_early": 3,
        "development_late": 5,
        "confirmatory_early": 3,
        "confirmatory_late": 3,
    }:
        raise ApplauseTg13Error("development/confirmatory support drifted")
    for row in rows:
        if not str(row["filename_wedge"]).startswith(
            "DR4/wedges/Bamberg-South/"
        ):
            raise ApplauseTg13Error("wedge path escaped the frozen collection")
        if row["role"].startswith("development_") and len(
            str(row.get("sha256", ""))
        ) != 64:
            raise ApplauseTg13Error("development rows require exact hashes")
        if row["role"].startswith("confirmatory_") and "sha256" in row:
            raise ApplauseTg13Error(
                "confirmatory hashes must remain unseen in the contract"
            )
    download = config.get("download", {})
    base = urlparse(str(download.get("base_url", "")))
    if (
        base.scheme != "https"
        or base.hostname != download.get("allowed_host")
        or int(download.get("maximum_files", 0)) != 14
        or int(download.get("maximum_total_bytes", 0)) > 200_000_000
    ):
        raise ApplauseTg13Error("download boundary drifted")
    edges = config.get("measurement", {}).get("step_edges_x")
    if (
        not isinstance(edges, list)
        or len(edges) != 15
        or edges != sorted(edges)
        or edges[0] != 0
        or edges[-1] != int(config["fits"]["naxis1"])
    ):
        raise ApplauseTg13Error("step geometry drifted")
    if config.get("training_allowed") is not False:
        raise ApplauseTg13Error("training must remain forbidden")
    if config.get("stock_or_emulsion_fitting_allowed") is not False:
        raise ApplauseTg13Error("stock/emulsion fitting must remain forbidden")
    if config.get("scanner_calibration_allowed") is not False:
        raise ApplauseTg13Error("scanner calibration must remain forbidden")


def _parse_card_value(text: str) -> Any:
    value = text.split("/", 1)[0].strip()
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].strip()
    if value in {"T", "F"}:
        return value == "T"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value.replace("D", "E"))
        except ValueError:
            return value


def read_fits_header(path: Path) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {}
    cards = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(2880)
            if len(block) != 2880:
                raise ApplauseTg13Error("truncated FITS header")
            for start in range(0, 2880, 80):
                cards += 1
                card = block[start : start + 80].decode("ascii", "strict")
                key = card[:8].strip()
                if key == "END":
                    return header, ((cards * 80 + 2879) // 2880) * 2880
                if card[8:10] == "= ":
                    header[key] = _parse_card_value(card[10:])


def open_fits_image(
    path: Path, fits_contract: Mapping[str, Any]
) -> tuple[np.memmap, dict[str, Any]]:
    header, offset = read_fits_header(path)
    expected = {
        "BITPIX": int(fits_contract["bitpix"]),
        "NAXIS": 2,
        "NAXIS1": int(fits_contract["naxis1"]),
        "NAXIS2": int(fits_contract["naxis2"]),
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ApplauseTg13Error(f"FITS {key} drifted")
    if float(header.get("BSCALE", 1.0)) != float(fits_contract["bscale"]):
        raise ApplauseTg13Error("FITS BSCALE drifted")
    if float(header.get("BZERO", 0.0)) != float(fits_contract["bzero"]):
        raise ApplauseTg13Error("FITS BZERO drifted")
    expected_data = expected["NAXIS1"] * expected["NAXIS2"] * 2
    if path.stat().st_size < offset + expected_data:
        raise ApplauseTg13Error("FITS payload is truncated")
    image = np.memmap(
        path,
        dtype=">i2",
        mode="r",
        offset=offset,
        shape=(expected["NAXIS2"], expected["NAXIS1"]),
    )
    return image, header


def extract_step_vector_from_array(
    image: np.ndarray,
    *,
    edges: Sequence[int],
    row_start: int,
    row_stop: int,
    half_width: int,
    bzero: float = 0.0,
) -> tuple[np.ndarray, float]:
    if image.ndim != 2 or not 0 <= row_start < row_stop <= image.shape[0]:
        raise ApplauseTg13Error("invalid measurement rows")
    centers = [(int(a) + int(b)) // 2 for a, b in zip(edges[:-1], edges[1:])]
    values = np.asarray(
        [
            np.median(
                image[
                    row_start:row_stop,
                    max(0, center - half_width) : min(
                        image.shape[1], center + half_width + 1
                    ),
                ]
            )
            + bzero
            for center in centers
        ],
        dtype=np.float64,
    )
    span = float(values[-1] - values[0])
    if not np.isfinite(values).all() or span <= 0.0:
        raise ApplauseTg13Error("step response is non-finite or non-increasing")
    return (values - values[0]) / span, span


def extract_step_vector(
    path: Path, config: Mapping[str, Any]
) -> tuple[np.ndarray, float]:
    image, _ = open_fits_image(path, config["fits"])
    measurement = config["measurement"]
    return extract_step_vector_from_array(
        image,
        edges=measurement["step_edges_x"],
        row_start=int(measurement["row_start"]),
        row_stop=int(measurement["row_stop"]),
        half_width=int(measurement["plateau_half_width"]),
        bzero=float(config["fits"]["bzero"]),
    )


def acquire_rows(
    root: Path,
    config: Mapping[str, Any],
    *,
    roles: set[str] | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    validate_config(config)
    download = config["download"]
    client = session or requests.Session()
    records: list[dict[str, Any]] = []
    try:
        for row in config["rows"]:
            if roles is not None and str(row["role"]) not in roles:
                continue
            destination = _row_path(root, config, row)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                size = destination.stat().st_size
                digest = sha256_file(destination)
            else:
                url = urljoin(str(download["base_url"]), str(row["filename_wedge"]))
                parsed = urlparse(url)
                if (
                    parsed.scheme != "https"
                    or parsed.hostname != download["allowed_host"]
                ):
                    raise ApplauseTg13Error("download URL escaped frozen host")
                partial = destination.with_suffix(".fits.part")
                if partial.exists():
                    partial.unlink()
                response = None
                try:
                    response = client.get(
                        url,
                        stream=True,
                        timeout=(
                            int(download["connect_timeout_seconds"]),
                            int(download["read_timeout_seconds"]),
                        ),
                        headers={"User-Agent": "neuro-film-u6-p6k/1.0"},
                    )
                    response.raise_for_status()
                    expected = int(response.headers.get("Content-Length", "0"))
                    if expected != int(download["expected_bytes_per_file"]):
                        raise ApplauseTg13Error("wedge content length drifted")
                    digest_state = hashlib.sha256()
                    size = 0
                    with partial.open("xb") as handle:
                        for block in response.iter_content(
                            chunk_size=int(download["chunk_bytes"])
                        ):
                            if not block:
                                continue
                            size += len(block)
                            if size > expected:
                                raise ApplauseTg13Error(
                                    "wedge exceeded byte boundary"
                                )
                            digest_state.update(block)
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if size != expected:
                        raise ApplauseTg13Error("wedge download is incomplete")
                    digest = digest_state.hexdigest()
                    partial.replace(destination)
                except Exception:
                    if partial.exists():
                        partial.unlink()
                    raise
                finally:
                    if response is not None:
                        response.close()
            if size != int(download["expected_bytes_per_file"]):
                raise ApplauseTg13Error("local wedge size drifted")
            expected_digest = row.get("sha256")
            if expected_digest is not None and digest != expected_digest:
                raise ApplauseTg13Error("development wedge hash drifted")
            records.append(
                {
                    "scan_id": int(row["scan_id"]),
                    "role": str(row["role"]),
                    "path": str(destination.relative_to(root)).replace("\\", "/"),
                    "bytes": size,
                    "sha256": digest,
                }
            )
    finally:
        if session is None:
            client.close()
    if sum(record["bytes"] for record in records) > int(
        download["maximum_total_bytes"]
    ):
        raise ApplauseTg13Error("acquisition exceeded total byte boundary")
    return records


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def build_report(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    vectors: dict[int, np.ndarray] = {}
    spans: dict[int, float] = {}
    file_rows: list[dict[str, Any]] = []
    for row in config["rows"]:
        path = _row_path(root, config, row)
        if not path.is_file():
            raise ApplauseTg13Error(f"missing wedge scan {row['scan_id']}")
        digest = sha256_file(path)
        if row.get("sha256") is not None and digest != row["sha256"]:
            raise ApplauseTg13Error("development wedge hash drifted")
        vector, span = extract_step_vector(path, config)
        scan_id = int(row["scan_id"])
        vectors[scan_id] = vector
        spans[scan_id] = span
        file_rows.append(
            {
                "scan_id": scan_id,
                "role": str(row["role"]),
                "scan_date": str(row["scan_date"]),
                "sha256": digest,
                "raw_code_span": span,
                "normalized_steps": [float(value) for value in vector],
            }
        )

    roles = config["measurement"]["development_roles"]
    prototypes = {
        name: np.mean([vectors[int(scan_id)] for scan_id in scan_ids], axis=0)
        for name, scan_ids in roles.items()
    }
    prototype_rmse = _rmse(prototypes["early"], prototypes["late"])
    gates = config["gates"]
    development_rows: list[dict[str, Any]] = []
    confirmatory_rows: list[dict[str, Any]] = []
    for row in config["rows"]:
        role = str(row["role"])
        expected = "early" if role.endswith("_early") else "late"
        wrong = "late" if expected == "early" else "early"
        scan_id = int(row["scan_id"])
        correct_rmse = _rmse(vectors[scan_id], prototypes[expected])
        wrong_rmse = _rmse(vectors[scan_id], prototypes[wrong])
        result = {
            "scan_id": scan_id,
            "expected_regime": expected,
            "correct_prototype_rmse": correct_rmse,
            "wrong_prototype_rmse": wrong_rmse,
            "wrong_minus_correct_margin": wrong_rmse - correct_rmse,
        }
        if role.startswith("development_"):
            development_rows.append(result)
        else:
            confirmatory_rows.append(result)

    all_vectors = list(vectors.values())
    minimum_adjacent = min(float(np.min(np.diff(vector))) for vector in all_vectors)
    minimum_span = min(spans.values())
    gate_results = {
        "raw_span": minimum_span >= float(gates["minimum_raw_code_span"]),
        "monotone_steps": minimum_adjacent
        >= float(gates["minimum_adjacent_normalized_step"]),
        "development_separation": prototype_rmse
        >= float(gates["minimum_development_prototype_rmse"]),
        "development_repeatability": max(
            row["correct_prototype_rmse"] for row in development_rows
        )
        <= float(gates["maximum_development_correct_prototype_rmse"]),
        "confirmatory_repeatability": max(
            row["correct_prototype_rmse"] for row in confirmatory_rows
        )
        <= float(gates["maximum_confirmatory_correct_prototype_rmse"]),
        "confirmatory_margin": min(
            row["wrong_minus_correct_margin"] for row in confirmatory_rows
        )
        >= float(gates["minimum_confirmatory_wrong_minus_correct_margin"]),
        "confirmatory_support": len(confirmatory_rows)
        == int(gates["required_confirmatory_passes"]),
    }
    automatic_pass = all(gate_results.values())
    stable_payload = {
        "contract_schema": config["schema"],
        "files": file_rows,
        "development_prototypes": {
            name: [float(value) for value in vector]
            for name, vector in prototypes.items()
        },
        "development_prototype_rmse": prototype_rmse,
        "development_rows": development_rows,
        "confirmatory_rows": confirmatory_rows,
        "minimum_raw_code_span": minimum_span,
        "minimum_adjacent_normalized_step": minimum_adjacent,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    stable_id = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        **stable_payload,
        "stable_evidence_id": stable_id,
        "decision": (
            "retain_two_repeat_stable_scanner_wedge_response_regimes_as_nuisance_evidence"
            if automatic_pass
            else "close_two_regime_hypothesis_retain_unidentified_scanner_wedge_nuisance"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "ApplauseTg13Error",
    "acquire_rows",
    "build_report",
    "extract_step_vector",
    "extract_step_vector_from_array",
    "open_fits_image",
    "read_fits_header",
    "sha256_file",
    "validate_config",
    "write_report",
]
