"""Metadata-only YFCC Ektar digital-companion feasibility audit."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus


class YfccEktarCompanionError(ValueError):
    """Raised when a frozen input or query boundary is violated."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time(value: str) -> datetime | None:
    clean = value.strip().removesuffix(".0")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def _epoch(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: object) -> str:
    return " ".join(unquote_plus(str(value or "")).casefold().split())


def _tokens(value: object, minimum_length: int) -> set[str]:
    normalized = "".join(char if char.isalnum() else " " for char in _text(value))
    return {token for token in normalized.split() if len(token) >= minimum_length}


def _float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _distance_metres(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    lat1, lon1 = _float(left.get("latitude")), _float(left.get("longitude"))
    lat2, lon2 = _float(right.get("latitude")), _float(right.get("longitude"))
    if None in (lat1, lon1, lat2, lon2):
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "sf3.a3f-yfcc-ektar-digital-companion-metadata-v1":
        raise YfccEktarCompanionError("unexpected experiment identity")
    for field, hash_field in (
        ("yfcc_sqlite", "yfcc_sqlite_sha256"),
        ("yfcc_config", "yfcc_config_sha256"),
        ("ektar_target_manifest", "ektar_target_manifest_sha256"),
    ):
        source = root / str(contract["sources"][field])
        if not source.is_file() or sha256_file(source) != contract["sources"][hash_field]:
            raise YfccEktarCompanionError(f"source identity drift: {field}")
    return contract


def _target_rows(root: Path, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads((root / str(contract["sources"]["ektar_target_manifest"])).read_text("utf-8"))
    rows = [
        row
        for row in manifest["attempts"]
        if row.get("decision") == contract["sources"]["target_decision"]
        and row.get("film_stock_id") == contract["sources"]["target_stock_id"]
    ]
    rows.sort(key=lambda row: int(row["photoid"]))
    return rows


def _candidate_evaluation(
    target: Mapping[str, Any], candidate: Mapping[str, Any], query: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    if int(candidate["photoid"]) == int(target["photoid"]):
        return "same_photo", None
    if candidate["licenseurl"] not in set(query["allowed_license_urls"]):
        return "license", None
    capture_device = _text(candidate["capturedevice"])
    if not capture_device:
        return "missing_capture_device", None
    if any(term in capture_device for term in query["scanner_device_terms"]):
        return "scanner_device", None
    candidate_text = _text(
        " ".join(str(candidate.get(key) or "") for key in ("title", "description", "usertags", "machinetags"))
    )
    if any(term in candidate_text for term in query["film_text_terms"]):
        return "film_text", None
    target_capture, candidate_capture = _time(str(target["datetaken"])), _time(str(candidate["datetaken"]))
    if target_capture is None or candidate_capture is None:
        return "missing_capture_time", None
    capture_delta = abs((candidate_capture - target_capture).total_seconds())
    if capture_delta > float(query["maximum_capture_time_delta_seconds"]):
        return "capture_time", None
    target_upload, candidate_upload = _epoch(str(target["dateuploaded"])), _epoch(str(candidate["dateuploaded"]))
    if target_upload is None or candidate_upload is None:
        return "missing_upload_time", None
    upload_delta = abs(candidate_upload - target_upload)
    if upload_delta > int(query["maximum_upload_time_delta_seconds"]):
        return "upload_time", None
    scene = query["same_scene_evidence"]
    if capture_delta > float(scene["maximum_capture_time_delta_seconds"]):
        return "scene_time", None
    title_overlap = sorted(
        _tokens(target["title"], int(scene["minimum_title_token_length"]))
        & _tokens(candidate["title"], int(scene["minimum_title_token_length"]))
    )
    distance = _distance_metres(target, candidate)
    accuracy_ok = min(int(target.get("accuracy") or 0), int(candidate.get("accuracy") or 0)) >= int(
        scene["minimum_location_accuracy"]
    )
    routes: list[str] = []
    if accuracy_ok and distance is not None and distance <= float(scene["maximum_geodesic_distance_metres"]):
        routes.append("time_and_geo")
    if len(title_overlap) >= int(scene["minimum_shared_title_tokens"]):
        routes.append("time_and_title")
    routes = [route for route in routes if route in set(scene["allowed_routes"])]
    if not routes:
        return "same_scene", None
    result = dict(candidate)
    result.update(
        {
            "target_photoid": int(target["photoid"]),
            "capture_time_delta_seconds": capture_delta,
            "upload_time_delta_seconds": upload_delta,
            "same_scene_routes": routes,
            "shared_title_tokens": title_overlap,
            "geodesic_distance_metres": distance,
        }
    )
    return "eligible", result


def audit(root: Path, contract_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    targets = _target_rows(root, contract)
    database = root / str(contract["sources"]["yfcc_sqlite"])
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    columns = (
        "photoid,uid,datetaken,dateuploaded,capturedevice,title,description,usertags,machinetags,"
        "longitude,latitude,accuracy,pageurl,downloadurl,licensename,licenseurl"
    )
    radius = int(contract["query"]["photoid_radius"])
    reasons: Counter[str] = Counter()
    eligible: dict[tuple[int, int], dict[str, Any]] = {}
    examined: set[int] = set()
    ordered_targets = list(reversed(targets)) if reverse else targets
    try:
        for manifest_row in ordered_targets:
            target_row = connection.execute(
                f"select {columns} from {contract['sources']['yfcc_table']} where photoid=?",
                (int(manifest_row["photoid"]),),
            ).fetchone()
            if target_row is None or str(target_row["uid"]) != str(manifest_row["uid"]):
                raise YfccEktarCompanionError("target row identity mismatch")
            target = dict(target_row)
            candidates = connection.execute(
                f"select {columns} from {contract['sources']['yfcc_table']} "
                "where photoid between ? and ? and uid=? order by photoid",
                (int(target["photoid"]) - radius, int(target["photoid"]) + radius, str(target["uid"])),
            )
            for candidate_row in candidates:
                candidate = dict(candidate_row)
                examined.add(int(candidate["photoid"]))
                reason, result = _candidate_evaluation(target, candidate, contract["query"])
                reasons[reason] += 1
                if result is not None:
                    eligible[(int(result["target_photoid"]), int(result["photoid"]))] = result
    finally:
        connection.close()
    if len(examined) > int(contract["gates"]["maximum_sqlite_rows_examined"]):
        raise YfccEktarCompanionError("examined-row ceiling exceeded")
    target_uids = {str(row["uid"]) for row in targets}
    eligible_rows = [eligible[key] for key in sorted(eligible)]
    eligible_uids = {str(row["uid"]) for row in eligible_rows}
    checks = {
        "target_rows": len(targets) >= int(contract["gates"]["minimum_target_rows"]),
        "target_uids": len(target_uids) >= int(contract["gates"]["minimum_target_uids"]),
        "eligible_pairs": len(eligible_rows) >= int(contract["gates"]["minimum_eligible_pairs"]),
        "eligible_uids": len(eligible_uids) >= int(contract["gates"]["minimum_eligible_uids"]),
        "network_requests_zero": int(contract["gates"]["maximum_network_requests"]) == 0,
        "image_requests_zero": int(contract["gates"]["maximum_image_requests"]) == 0,
        "pixel_decodes_zero": int(contract["gates"]["maximum_pixel_decodes"]) == 0,
    }
    scientific = {
        "experiment_id": contract["experiment_id"],
        "target_manifest_sha256": contract["sources"]["ektar_target_manifest_sha256"],
        "target_rows": len(targets),
        "target_uids": len(target_uids),
        "sqlite_rows_examined": len(examined),
        "reason_counts": dict(sorted(reasons.items())),
        "eligible_pairs": eligible_rows,
        "eligible_pair_count": len(eligible_rows),
        "eligible_uid_count": len(eligible_uids),
        "checks": checks,
        "decision": "PASS_OPEN_BOUNDED_LIVE_RIGHTS_PREFLIGHT" if all(checks.values()) else "FAIL_CLOSE_EXACT_YFCC_COMPANION_ROUTE",
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro-film.sf3-a3f-yfcc-ektar-digital-companion-metadata-result.v1",
        **scientific,
        "network_requests": 0,
        "image_requests": 0,
        "pixel_decodes": 0,
        "stable_evidence_id": sha256_bytes(canonical_json(scientific)),
    }
