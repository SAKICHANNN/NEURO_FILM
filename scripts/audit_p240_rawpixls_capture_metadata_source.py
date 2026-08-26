"""Run the frozen P240 metadata-only source audit without network or RAW access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.rawpixls_capture_metadata_audit import (
    RawPixlsCaptureMetadataError,
    parse_rawpixls_exif_text,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def run(config_path: Path, data_root: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "P240":
        raise ValueError("unexpected P240 contract")
    snapshot = data_root / "repository_snapshot.json"
    source = config["source"]
    if snapshot.stat().st_size != source["repository_snapshot_bytes"]:
        raise ValueError("repository snapshot length mismatch")
    if _sha256(snapshot) != source["repository_snapshot_sha256"]:
        raise ValueError("repository snapshot hash mismatch")

    candidates = list(config["candidates"])
    if reverse:
        candidates.reverse()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        path = data_root / "exif" / f"{int(candidate['id']):04d}.txt"
        if not path.is_file():
            raise ValueError(f"missing frozen EXIF payload: {candidate['id']}")
        try:
            parsed = parse_rawpixls_exif_text(path.read_text(encoding="utf-8"))
            parse_error = None
        except (UnicodeError, RawPixlsCaptureMetadataError) as exc:
            parsed = {"complete": False, "facts": {}, "missing": []}
            parse_error = str(exc)
        rows.append(
            {
                "complete": bool(parsed["complete"]),
                "exif_bytes": path.stat().st_size,
                "exif_sha256": _sha256(path),
                "facts": parsed["facts"],
                "id": int(candidate["id"]),
                "make": candidate["make"],
                "missing": parsed["missing"],
                "model": candidate["model"],
                "parse_error": parse_error,
                "raw_sha256": candidate["raw_sha256"],
            }
        )
    rows.sort(key=lambda row: row["id"])
    complete = [row for row in rows if row["complete"]]
    complete_makes = sorted({row["make"].casefold() for row in complete})
    counts = {
        make: sum(row["make"].casefold() == make for row in complete)
        for make in complete_makes
    }
    largest_fraction = max(counts.values(), default=0) / max(len(complete), 1)
    gates_config = config["gates"]
    gates = {
        "complete_rows": len(complete) >= gates_config["minimum_complete_rows"],
        "complete_makes": len(complete_makes) >= gates_config["minimum_complete_makes"],
        "largest_make_fraction": largest_fraction
        <= gates_config["maximum_largest_make_fraction"],
        "source_inventory": len(rows) == config["selection"]["candidate_count"],
        "zero_pixel_decodes": True,
        "zero_raw_downloads": True,
    }
    passed = all(gates.values())
    return {
        "schema": "neuro-film.p240-rawpixls-capture-metadata-source-audit-report.v1",
        "experiment_id": "P240",
        "status": (
            "PASS_PRIVATE_CAPTURE_METADATA_SOURCE_ELIGIBILITY"
            if passed
            else "FAIL_CLOSED_CAPTURE_METADATA_SOURCE_ELIGIBILITY"
        ),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "repository_snapshot_sha256": _sha256(snapshot),
        },
        "inventory": {
            "candidate_rows": len(rows),
            "complete_makes": len(complete_makes),
            "complete_rows": len(complete),
            "exif_bytes": sum(row["exif_bytes"] for row in rows),
            "largest_make_fraction": largest_fraction,
            "raw_bytes_downloaded": 0,
            "raw_pixels_decoded": 0,
        },
        "gates": gates,
        "rows": rows,
        "decision": (
            "open_separately_preregistered_member_integrity_and_source_only_awb_d0"
            if passed
            else "close_exact_p240_source_without_raw_download_or_replacement"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.data_root, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))


if __name__ == "__main__":
    main()
