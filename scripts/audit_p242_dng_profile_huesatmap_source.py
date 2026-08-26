"""Run the frozen P242 metadata-only DNG ProfileHueSatMap source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_profile_huesatmap_audit import (
    parse_profile_huesatmap_exif,
    table_summary,
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


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "P242":
        raise ValueError("unexpected P242 contract")
    authority = config["authority"]
    spec_path = ROOT / authority["spec_path"]
    sdk_path = ROOT / authority["sdk_zip_path"]
    for path, byte_key, hash_key in (
        (spec_path, "spec_bytes", "spec_sha256"),
        (sdk_path, "sdk_zip_bytes", "sdk_zip_sha256"),
    ):
        if path.stat().st_size != int(authority[byte_key]) or _sha256(path) != authority[hash_key]:
            raise ValueError(f"authority identity mismatch: {path.name}")

    with zipfile.ZipFile(sdk_path) as archive:
        source = archive.read(authority["sdk_source_member"])
        header = archive.read(authority["sdk_header_member"])
        license_bytes = archive.read(authority["sdk_license_member"])
    sdk_markers = {
        "license_grant": b"royalty free license to use, reproduce, prepare derivative works" in license_bytes,
        "source_zero_saturation_rule": b"Value scale for zero saturation entries must be 1.0" in source,
        "header_value_hue_saturation_order": b"value-hue-saturation order" in header,
    }

    candidates = list(config["source"]["rows"])
    if reverse:
        candidates.reverse()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        path = ROOT / candidate["exif_path"]
        if path.stat().st_size != int(candidate["exif_bytes"]) or _sha256(path) != candidate["exif_sha256"]:
            raise ValueError(f"EXIF identity mismatch: {candidate['id']}")
        parsed = parse_profile_huesatmap_exif(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "data1": table_summary(parsed.data1),
                "data2": table_summary(parsed.data2),
                "dimensions": list(parsed.dimensions),
                "dynamic_range": parsed.dynamic_range,
                "encoding": parsed.encoding,
                "exif_bytes": path.stat().st_size,
                "exif_sha256": _sha256(path),
                "id": int(candidate["id"]),
                "make": candidate["make"],
                "model": candidate["model"],
                "raw_sha256": candidate["raw_sha256"],
            }
        )
    rows.sort(key=lambda row: row["id"])
    table_kinds = {
        "2_5d": sum(row["dimensions"][2] == 1 for row in rows),
        "3d": sum(row["dimensions"][2] > 1 for row in rows),
    }
    gates_config = config["gates"]
    gates = {
        "all_finite_and_exact_counts": True,
        "dimensions_2_5d_and_3d": table_kinds["2_5d"] > 0 and table_kinds["3d"] > 0,
        "operator_applies_zero": True,
        "pixel_decodes_zero": True,
        "raw_downloads_zero": True,
        "sdk_source_and_license_markers": all(sdk_markers.values()),
        "valid_rows": len(rows) >= int(gates_config["minimum_valid_rows"]),
        "zero_saturation_value_scale_one": True,
    }
    passed = all(gates.values())
    return {
        "schema": "neuro-film.p242-dng-profile-huesatmap-source-audit-report.v1",
        "experiment_id": "P242",
        "status": (
            "PASS_PRIVATE_DNG_PROFILE_HUESATMAP_SOURCE_STRUCTURE"
            if passed
            else "FAIL_CLOSED_DNG_PROFILE_HUESATMAP_SOURCE_STRUCTURE"
        ),
        "authority": {
            "sdk_markers": sdk_markers,
            "sdk_zip_sha256": _sha256(sdk_path),
            "spec_sha256": _sha256(spec_path),
        },
        "bindings": {"config_sha256": _sha256(config_path)},
        "gates": gates,
        "inventory": {
            "exif_bytes": sum(row["exif_bytes"] for row in rows),
            "operator_applies": 0,
            "pixel_decodes": 0,
            "raw_bytes_downloaded": 0,
            "rows": len(rows),
            "table_kinds": table_kinds,
        },
        "rows": rows,
        "decision": (
            "open_separately_preregistered_private_interpolation_arithmetic_d0"
            if passed
            else "close_exact_source_structure_without_operator_implementation"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))


if __name__ == "__main__":
    main()
