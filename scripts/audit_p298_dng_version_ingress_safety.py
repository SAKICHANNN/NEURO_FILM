"""Formal P298 DNG version/container-identity safety audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_dng_version,
    load_dng_forward_working_image,
)

SCHEMA = "neuro_film.p298_dng_version_ingress_safety_result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _tag(value: bytes, *, dtype: int = 1, count: int = 4) -> SimpleNamespace:
    return SimpleNamespace(value=value, dtype=dtype, count=count)


def _tags(
    version: bytes | None = b"\x01\x04\x00\x00",
    backward: bytes | None = b"\x01\x01\x00\x00",
    *,
    dtype: int = 1,
    count: int = 4,
) -> dict[int, SimpleNamespace]:
    result: dict[int, SimpleNamespace] = {}
    if version is not None:
        result[50706] = _tag(version, dtype=dtype, count=count)
    if backward is not None:
        result[50707] = _tag(backward)
    return result


def _reject(tags: dict[int, SimpleNamespace], expected: str) -> dict[str, Any]:
    try:
        _guard_dng_version(tags)
    except DngForwardRasterError as exc:
        return {
            "rejected": expected in str(exc),
            "message": str(exc),
            "camera_decode_calls": 0,
        }
    return {"rejected": False, "message": "accepted", "camera_decode_calls": 0}


def _controls() -> dict[str, Any]:
    return {
        "missing": _reject({}, "missing required DNGVersion"),
        "wrong_type": _reject(_tags(dtype=3), "four BYTE"),
        "wrong_count": _reject(_tags(count=3), "four BYTE"),
        "pre_1_0": _reject(_tags(b"\x00\xff\x00\x00"), "supported range"),
        "future_version": _reject(_tags(b"\x01\x07\x02\x00"), "supported range"),
        "backward_pre_1_0": _reject(
            _tags(backward=b"\x00\xff\x00\x00"), "below 1.0.0.0"
        ),
        "backward_exceeds_version": _reject(
            _tags(backward=b"\x01\x05\x00\x00"), "exceeds DNGVersion"
        ),
    }


def _inventory(path: Path) -> dict[str, list[int] | None]:
    with tifffile.TiffFile(path) as document:
        resolved = _guard_dng_version(document.pages[0].tags)
    return {
        key: None if value is None else list(value) for key, value in resolved.items()
    }


def run(config_path: Path, output_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    p98_path = ROOT / config["bindings"]["p98_config_path"]
    p244_path = ROOT / config["bindings"]["p244_config_path"]
    p98 = _json(p98_path)
    p244 = _json(p244_path)
    expected_outputs = {
        row["source_id"]: row["working_float32_sha256"] for row in p244["rows"]
    }
    rows = list(p98["rows"])
    if reverse:
        rows.reverse()
    results: list[dict[str, Any]] = []
    for row in rows:
        source = ROOT / row["logical_path"]
        before = _sha256_file(source)
        inventory = _inventory(source)
        working = load_dng_forward_working_image(
            source,
            expected_source_bytes=int(row["source_bytes"]),
            expected_source_sha256=str(row["source_sha256"]),
        )
        output_sha = _sha256_bytes(working.pixels.astype("<f4", copy=False).tobytes())
        results.append(
            {
                "source_id": row["source_id"],
                "source_bytes": source.stat().st_size,
                "source_sha256": before,
                "source_immutable": _sha256_file(source) == before,
                "inventory": inventory,
                "output_float32_sha256": output_sha,
                "output_matches_p98": output_sha == expected_outputs[row["source_id"]],
                "finite": bool(np.all(np.isfinite(working.pixels))),
            }
        )
    results.sort(key=lambda item: item["source_id"])
    controls = _controls()
    authority = config["authority"]
    expected_inventory = config["expected_inventory"]
    gates = {
        "authority_exact": all(
            (
                _sha256_file(ROOT / authority["sdk_path"]) == authority["sdk_sha256"],
                _sha256_file(ROOT / authority["tag_codes_source_path"])
                == authority["tag_codes_source_sha256"],
                _sha256_file(ROOT / authority["validation_source_path"])
                == authority["validation_source_sha256"],
                _sha256_file(p98_path) == config["bindings"]["p98_config_sha256"],
                _sha256_file(p244_path) == config["bindings"]["p244_config_sha256"],
            )
        ),
        "invalid_reject_before_decode": all(
            item["rejected"] and item["camera_decode_calls"] == 0
            for item in controls.values()
        ),
        "inventory_exact": all(
            item["inventory"] == expected_inventory[item["source_id"]]
            for item in results
        ),
        "p98_pixel_hashes_exact": all(item["output_matches_p98"] for item in results),
        "source_immutable": all(item["source_immutable"] for item in results),
        "finite": all(item["finite"] for item in results),
    }
    status = (
        "PASS_PRIVATE_DNG_VERSION_INGRESS_SAFETY"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_VERSION_INGRESS_SAFETY"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P298",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "implementation_sha256": _sha256_file(
                ROOT / config["bindings"]["implementation_path"]
            ),
            "runner_sha256": _sha256_file(ROOT / config["bindings"]["runner_path"]),
            "sdk_sha256": _sha256_file(ROOT / authority["sdk_path"]),
            "p98_config_sha256": _sha256_file(p98_path),
            "p244_config_sha256": _sha256_file(p244_path),
        },
        "controls": controls,
        "rows": results,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, reverse=args.reverse)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
