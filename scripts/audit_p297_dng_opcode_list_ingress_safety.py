"""Formal P297 DNG OpcodeList no-silent-required-opcode audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tifffile

from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_dng_opcode_lists,
    _parse_optional_opcode_list,
    load_dng_forward_working_image,
)
from src.preprocess.dng_metadata import _walk_pages

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neuro_film.p297_dng_opcode_list_ingress_safety_result.v1"


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


def _opcode(
    *,
    opcode_id: int = 9,
    version: int = 0x01030000,
    flags: int = 1,
    data: bytes = b"data",
) -> bytes:
    return struct.pack(">IIII", opcode_id, version, flags, len(data)) + data


def _opcode_list(*entries: bytes, count: int | None = None, tail: bytes = b"") -> bytes:
    return (
        struct.pack(">I", len(entries) if count is None else count)
        + b"".join(entries)
        + tail
    )


def _rejection(payload: bytes, expected: str) -> dict[str, Any]:
    try:
        _parse_optional_opcode_list(payload, tag_name="OpcodeList2", ifd_path="0")
    except DngForwardRasterError as exc:
        return {"rejected": expected in str(exc), "message": str(exc)}
    return {"rejected": False, "message": "accepted"}


def _synthetic_controls() -> dict[str, Any]:
    required: dict[str, Any] = {}
    for code, name in (
        (51008, "OpcodeList1"),
        (51009, "OpcodeList2"),
        (51022, "OpcodeList3"),
    ):
        page = SimpleNamespace(
            tags={code: SimpleNamespace(value=_opcode_list(_opcode(flags=0)))}
        )
        try:
            _guard_dng_opcode_lists([("0", page)])
        except DngForwardRasterError as exc:
            required[name] = {
                "rejected": "required DNG" in str(exc),
                "message": str(exc),
            }
        else:
            required[name] = {"rejected": False, "message": "accepted"}
    malformed = {
        "truncated_count": _rejection(b"\0\0\0", "truncated count"),
        "impossible_count": _rejection(
            _opcode_list(count=1), "impossible opcode count"
        ),
        "truncated_payload": _rejection(
            _opcode_list(_opcode(data=b"abc")[:-1]), "truncated payload"
        ),
        "trailing_bytes": _rejection(
            _opcode_list(_opcode(), tail=b"x"), "trailing bytes"
        ),
        "reserved_flags": _rejection(_opcode_list(_opcode(flags=5)), "reserved flags"),
        "zero_version": _rejection(
            _opcode_list(_opcode(version=0)), "unsupported version"
        ),
        "future_version": _rejection(
            _opcode_list(_opcode(version=0x01070200)), "unsupported version"
        ),
    }
    optional_page = SimpleNamespace(
        tags={
            51009: SimpleNamespace(value=_opcode_list(_opcode(), _opcode())),
            51022: SimpleNamespace(value=_opcode_list(_opcode(opcode_id=1))),
        }
    )
    optional_summary = list(_guard_dng_opcode_lists([("0", optional_page)]))
    return {
        "required": required,
        "malformed": malformed,
        "optional_summary": optional_summary,
    }


def _inventory(path: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    with tifffile.TiffFile(path) as document:
        pages = _walk_pages(document.pages)
        _guard_dng_opcode_lists(pages)
        for ifd_path, page in pages:
            for code in (51008, 51009, 51022):
                if code not in page.tags:
                    continue
                payload = bytes(page.tags[code].value)
                entries = _parse_optional_opcode_list(
                    payload,
                    tag_name={
                        51008: "OpcodeList1",
                        51009: "OpcodeList2",
                        51022: "OpcodeList3",
                    }[code],
                    ifd_path=ifd_path,
                )
                found.append(
                    {
                        "ifd": ifd_path,
                        "tag": code,
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                        "ids": [entry[0] for entry in entries],
                        "versions": [entry[1] for entry in entries],
                        "flags": [entry[2] for entry in entries],
                    }
                )
    return found


def run(config_path: Path, output_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    p98_path = ROOT / config["bindings"]["p98_config_path"]
    p98 = _json(p98_path)
    p98_evidence = _json(ROOT / "docs/evidence/P98_DNG_FORWARD_RASTER_RESULT.json")
    expected_outputs = {
        row["source_id"]: row["working_float32_sha256"] for row in p98_evidence["rows"]
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
                "warning_codes": [warning.code for warning in working.warnings],
                "optional_warning_count": sum(
                    warning.code == "optional_dng_opcodes_may_be_skipped"
                    for warning in working.warnings
                ),
                "finite": bool(np.all(np.isfinite(working.pixels))),
            }
        )
    results.sort(key=lambda item: item["source_id"])
    controls = _synthetic_controls()
    expected_inventory = config["expected_inventory"]
    gates = {
        "authority_exact": (
            _sha256_file(ROOT / config["authority"]["sdk_path"])
            == config["authority"]["sdk_sha256"]
            and _sha256_file(p98_path) == config["bindings"]["p98_config_sha256"]
        ),
        "required_reject_before_decode": all(
            item["rejected"] for item in controls["required"].values()
        ),
        "malformed_reject_before_decode": all(
            item["rejected"] for item in controls["malformed"].values()
        ),
        "optional_warning_exact": (
            controls["optional_summary"]
            == ["OpcodeList2(51009)@0:9/9", "OpcodeList3(51022)@0:1"]
            and all(
                item["optional_warning_count"]
                == (
                    1
                    if item["inventory"]
                    and any(entry["ids"] for entry in item["inventory"])
                    else 0
                )
                for item in results
            )
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
        "PASS_PRIVATE_DNG_OPCODE_LIST_INGRESS_SAFETY"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_OPCODE_LIST_INGRESS_SAFETY"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P297",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "implementation_sha256": _sha256_file(
                ROOT / config["bindings"]["implementation_path"]
            ),
            "runner_sha256": _sha256_file(ROOT / config["bindings"]["runner_path"]),
            "sdk_sha256": _sha256_file(ROOT / config["authority"]["sdk_path"]),
            "p98_config_sha256": _sha256_file(p98_path),
        },
        "controls": controls,
        "rows": results,
        "gates": gates,
        "information_flow": {
            "new_downloads": 0,
            "new_raw_dng_pixel_roles": 0,
            "existing_p98_rows_read": len(results),
            "persistent_pixel_outputs": 0,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p297_dng_opcode_list_ingress_safety_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config_path, output_path, reverse=args.reverse)
    print(
        json.dumps(
            {"status": report["status"], "output": str(output_path.relative_to(ROOT))},
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
