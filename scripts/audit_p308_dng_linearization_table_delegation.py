"""Formal P308 DNG LinearizationTable delegation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Callable
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
    _guard_dng_opcode_lists,
    _guard_dng_version,
    _guard_unsupported_profile_gain_table_map,
    _guard_unsupported_profile_huesatmap,
    _guard_unsupported_profile_look_table,
    _guard_unsupported_profile_tone_curve,
    _linearization_table_delegation,
    load_dng_forward_working_image,
)
from src.preprocess.dng_metadata import _walk_pages

SCHEMA = "neuro_film.p308_dng_linearization_table_delegation_result.v1"
BASE_WARNING_CODES = [
    "private_dng_forward_raster",
    "unqualified_demosaic_and_calibration",
    "scene_linear_no_tone_map",
]


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


def _tag(
    values: np.ndarray | list[int], *, dtype: int = 3, count: int | None = None
) -> SimpleNamespace:
    array = np.asarray(values)
    return SimpleNamespace(
        value=array,
        dtype=dtype,
        count=array.size if count is None else count,
    )


def _page(tags: dict[int, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(tags=tags, pages=None)


def _reject_linearization(
    pages: list[tuple[str, SimpleNamespace]], expected: str
) -> dict[str, Any]:
    try:
        _linearization_table_delegation(pages)  # type: ignore[arg-type]
    except DngForwardRasterError as exc:
        return {
            "rejected": expected in str(exc),
            "message": str(exc),
            "camera_decode_calls": 0,
        }
    return {"rejected": False, "message": "accepted", "camera_decode_calls": 0}


def _linearization_controls() -> dict[str, Any]:
    valid = _tag(np.asarray([0, 1], dtype=np.uint16))
    return {
        "multiple": _reject_linearization(
            [("0", _page({50712: valid})), ("0/1", _page({50712: valid}))],
            "multiple",
        ),
        "wrong_type": _reject_linearization(
            [("0", _page({50712: _tag([0, 1], dtype=4)}))], "TIFF SHORT"
        ),
        "count_below_minimum": _reject_linearization(
            [("0", _page({50712: _tag([0], count=1)}))], "count is out of range"
        ),
        "count_above_maximum": _reject_linearization(
            [
                (
                    "0",
                    _page(
                        {
                            50712: _tag(
                                np.arange(65537, dtype=np.uint32), count=65537
                            )
                        }
                    ),
                )
            ],
            "count is out of range",
        ),
        "wrong_payload_dtype": _reject_linearization(
            [("0", _page({50712: _tag(np.asarray([0, 1], dtype=np.uint32))}))],
            "one uint16 array",
        ),
        "wrong_payload_shape": _reject_linearization(
            [
                (
                    "0",
                    _page({50712: _tag(np.asarray([[0, 1]], dtype=np.uint16))}),
                )
            ],
            "one uint16 array",
        ),
        "count_payload_mismatch": _reject_linearization(
            [
                (
                    "0",
                    _page(
                        {
                            50712: _tag(
                                np.asarray([0, 1], dtype=np.uint16), count=3
                            )
                        }
                    ),
                )
            ],
            "one uint16 array",
        ),
    }


def _reject_prior(call: Callable[[], Any], expected: str) -> bool:
    try:
        call()
    except DngForwardRasterError as exc:
        return expected in str(exc)
    return False


def _prior_guard_controls() -> dict[str, bool]:
    tag = SimpleNamespace(value=b"")
    required_opcode = (
        struct.pack(">I", 1)
        + struct.pack(">IIII", 5, 0x01030000, 0, 0)
    )
    return {
        "huesatmap": _reject_prior(
            lambda: _guard_unsupported_profile_huesatmap(
                [("0", _page({50937: tag}))]  # type: ignore[list-item]
            ),
            "ProfileHueSatMap",
        ),
        "gain_table_map": _reject_prior(
            lambda: _guard_unsupported_profile_gain_table_map(
                [("0", _page({52525: tag}))]  # type: ignore[list-item]
            ),
            "ProfileGainTableMap",
        ),
        "tone_curve": _reject_prior(
            lambda: _guard_unsupported_profile_tone_curve(
                [("0", _page({50940: tag}))]  # type: ignore[list-item]
            ),
            "ProfileToneCurve",
        ),
        "look_table": _reject_prior(
            lambda: _guard_unsupported_profile_look_table(
                [("0", _page({50981: tag}))]  # type: ignore[list-item]
            ),
            "ProfileLookTable",
        ),
        "required_opcode": _reject_prior(
            lambda: _guard_dng_opcode_lists(
                [("0", _page({51008: SimpleNamespace(value=required_opcode)}))]  # type: ignore[list-item]
            ),
            "required DNG OpcodeList1",
        ),
        "missing_version": _reject_prior(lambda: _guard_dng_version({}), "missing"),
    }


def _inventory(path: Path) -> dict[str, Any] | None:
    with tifffile.TiffFile(path) as document:
        pages = _walk_pages(document.pages)
        return _linearization_table_delegation(pages)


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
        warning_codes = [warning.code for warning in working.warnings]
        warning_messages = {
            warning.code: warning.message for warning in working.warnings
        }
        expected_warning_codes = list(BASE_WARNING_CODES)
        if row["source_id"] != "blackmagic_pocket_cinema_camera_4k":
            expected_warning_codes.append("optional_dng_opcodes_may_be_skipped")
        if inventory is not None:
            expected_warning_codes.append(
                "dng_linearization_table_delegated_to_libraw"
            )
        expected_delegation_message = None
        if inventory is not None:
            expected_delegation_message = (
                "DNG LinearizationTable was structurally validated before camera "
                "decode but its arithmetic remains delegated to LibRaw; "
                f"tag={inventory['tag_code']}, ifd={inventory['ifd_path']}, "
                f"count={inventory['count']}, payload_u16le_sha256="
                f"{inventory['payload_u16le_sha256']}."
            )
        output_sha = _sha256_bytes(working.pixels.astype("<f4", copy=False).tobytes())
        results.append(
            {
                "source_id": row["source_id"],
                "source_bytes": source.stat().st_size,
                "source_sha256": before,
                "source_immutable": _sha256_file(source) == before,
                "inventory": inventory,
                "warning_codes": warning_codes,
                "expected_warning_codes": expected_warning_codes,
                "warning_codes_exact": warning_codes == expected_warning_codes,
                "delegation_message_exact": warning_messages.get(
                    "dng_linearization_table_delegated_to_libraw"
                )
                == expected_delegation_message,
                "output_float32_sha256": output_sha,
                "output_matches_p98": output_sha == expected_outputs[row["source_id"]],
                "working_contract_exact": all(
                    (
                        working.pixels.dtype == np.float32,
                        working.pixels.ndim == 3,
                        working.pixels.shape[2] == 3,
                        working.working_space == "linear_rec2020",
                        working.transfer_state == "scene_linear",
                        working.source_transfer_state == "scene_linear",
                        working.alpha_policy == "absent",
                    )
                ),
                "finite": bool(np.all(np.isfinite(working.pixels))),
            }
        )
    results.sort(key=lambda item: item["source_id"])
    controls = _linearization_controls()
    prior_controls = _prior_guard_controls()
    expected_inventory = config["expected_inventory"]
    authority = config["authority"]
    gates = {
        "authority_exact": all(
            (
                _sha256_file(ROOT / authority["sdk_path"]) == authority["sdk_sha256"],
                _sha256_file(ROOT / authority["tag_codes_source_path"])
                == authority["tag_codes_source_sha256"],
                _sha256_file(p98_path) == config["bindings"]["p98_config_sha256"],
                _sha256_file(p244_path) == config["bindings"]["p244_config_sha256"],
            )
        ),
        "inventory_exact": all(
            (
                item["inventory"] is None
                and expected_inventory[item["source_id"]] is None
            )
            or (
                item["inventory"] is not None
                and item["inventory"]["tag_code"] == authority["tag_code"]
                and {
                    key: value
                    for key, value in item["inventory"].items()
                    if key != "tag_code"
                }
                == expected_inventory[item["source_id"]]
            )
            for item in results
        ),
        "invalid_reject_before_decode": all(
            item["rejected"] and item["camera_decode_calls"] == 0
            for item in controls.values()
        ),
        "explicit_warning_exact": all(
            item["warning_codes_exact"] and item["delegation_message_exact"]
            for item in results
        ),
        "p98_pixel_hashes_exact": all(item["output_matches_p98"] for item in results),
        "prior_guards_exact": all(prior_controls.values()),
        "working_contract_exact": all(item["working_contract_exact"] for item in results),
        "source_immutable": all(item["source_immutable"] for item in results),
        "finite": all(item["finite"] for item in results),
    }
    status = (
        "PASS_PRIVATE_DNG_LINEARIZATION_TABLE_DELEGATION"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_LINEARIZATION_TABLE_DELEGATION"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": "P308",
        "status": status,
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "contract_sha256": _sha256_file(
                ROOT / config["bindings"]["contract_path"]
            ),
            "implementation_sha256": _sha256_file(
                ROOT / config["bindings"]["implementation_path"]
            ),
            "runner_sha256": _sha256_file(ROOT / config["bindings"]["runner_path"]),
            "test_sha256": _sha256_file(ROOT / config["bindings"]["test_path"]),
            "sdk_sha256": _sha256_file(ROOT / authority["sdk_path"]),
            "p98_config_sha256": _sha256_file(p98_path),
            "p244_config_sha256": _sha256_file(p244_path),
        },
        "controls": controls,
        "prior_guard_controls": prior_controls,
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
