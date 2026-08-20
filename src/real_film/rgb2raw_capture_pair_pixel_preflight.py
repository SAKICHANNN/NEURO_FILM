"""Bounded member/pixel preflight for the NTIRE 2025 RGB2RAW source."""

from __future__ import annotations

import io
import json
import pickletools
import struct
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    canonical_sha256,
    http_range_get,
    parse_central_directory,
    sha256_bytes,
)


class RGB2RAWPixelPreflightError(ValueError):
    """Raised when a frozen member or pixel contract is invalid."""


RangeReader = Callable[[str, int, int, int], bytes]


def extract_member(
    url: str,
    member: ZipMember,
    archive_size: int,
    range_reader: RangeReader,
) -> tuple[bytes, int]:
    header = range_reader(
        url, member.local_offset, member.local_offset + 29, archive_size
    )
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise RGB2RAWPixelPreflightError("invalid local-header signature")
    name_length, extra_length = values[-2:]
    body_start = member.local_offset + 30 + name_length + extra_length
    compressed = range_reader(
        url,
        body_start,
        body_start + member.compressed_size - 1,
        archive_size,
    )
    if member.method == 0:
        payload = compressed
    elif member.method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RGB2RAWPixelPreflightError("unsupported compression method")
    if len(payload) != member.uncompressed_size:
        raise RGB2RAWPixelPreflightError("uncompressed size differs")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != member.crc32:
        raise RGB2RAWPixelPreflightError("member CRC differs")
    return payload, len(header) + len(compressed)


def _rank(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    return (ranks.reshape(values.shape) + 0.5) / flat.size


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False).reshape(-1)
    bb = b.astype(np.float64, copy=False).reshape(-1)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom else 0.0


def _census_agreement(a: np.ndarray, b: np.ndarray) -> float:
    agreements = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ay = slice(max(0, dy), a.shape[0] + min(0, dy))
        ax = slice(max(0, dx), a.shape[1] + min(0, dx))
        by = slice(max(0, -dy), a.shape[0] + min(0, -dy))
        bx = slice(max(0, -dx), a.shape[1] + min(0, -dx))
        agreements.append(np.mean((a[ay, ax] > a[by, bx]) == (b[ay, ax] > b[by, bx])))
    return float(np.mean(agreements))


def _pickle_facts(payload: bytes) -> dict[str, Any]:
    opcode_count = 0
    global_symbols: list[str] = []
    stopped = False
    forbidden = False
    try:
        for opcode, argument, _ in pickletools.genops(payload):
            opcode_count += 1
            if opcode.name in ("GLOBAL", "STACK_GLOBAL"):
                global_symbols.append(str(argument))
            if opcode.name in ("PERSID", "BINPERSID", "EXT1", "EXT2", "EXT4"):
                forbidden = True
            stopped = opcode.name == "STOP"
    except ValueError:
        return {"valid": False, "opcode_count": opcode_count, "global_symbols": []}
    return {
        "valid": stopped and not forbidden and opcode_count < 1_000_000,
        "opcode_count": opcode_count,
        "global_symbols": sorted(set(global_symbols)),
    }


def _row_metrics(
    raw_payload: bytes, png_payload: bytes, metadata_payload: bytes
) -> dict[str, Any]:
    raw = np.load(io.BytesIO(raw_payload), allow_pickle=False)
    with Image.open(io.BytesIO(png_payload)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"))
        small = (
            np.asarray(
                image.convert("RGB").resize((512, 512), Image.Resampling.BOX),
                dtype=np.float64,
            )
            / 255.0
        )
    raw64 = np.asarray(raw, dtype=np.float64)
    raw_luma = raw64.mean(axis=2)
    display_luma = small.mean(axis=2)
    raw_rank = _rank(raw_luma)
    display_rank = _rank(display_luma)
    raw_gradient = np.hypot(*np.gradient(raw_rank))
    display_gradient = np.hypot(*np.gradient(display_rank))
    raw_rgb = np.stack(
        (raw64[..., 0], 0.5 * (raw64[..., 1] + raw64[..., 2]), raw64[..., 3]),
        axis=2,
    )
    lo = np.percentile(raw_rgb, 0.5, axis=(0, 1), keepdims=True)
    hi = np.percentile(raw_rgb, 99.5, axis=(0, 1), keepdims=True)
    raw_proxy = np.clip((raw_rgb - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    dtype_limits = (
        np.iinfo(raw.dtype)
        if np.issubdtype(raw.dtype, np.integer)
        else np.finfo(raw.dtype)
    )
    return {
        "npy_shape": list(raw.shape),
        "npy_dtype": str(raw.dtype),
        "npy_dtype_min": float(dtype_limits.min),
        "npy_dtype_max": float(dtype_limits.max),
        "npy_finite": bool(np.isfinite(raw64).all()),
        "npy_min": float(raw64.min()),
        "npy_max": float(raw64.max()),
        "png_shape": list(rgb.shape),
        "png_dtype": str(rgb.dtype),
        "census_agreement": _census_agreement(raw_rank, display_rank),
        "gradient_correlation": _corr(raw_gradient, display_gradient),
        "display_raw_proxy_mean_abs_difference": float(
            np.mean(np.abs(small - raw_proxy))
        ),
        "pickle": _pickle_facts(metadata_payload),
        "raw_sha256": sha256_bytes(raw_payload),
        "png_sha256": sha256_bytes(png_payload),
        "metadata_sha256": sha256_bytes(metadata_payload),
    }


def run_pixel_preflight(
    config_path: Path, *, range_reader: RangeReader = http_range_get
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != (
        "neuro-film.sf3-a0s-rgb2raw-capture-pair-pixel-preflight-contract.v1"
    ):
        raise RGB2RAWPixelPreflightError("contract schema differs")
    source = config["source"]
    url = (
        f"https://huggingface.co/datasets/{source['dataset_id']}/resolve/"
        f"{source['revision']}/{source['archive_path']}?download=true"
    )
    central = range_reader(
        url,
        source["central_offset"],
        source["central_offset"] + source["central_size"] - 1,
        source["archive_size"],
    )
    if sha256_bytes(central) != source["central_sha256"]:
        raise RGB2RAWPixelPreflightError("central-directory identity differs")
    members = {member.name: member for member in parse_central_directory(central)}
    rows = config["rows"]
    if canonical_sha256(rows) != config["selection_manifest_sha256"]:
        raise RGB2RAWPixelPreflightError("selection manifest identity differs")
    result_rows = []
    network_bytes = len(central)
    for row in rows:
        payloads = {}
        member_rows = {}
        for role in ("png", "npy", "metadata"):
            name = row[role]
            if name not in members:
                raise RGB2RAWPixelPreflightError(f"selected member missing: {name}")
            payload, consumed = extract_member(
                url, members[name], source["archive_size"], range_reader
            )
            network_bytes += consumed
            payloads[role] = payload
            member_rows[role] = {
                "name": name,
                "crc32": members[name].crc32,
                "compressed_size": members[name].compressed_size,
                "uncompressed_size": members[name].uncompressed_size,
            }
        result_rows.append(
            {
                "camera_group": row["camera_group"],
                "scene_group": row["scene_group"],
                "canonical_scene_group": row["canonical_scene_group"],
                "pair_stem": row["pair_stem"],
                "members": member_rows,
                **_row_metrics(payloads["npy"], payloads["png"], payloads["metadata"]),
            }
        )
    gates_config = config["gates"]
    counts = Counter(row["camera_group"] for row in result_rows)
    gates = {
        "row_count_exact": len(result_rows) == gates_config["required_row_count"],
        "rows_per_camera_exact": all(
            counts[camera] == gates_config["required_rows_per_camera"]
            for camera in ("iphone-x", "samsung-s9")
        ),
        "canonical_scene_groups_unique": len(
            {(row["camera_group"], row["canonical_scene_group"]) for row in result_rows}
        )
        == gates_config["required_unique_canonical_scene_groups"],
        "npy_shape_each": all(
            row["npy_shape"] == gates_config["required_npy_shape"]
            for row in result_rows
        ),
        "png_shape_each": all(
            row["png_shape"] == gates_config["required_png_shape"]
            for row in result_rows
        ),
        "npy_dtype_each": all(
            row["npy_dtype"] == gates_config["required_npy_dtype"]
            for row in result_rows
        ),
        "finite_dtype_bounded_raw_each": all(
            row["npy_finite"]
            and row["npy_min"] >= row["npy_dtype_min"]
            and row["npy_max"] <= row["npy_dtype_max"]
            for row in result_rows
        ),
        "census_agreement_each": all(
            row["census_agreement"] >= gates_config["minimum_census_agreement"]
            for row in result_rows
        ),
        "gradient_correlation_each": all(
            row["gradient_correlation"] >= gates_config["minimum_gradient_correlation"]
            for row in result_rows
        ),
        "display_raw_proxy_difference_each": all(
            row["display_raw_proxy_mean_abs_difference"]
            >= gates_config["minimum_display_raw_proxy_difference"]
            for row in result_rows
        ),
        "valid_nonexecuted_pickle_stream_each": all(
            row["pickle"]["valid"] for row in result_rows
        ),
        "bounded_network_bytes": network_bytes
        <= config["network_limits"]["maximum_total_bytes"],
        "zero_persisted_member_payloads": not config["network_limits"][
            "persist_member_payloads"
        ],
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0s-rgb2raw-capture-pair-pixel-preflight-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source_lock_stable_identity": source["source_lock_stable_identity"],
        "selection_manifest_sha256": config["selection_manifest_sha256"],
        "rows": result_rows,
        "network_bytes_read": network_bytes,
        "member_payloads_persisted": 0,
        "operator_fits": 0,
        "renders": 0,
        "scores": 0,
        "bounded_final_candidate_counter_before": 0,
        "bounded_final_candidate_counter_after": 0,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
