"""U6.P4CH bounded official-Newson versus exact-spectrum Gaussian audit."""

from __future__ import annotations

import binascii
import hashlib
import io
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import label

from src.eval.boolean_gaussian_lod_boundary import (
    synthesize_exact_spectrum_gaussian,
)

SCHEMA = "neuro_film.u6_p4ch_official_newson_gaussian_lod_contract.v1"
SOURCE_SCHEMA = "neuro_film.u6_p4ch_official_newson_source_lock.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ch_official_newson_gaussian_lod_report.v1"
ARCHIVE_ROOT = "NeuralFilmGrainRendering_Dataset"


class OfficialNewsonLodError(RuntimeError):
    """Raised when the frozen P4CH source or analysis contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> None:
    archive = contract.get("archive", {})
    selection = contract.get("bounded_source_selection", {})
    comparison = contract.get("comparison", {})
    gates = contract.get("automatic_gates", {})
    expected_indices = selection.get("selected_indices_by_radius", {})
    if (
        contract.get("schema") != SCHEMA
        or archive.get("version") != 1
        or archive.get("content_length_bytes") != 2_354_389_185
        or archive.get("entry_count") != 8_722
        or archive.get("test_hyperparameters_rows") != 4_000
        or archive.get("png_member_bytes_read_before_freeze") is not False
        or selection.get("split") != "test"
        or selection.get("radii_input_pixels") != [0.05, 0.2, 0.4, 0.8]
        or selection.get("rows_per_radius") != 16
        or selection.get("selection_salt") != "u6.p4ch-v1"
        or sorted(expected_indices) != ["0.05", "0.2", "0.4", "0.8"]
        or any(len(expected_indices[key]) != 16 for key in expected_indices)
        or len({index for rows in expected_indices.values() for index in rows}) != 64
        or comparison.get("gaussian_seed_rule")
        != "first 64 bits of SHA-256('u6.p4ch-gaussian-v1|row_index|replicate_index')"
        or comparison.get("feature_families", {})
        .get("local_rms", {})
        .get("block_pixels")
        != 16
        or gates.get("maximum_small_radius_median_ratio") != 1.5
        or gates.get("minimum_large_radius_median_ratio") != 2.0
        or gates.get("minimum_large_to_small_ratio_of_medians") != 1.5
        or gates.get("minimum_rows_per_radius") != 16
        or gates.get("maximum_periodogram_relative_error") != 1e-12
        or gates.get("require_two_byte_identical_reports") is not True
    ):
        raise OfficialNewsonLodError("P4CH frozen contract drift")


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _zip64_values(extra: bytes) -> list[int]:
    offset = 0
    while offset + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, offset)
        value = extra[offset + 4 : offset + 4 + size]
        offset += 4 + size
        if tag == 1:
            if len(value) % 8:
                raise OfficialNewsonLodError("malformed ZIP64 extra")
            return [
                struct.unpack_from("<Q", value, index)[0]
                for index in range(0, len(value), 8)
            ]
    return []


def parse_central_directory(data: bytes) -> dict[str, ZipEntry]:
    entries: dict[str, ZipEntry] = {}
    offset = 0
    while offset < len(data):
        if data[offset : offset + 4] != b"PK\x01\x02":
            raise OfficialNewsonLodError(
                f"bad central-directory signature at {offset}"
            )
        fields = struct.unpack_from("<4s6H3L5H2L", data, offset)
        name_length, extra_length, comment_length = fields[10:13]
        name_start = offset + 46
        name_end = name_start + name_length
        extra_end = name_end + extra_length
        try:
            name = data[name_start:name_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OfficialNewsonLodError("non-UTF-8 archive member") from exc
        extra = data[name_end:extra_end]
        uncompressed_size = fields[9]
        compressed_size = fields[8]
        local_header_offset = fields[16]
        values = iter(_zip64_values(extra))
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = next(values)
        if compressed_size == 0xFFFFFFFF:
            compressed_size = next(values)
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = next(values)
        if name in entries:
            raise OfficialNewsonLodError(f"duplicate archive member: {name}")
        entries[name] = ZipEntry(
            name=name,
            compression_method=int(fields[4]),
            crc32=int(fields[7]),
            compressed_size=int(compressed_size),
            uncompressed_size=int(uncompressed_size),
            local_header_offset=int(local_header_offset),
        )
        offset = extra_end + comment_length
    return entries


class HttpRangeArchive:
    """Minimal bounded HTTP ZIP reader with no full-archive fallback."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self._contract = contract
        self._url = str(contract["archive"]["public_download_endpoint"])
        self._session = requests.Session()
        self.request_count = 0
        self.member_compressed_bytes = 0
        self.member_uncompressed_bytes = 0
        self.content_length = 0
        self.etag = ""
        self.entries: dict[str, ZipEntry] = {}

    def _get_range(self, start: int, end: int, *, count_member: bool = False) -> bytes:
        if start < 0 or end < start:
            raise OfficialNewsonLodError("invalid HTTP range")
        limit = int(self._contract["bounded_source_selection"]["maximum_http_requests"])
        if self.request_count >= limit:
            raise OfficialNewsonLodError("HTTP request bound exhausted")
        response = self._session.get(
            self._url,
            headers={"Range": f"bytes={start}-{end}"},
            allow_redirects=True,
            timeout=120,
        )
        self.request_count += 1
        if response.status_code != 206:
            raise OfficialNewsonLodError(
                f"server rejected bounded range: {response.status_code}"
            )
        expected = end - start + 1
        if len(response.content) != expected:
            raise OfficialNewsonLodError("HTTP range length drift")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise OfficialNewsonLodError("HTTP Content-Range drift")
        if count_member:
            self.member_compressed_bytes += len(response.content)
        return response.content

    def open(self) -> None:
        probe = self._get_range(0, 0)
        if len(probe) != 1:
            raise OfficialNewsonLodError("archive probe drift")
        response = self._session.get(
            self._url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            stream=True,
            timeout=120,
        )
        self.request_count += 1
        try:
            if response.status_code != 206:
                raise OfficialNewsonLodError("archive identity probe rejected")
            content_range = response.headers.get("Content-Range", "")
            self.content_length = int(content_range.rsplit("/", 1)[1])
            self.etag = response.headers.get("ETag", "").strip('"')
        finally:
            response.close()
        expected = self._contract["archive"]
        if (
            self.content_length != expected["content_length_bytes"]
            or self.etag != expected["etag_md5"]
        ):
            raise OfficialNewsonLodError("archive identity drift")

        tail_size = min(2_000_000, self.content_length)
        tail_start = self.content_length - tail_size
        tail = self._get_range(tail_start, self.content_length - 1)
        eocd_offset = tail.rfind(b"PK\x05\x06")
        if eocd_offset < 0:
            raise OfficialNewsonLodError("ZIP EOCD missing")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
        entry_count, directory_size, directory_offset = eocd[4], eocd[5], eocd[6]
        if (
            entry_count != expected["entry_count"]
            or directory_size != expected["central_directory_size"]
            or directory_offset != expected["central_directory_offset"]
        ):
            raise OfficialNewsonLodError("central-directory identity drift")
        if not (
            tail_start <= directory_offset
            and directory_offset + directory_size <= self.content_length
        ):
            raise OfficialNewsonLodError("central directory escaped bounded tail")
        start = directory_offset - tail_start
        central = tail[start : start + directory_size]
        self.entries = parse_central_directory(central)
        if len(self.entries) != entry_count:
            raise OfficialNewsonLodError("central-directory entry count drift")

    def read_member(self, name: str) -> bytes:
        if name not in self.entries:
            raise OfficialNewsonLodError(f"missing archive member: {name}")
        entry = self.entries[name]
        if entry.compression_method not in (0, 8):
            raise OfficialNewsonLodError("unsupported ZIP compression method")
        header = self._get_range(
            entry.local_header_offset, entry.local_header_offset + 4095
        )
        if header[:4] != b"PK\x03\x04":
            raise OfficialNewsonLodError("local ZIP header missing")
        local = struct.unpack_from("<4s5H3L2H", header, 0)
        name_length, extra_length = local[9], local[10]
        local_name = header[30 : 30 + name_length].decode("utf-8")
        if local_name != name:
            raise OfficialNewsonLodError("central/local member identity drift")
        data_offset = entry.local_header_offset + 30 + name_length + extra_length
        compressed = self._get_range(
            data_offset,
            data_offset + entry.compressed_size - 1,
            count_member=True,
        )
        if entry.compression_method == 8:
            try:
                result = zlib.decompress(compressed, -15)
            except zlib.error as exc:
                raise OfficialNewsonLodError("member DEFLATE failed") from exc
        else:
            result = compressed
        if (
            len(result) != entry.uncompressed_size
            or (binascii.crc32(result) & 0xFFFFFFFF) != entry.crc32
        ):
            raise OfficialNewsonLodError("member CRC or size drift")
        self.member_uncompressed_bytes += len(result)
        bound = int(
            self._contract["bounded_source_selection"][
                "maximum_downloaded_member_bytes"
            ]
        )
        if self.member_compressed_bytes > bound or self.member_uncompressed_bytes > bound:
            raise OfficialNewsonLodError("member byte bound exhausted")
        return result


def _selected_rows(
    contract: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    entries: Mapping[str, ZipEntry],
) -> list[dict[str, Any]]:
    selection = contract["bounded_source_selection"]
    salt = str(selection["selection_salt"])
    used: set[str] = set()
    selected: list[dict[str, Any]] = []
    for radius in selection["radii_input_pixels"]:
        candidates: list[tuple[str, int, Mapping[str, Any]]] = []
        for index, row in enumerate(metadata_rows):
            image_name = str(row["img_name"])
            grain_name = f"{index:06d}.png"
            clean_path = f"{ARCHIVE_ROOT}/Mixed/test/{image_name}"
            grain_path = f"{ARCHIVE_ROOT}/Mixed_grain/test/{grain_name}"
            if (
                float(row["radius"]) == float(radius)
                and clean_path in entries
                and grain_path in entries
            ):
                key = _sha256(
                    f"{salt}|test|{index}|{image_name}|{row['seed']}|{row['radius']}".encode()
                )
                candidates.append((key, index, row))
        chosen: list[dict[str, Any]] = []
        for key, index, row in sorted(candidates):
            image_name = str(row["img_name"])
            if image_name in used:
                continue
            used.add(image_name)
            chosen.append(
                {
                    "index": index,
                    "img_name": image_name,
                    "grain_member": f"{index:06d}.png",
                    "seed": int(row["seed"]),
                    "radius": float(row["radius"]),
                    "zoom": float(row["zoom"]),
                    "selection_key": key,
                }
            )
            if len(chosen) == int(selection["rows_per_radius"]):
                break
        expected = selection["selected_indices_by_radius"][str(radius)]
        if [row["index"] for row in chosen] != expected:
            raise OfficialNewsonLodError("selected row identity drift")
        selected.extend(chosen)
    return selected


def _decode_png(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "L" or image.n_frames != 1:
                raise OfficialNewsonLodError("PNG mode, format or frame count drift")
            values = np.asarray(image, dtype=np.uint8).copy()
    except OfficialNewsonLodError:
        raise
    except Exception as exc:
        raise OfficialNewsonLodError("PNG decode failed") from exc
    if values.ndim != 2 or min(values.shape) <= 17:
        raise OfficialNewsonLodError("PNG geometry is invalid")
    values.setflags(write=False)
    return values


def acquire_selected_pairs(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], np.ndarray, np.ndarray]]]:
    validate_contract(contract)
    archive = HttpRangeArchive(contract)
    archive.open()
    metadata_name = f"{ARCHIVE_ROOT}/Mixed_grain/test/hyperparameters.json"
    metadata_bytes = archive.read_member(metadata_name)
    if _sha256(metadata_bytes) != contract["archive"]["test_hyperparameters_sha256"]:
        raise OfficialNewsonLodError("test hyperparameters SHA-256 drift")
    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as exc:
        raise OfficialNewsonLodError("test hyperparameters JSON failed") from exc
    rows = metadata.get("data")
    if not isinstance(rows, list) or len(rows) != 4_000:
        raise OfficialNewsonLodError("test hyperparameters row count drift")

    clean_names = {
        name
        for name in archive.entries
        if name.startswith(f"{ARCHIVE_ROOT}/Mixed/test/") and name.endswith(".png")
    }
    grain_names = {
        name
        for name in archive.entries
        if name.startswith(f"{ARCHIVE_ROOT}/Mixed_grain/test/")
        and name.endswith(".png")
    }
    expected_archive = contract["archive"]
    if (
        len(clean_names) != expected_archive["test_clean_png_count"]
        or len(grain_names) != expected_archive["test_grain_png_count"]
        or 4_000 - len(grain_names)
        != expected_archive["known_missing_test_grain_rows"]
    ):
        raise OfficialNewsonLodError("test member inventory drift")

    selected = _selected_rows(contract, rows, archive.entries)
    clean_cache: dict[str, tuple[bytes, np.ndarray]] = {}
    acquired: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    source_rows: list[dict[str, Any]] = []
    for row in selected:
        clean_member = f"{ARCHIVE_ROOT}/Mixed/test/{row['img_name']}"
        grain_member = f"{ARCHIVE_ROOT}/Mixed_grain/test/{row['grain_member']}"
        if clean_member not in clean_cache:
            clean_bytes = archive.read_member(clean_member)
            clean_cache[clean_member] = (clean_bytes, _decode_png(clean_bytes))
        clean_bytes, clean = clean_cache[clean_member]
        grain_bytes = archive.read_member(grain_member)
        grain = _decode_png(grain_bytes)
        if clean.shape != grain.shape:
            raise OfficialNewsonLodError("clean/grain shape drift")
        acquired.append((row, clean, grain))
        source_rows.append(
            {
                **row,
                "clean_member": clean_member,
                "clean_member_sha256": _sha256(clean_bytes),
                "grain_member_path": grain_member,
                "grain_member_sha256": _sha256(grain_bytes),
                "shape": list(clean.shape),
                "mode": "L",
            }
        )

    source_lock = {
        "schema": SOURCE_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "archive": {
            "version": contract["archive"]["version"],
            "content_length_bytes": archive.content_length,
            "etag_md5": archive.etag,
            "entry_count": len(archive.entries),
            "test_hyperparameters_sha256": _sha256(metadata_bytes),
        },
        "request_count": archive.request_count,
        "member_compressed_bytes": archive.member_compressed_bytes,
        "member_uncompressed_bytes": archive.member_uncompressed_bytes,
        "selected_rows": source_rows,
        "claim_ceiling": contract["claim_ceiling"],
    }
    source_lock["source_lock_id"] = _sha256(canonical_json(source_lock))
    return source_lock, acquired


def _standardize(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    centered = field - float(np.mean(field, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise OfficialNewsonLodError("residual variance is zero or non-finite")
    result = np.ascontiguousarray(centered / rms, dtype=np.float64)
    result.setflags(write=False)
    return result


def _periodogram(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    result = np.square(np.abs(np.fft.fft2(field))) / field.size
    result = np.ascontiguousarray(result, dtype=np.float64)
    result[0, 0] = 0.0
    result.setflags(write=False)
    return result


def higher_order_features(
    values: np.ndarray, contract: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    field = _standardize(values)
    definition = contract["comparison"]["feature_families"]
    marginal = np.quantile(
        field,
        np.asarray(definition["marginal_quantiles"], dtype=np.float64),
        method="linear",
    )
    block = int(definition["local_rms"]["block_pixels"])
    if field.shape[0] % block or field.shape[1] % block:
        raise OfficialNewsonLodError("feature field is not block divisible")
    local_rms = np.sqrt(
        np.mean(
            np.square(
                field.reshape(
                    field.shape[0] // block,
                    block,
                    field.shape[1] // block,
                    block,
                )
            ),
            axis=(1, 3),
            dtype=np.float64,
        )
    )
    energy = np.quantile(
        local_rms,
        np.asarray(definition["local_rms"]["quantiles"], dtype=np.float64),
        method="linear",
    )
    topology: list[float] = []
    connectivity = np.ones((3, 3), dtype=np.uint8)
    per_megapixel = 1_000_000.0 / field.size
    excursion = definition["excursion_topology"]
    for threshold in excursion["thresholds_sigma"]:
        for mask in (field >= float(threshold), field <= -float(threshold)):
            components, count = label(mask, structure=connectivity)
            sizes = np.bincount(components.reshape(-1), minlength=count + 1)[1:]
            for lower, upper in excursion["component_area_bins_pixels"]:
                number = int(np.count_nonzero((sizes >= lower) & (sizes <= upper)))
                topology.append(math.log1p(number * per_megapixel))
    result = {
        "marginal_quantiles": np.ascontiguousarray(marginal, dtype=np.float64),
        "local_rms_quantiles": np.ascontiguousarray(energy, dtype=np.float64),
        "excursion_topology": np.ascontiguousarray(topology, dtype=np.float64),
    }
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise OfficialNewsonLodError("non-finite higher-order feature")
    return result


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(left - right), dtype=np.float64)))


def _ratio_json(value: float) -> float | str:
    return "infinite" if math.isinf(value) else float(value)


def _median_with_infinity(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _evaluate_pair(
    row: Mapping[str, Any],
    clean: np.ndarray,
    grain: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    clean_work = np.asarray(clean[1:, 1:], dtype=np.float64) / 255.0
    grain_work = np.asarray(grain[1:, 1:], dtype=np.float64) / 255.0
    block = int(
        contract["comparison"]["feature_families"]["local_rms"]["block_pixels"]
    )
    height = (clean_work.shape[0] // block) * block
    width = (clean_work.shape[1] // block) * block
    residual = _standardize(
        np.ascontiguousarray(
            grain_work[:height, :width] - clean_work[:height, :width],
            dtype=np.float64,
        )
    )
    spectrum = _periodogram(residual)
    target_features = higher_order_features(residual, contract)
    gaussian_features: list[dict[str, np.ndarray]] = []
    maximum_error = 0.0
    gaussian_hashes: list[str] = []
    for replicate in range(4):
        seed_digest = hashlib.sha256(
            f"u6.p4ch-gaussian-v1|{row['index']}|{replicate}".encode()
        ).hexdigest()
        gaussian, error = synthesize_exact_spectrum_gaussian(
            spectrum, seed=int(seed_digest[:16], 16)
        )
        maximum_error = max(maximum_error, error)
        gaussian_hashes.append(_sha256(np.ascontiguousarray(gaussian).tobytes()))
        gaussian_features.append(higher_order_features(gaussian, contract))

    family_rows: dict[str, Any] = {}
    ratios: list[float] = []
    for name, target in target_features.items():
        values = np.asarray([features[name] for features in gaussian_features])
        center = np.median(values, axis=0)
        target_distance = _rmse(target, center)
        pairwise = [
            _rmse(values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        stochastic_distance = float(np.median(pairwise))
        if stochastic_distance == 0.0:
            ratio = 0.0 if target_distance == 0.0 else math.inf
        else:
            ratio = target_distance / stochastic_distance
        ratios.append(ratio)
        family_rows[name] = {
            "target_to_gaussian_rmse": target_distance,
            "gaussian_pairwise_median_rmse": stochastic_distance,
            "distance_ratio": _ratio_json(ratio),
        }
    row_ratio = _median_with_infinity(ratios)
    result = {
        "index": int(row["index"]),
        "img_name": str(row["img_name"]),
        "radius": float(row["radius"]),
        "feature_shape": [height, width],
        "residual_sha256": _sha256(np.ascontiguousarray(residual).tobytes()),
        "residual_rms_before_standardization": float(
            np.sqrt(
                np.mean(
                    np.square(grain_work[:height, :width] - clean_work[:height, :width]),
                    dtype=np.float64,
                )
            )
        ),
        "output_boundary_fraction": float(
            np.mean(
                (grain_work[:height, :width] <= 0.0)
                | (grain_work[:height, :width] >= 1.0)
            )
        ),
        "gaussian_sha256": gaussian_hashes,
        "feature_families": family_rows,
        "median_family_distance_ratio": _ratio_json(row_ratio),
        "maximum_periodogram_relative_error": maximum_error,
    }
    return result, row_ratio


def evaluate_official_newson_gaussian_lod(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_lock, acquired = acquire_selected_pairs(contract)
    row_results: list[dict[str, Any]] = []
    ratios_by_radius: dict[float, list[float]] = {
        float(radius): []
        for radius in contract["bounded_source_selection"]["radii_input_pixels"]
    }
    maximum_error = 0.0
    for row, clean, grain in acquired:
        result, ratio = _evaluate_pair(row, clean, grain, contract)
        row_results.append(result)
        ratios_by_radius[float(row["radius"])].append(ratio)
        maximum_error = max(maximum_error, result["maximum_periodogram_relative_error"])

    radius_results: list[dict[str, Any]] = []
    radius_medians: dict[float, float] = {}
    for radius in contract["bounded_source_selection"]["radii_input_pixels"]:
        numeric_radius = float(radius)
        rows = [row for row in row_results if row["radius"] == numeric_radius]
        median_ratio = _median_with_infinity(ratios_by_radius[numeric_radius])
        radius_medians[numeric_radius] = median_ratio
        family_medians: dict[str, float | str] = {}
        for family in (
            "marginal_quantiles",
            "local_rms_quantiles",
            "excursion_topology",
        ):
            values = [
                math.inf
                if row["feature_families"][family]["distance_ratio"] == "infinite"
                else float(row["feature_families"][family]["distance_ratio"])
                for row in rows
            ]
            family_medians[family] = _ratio_json(_median_with_infinity(values))
        radius_results.append(
            {
                "radius": numeric_radius,
                "row_count": len(rows),
                "median_family_distance_ratio": _ratio_json(median_ratio),
                "family_median_distance_ratios": family_medians,
                "median_output_boundary_fraction": float(
                    np.median([row["output_boundary_fraction"] for row in rows])
                ),
            }
        )

    gates = contract["automatic_gates"]
    gate_results = {
        "row_counts": all(
            len(values) >= int(gates["minimum_rows_per_radius"])
            for values in ratios_by_radius.values()
        ),
        "small_radius_adequacy": all(
            radius_medians[float(radius)]
            <= float(gates["maximum_small_radius_median_ratio"])
            for radius in gates["small_radii_requiring_adequacy"]
        ),
        "large_radius_divergence": all(
            radius_medians[float(radius)]
            >= float(gates["minimum_large_radius_median_ratio"])
            for radius in gates["large_radii_requiring_divergence"]
        ),
        "scale_separation": (
            min(
                radius_medians[float(radius)]
                for radius in gates["large_radii_requiring_divergence"]
            )
            / max(
                max(
                    radius_medians[float(radius)]
                    for radius in gates["small_radii_requiring_adequacy"]
                ),
                np.finfo(float).tiny,
            )
            >= float(gates["minimum_large_to_small_ratio_of_medians"])
        ),
        "periodogram_exact": maximum_error
        <= float(gates["maximum_periodogram_relative_error"]),
    }
    passed = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "source_lock_id": source_lock["source_lock_id"],
        "selected_row_count": len(row_results),
        "radius_results": radius_results,
        "row_results": row_results,
        "maximum_periodogram_relative_error": maximum_error,
        "gates": gate_results,
        "passed": passed,
        "decision": (
            "retain_official_newson_gaussian_lod_boundary_research_only"
            if passed
            else "close_official_newson_gaussian_lod_boundary_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["report_id"] = _sha256(canonical_json(report))
    return source_lock, report


def write_json(value: Mapping[str, Any], path: Path) -> str:
    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)
