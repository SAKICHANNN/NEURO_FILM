"""Range acquisition and cross-scanner coherence for U6.P6AT."""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCHEMA = "neuro-film.u6-p6at-scorpion-cross-scanner-coherence-d0-contract.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P6AT contract")
    return value


def _range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError("range response length drift")
    return data


def _central_rows(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos = 0
    while pos + 46 <= len(data) and data[pos : pos + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<6H3I5H2I", data, pos + 4)
        method, crc, compressed, uncompressed = (
            fields[3],
            fields[6],
            fields[7],
            fields[8],
        )
        name_len, extra_len, comment_len = fields[9], fields[10], fields[11]
        local_offset = fields[15]
        name = data[pos + 46 : pos + 46 + name_len].decode("utf-8")
        rows.append(
            {
                "name": name,
                "method": method,
                "crc": crc,
                "compressed": compressed,
                "uncompressed": uncompressed,
                "local_offset": local_offset,
            }
        )
        pos += 46 + name_len + extra_len + comment_len
    if pos != len(data):
        raise ValueError("central directory parse drift")
    return rows


def load_inventory(contract: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    source = contract["source"]
    central = _range(
        source["archive_url"],
        int(source["central_directory_offset"]),
        int(source["central_directory_offset"])
        + int(source["central_directory_size"])
        - 1,
    )
    if _sha(central) != source["central_directory_sha256"]:
        raise ValueError("central directory identity drift")
    rows = _central_rows(central)
    jpeg = [row for row in rows if row["name"].lower().endswith(".jpg")]
    if len(jpeg) != int(source["expected_jpeg_members"]):
        raise ValueError("JPEG inventory drift")
    return central, rows


def _selected_pairs(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    groups = sorted(
        {row["name"].rsplit("/", 1)[0] for row in rows if row["name"].endswith(".jpg")}
    )
    salt = contract["selection"]["salt"]
    ranked = sorted(
        groups, key=lambda group: hashlib.sha256(f"{salt}|{group}".encode()).hexdigest()
    )
    selected: list[tuple[str, str]] = []
    seen_slides: set[str] = set()
    for primary in ranked:
        slide = primary.split("/")[1]
        if slide in seen_slides:
            continue
        seen_slides.add(slide)
        candidates = [
            group
            for group in groups
            if group.split("/")[1] == slide and group != primary
        ]
        wrong = min(
            candidates,
            key=lambda group: hashlib.sha256(
                f"{salt}|wrong|{primary}|{group}".encode()
            ).hexdigest(),
        )
        selected.append((primary, wrong))
        if len(selected) == int(contract["gates"]["required_rows"]):
            break
    expected = [tuple(item) for item in contract["selection"]["pairs"]]
    if selected != expected:
        raise ValueError("P6AT deterministic selection drift")
    return selected


def acquire(contract: dict[str, Any], destination: Path) -> dict[str, Any]:
    central, rows = load_inventory(contract)
    pairs = _selected_pairs(contract, rows)
    by_name = {row["name"]: row for row in rows}
    scanners = contract["source"]["scanner_files"]
    outputs: list[dict[str, Any]] = []
    for primary, wrong in pairs:
        for group in (primary, wrong):
            group_rows = [by_name[f"{group}/{scanner}"] for scanner in scanners]
            targets = [destination / row["name"] for row in group_rows]
            cached = all(target.exists() for target in targets)
            group_start = min(int(row["local_offset"]) for row in group_rows)
            group_end = max(
                int(row["local_offset"]) + 4095 + int(row["compressed"])
                for row in group_rows
            )
            group_blob = (
                b""
                if cached
                else _range(contract["source"]["archive_url"], group_start, group_end)
            )
            for scanner in scanners:
                name = f"{group}/{scanner}"
                row = by_name[name]
                target = destination / name
                if cached:
                    payload = target.read_bytes()
                    if len(payload) != int(row["uncompressed"]) or (
                        zlib.crc32(payload) & 0xFFFFFFFF
                    ) != int(row["crc"]):
                        raise ValueError("cached member payload drift")
                    outputs.append(
                        {"name": name, "sha256": _sha(payload), "bytes": len(payload)}
                    )
                    continue
                local = int(row["local_offset"]) - group_start
                header = group_blob[local : local + 4096]
                if header[:4] != b"PK\x03\x04":
                    raise ValueError("local header drift")
                (
                    _,
                    flag,
                    method,
                    _,
                    _,
                    local_crc,
                    local_compressed,
                    local_uncompressed,
                    name_len,
                    extra_len,
                ) = struct.unpack_from("<5H3I2H", header, 4)
                local_name = header[30 : 30 + name_len].decode("utf-8")
                descriptor_mode = bool(flag & 0x08)
                sizes_match = (local_crc, local_compressed, local_uncompressed) == (
                    row["crc"],
                    row["compressed"],
                    row["uncompressed"],
                )
                descriptor_placeholders = descriptor_mode and (
                    local_crc,
                    local_compressed,
                    local_uncompressed,
                ) == (0, 0, 0)
                if (
                    method != row["method"]
                    or local_name != name
                    or not (sizes_match or descriptor_placeholders)
                ):
                    raise ValueError("central/local member drift")
                data_start = local + 30 + name_len + extra_len
                packed = group_blob[data_start : data_start + int(row["compressed"])]
                payload = zlib.decompress(packed, -15) if method == 8 else packed
                if len(payload) != int(row["uncompressed"]) or (
                    zlib.crc32(payload) & 0xFFFFFFFF
                ) != int(row["crc"]):
                    raise ValueError("member payload drift")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != payload:
                    raise FileExistsError(f"nonmatching existing P6AT member: {target}")
                if not target.exists():
                    target.write_bytes(payload)
                outputs.append(
                    {"name": name, "sha256": _sha(payload), "bytes": len(payload)}
                )
    return {
        "central_directory_sha256": _sha(central),
        "selected_pairs": [list(pair) for pair in pairs],
        "outputs": outputs,
    }


def _rank_luminance(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if rgb.shape != (1024, 1024, 3):
        raise ValueError("P6AT decoded shape drift")
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    _, inverse, counts = np.unique(luminance, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    midranks = (starts + 0.5 * counts) / luminance.size
    return midranks[inverse].reshape(luminance.shape)


def _spectra(image: np.ndarray, analysis: dict[str, Any]) -> np.ndarray:
    crop = int(analysis["crop_size"])
    size = int(analysis["window_size"])
    stride = int(analysis["window_stride"])
    start = (image.shape[0] - crop) // 2
    image = image[start : start + crop, start : start + crop]
    window = np.outer(np.hanning(size), np.hanning(size))
    rows = []
    for y in range(0, crop - size + 1, stride):
        for x in range(0, crop - size + 1, stride):
            patch = image[y : y + size, x : x + size]
            rows.append(np.fft.rfft2((patch - patch.mean()) * window))
    return np.stack(rows)


def _coherence(a: np.ndarray, b: np.ndarray, analysis: dict[str, Any]) -> float:
    cross = np.mean(a * np.conj(b), axis=0)
    power_a = np.mean(np.abs(a) ** 2, axis=0)
    power_b = np.mean(np.abs(b) ** 2, axis=0)
    coherence = np.abs(cross) ** 2 / np.maximum(
        power_a * power_b, np.finfo(np.float64).tiny
    )
    fy = np.fft.fftfreq(a.shape[1])[:, None]
    fx = np.fft.rfftfreq((a.shape[2] - 1) * 2)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    mask = (radius >= float(analysis["frequency_min_cycles_per_pixel"])) & (
        radius <= float(analysis["frequency_max_cycles_per_pixel"])
    )
    return float(np.median(coherence[mask]))


def evaluate(
    contract: dict[str, Any], data_root: Path, acquisition_report: Path
) -> dict[str, Any]:
    _, inventory = load_inventory(contract)
    pairs = _selected_pairs(contract, inventory)
    acquisition_bytes = acquisition_report.read_bytes()
    acquisition = json.loads(acquisition_bytes)
    expected_names = {
        f"{group}/{scanner}"
        for pair in pairs
        for group in pair
        for scanner in contract["source"]["scanner_files"]
    }
    acquired = acquisition.get("outputs")
    if (
        not isinstance(acquired, list)
        or {row.get("name") for row in acquired} != expected_names
    ):
        raise ValueError("P6AT acquisition inventory drift")
    for row in acquired:
        payload = (data_root / row["name"]).read_bytes()
        if len(payload) != int(row["bytes"]) or _sha(payload) != row["sha256"]:
            raise ValueError("P6AT acquired member identity drift")
    scanners = contract["source"]["scanner_files"]
    analysis = contract["analysis"]
    rows = []
    for primary, wrong in pairs:
        primary_spectra = {
            scanner: _spectra(_rank_luminance(data_root / primary / scanner), analysis)
            for scanner in scanners
        }
        wrong_spectra = {
            scanner: _spectra(_rank_luminance(data_root / wrong / scanner), analysis)
            for scanner in scanners
        }
        dy, dx = map(int, analysis["shift_control_pixels"])
        shifted_spectra = {
            scanner: _spectra(
                np.roll(
                    _rank_luminance(data_root / primary / scanner),
                    shift=(dy, dx),
                    axis=(0, 1),
                ),
                analysis,
            )
            for scanner in scanners
        }
        for left_index, left in enumerate(scanners):
            for right in scanners[left_index + 1 :]:
                rows.append(
                    {
                        "primary": primary,
                        "scanner_pair": [left, right],
                        "correct": _coherence(
                            primary_spectra[left], primary_spectra[right], analysis
                        ),
                        "wrong": _coherence(
                            primary_spectra[left], wrong_spectra[right], analysis
                        ),
                        "shifted": _coherence(
                            primary_spectra[left], shifted_spectra[right], analysis
                        ),
                    }
                )
    correct = np.array([row["correct"] for row in rows])
    wrong = np.array([row["wrong"] for row in rows])
    shifted = np.array([row["shifted"] for row in rows])
    tiny = np.finfo(np.float64).tiny
    aggregate = {
        "rows": len(pairs),
        "scanner_pair_observations": len(rows),
        "correct_vs_wrong_win_rate": float(np.mean(correct > wrong)),
        "correct_vs_shifted_win_rate": float(np.mean(correct > shifted)),
        "median_correct_to_wrong_ratio": float(
            np.median(correct / np.maximum(wrong, tiny))
        ),
        "median_correct_to_shifted_ratio": float(
            np.median(correct / np.maximum(shifted, tiny))
        ),
        "median_correct_coherence": float(np.median(correct)),
    }
    gates = contract["gates"]
    expected_observations = int(gates["required_rows"]) * int(
        gates["required_scanner_pairs_per_row"]
    )
    if len(pairs) != int(gates["required_rows"]) or len(rows) != expected_observations:
        raise ValueError("P6AT evaluation population drift")
    failed = []
    comparisons = {
        "minimum_correct_vs_wrong_win_rate": aggregate["correct_vs_wrong_win_rate"],
        "minimum_correct_vs_shifted_win_rate": aggregate["correct_vs_shifted_win_rate"],
        "minimum_median_correct_to_wrong_ratio": aggregate[
            "median_correct_to_wrong_ratio"
        ],
        "minimum_median_correct_to_shifted_ratio": aggregate[
            "median_correct_to_shifted_ratio"
        ],
        "minimum_median_correct_coherence": aggregate["median_correct_coherence"],
    }
    for key, value in comparisons.items():
        if not np.isfinite(value) or value < float(gates[key]):
            failed.append(key)
    report = {
        "schema": "neuro-film.u6-p6at-scorpion-cross-scanner-coherence-d0-report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha(_canonical(contract)),
        "acquisition_report_sha256": _sha(acquisition_bytes),
        "rows": rows,
        "aggregate": aggregate,
        "failed_gates": failed,
        "automatic_pass": not failed,
        "decision": contract["decision_if_pass"]
        if not failed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = _sha(_canonical(report))
    return report
