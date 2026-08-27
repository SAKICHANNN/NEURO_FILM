from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class P269Error(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _safe_relative(role: str, relative_path: str) -> Path:
    if not SAFE_PART.fullmatch(role):
        raise P269Error("unsafe role")
    candidate = Path(relative_path)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or not SAFE_PART.fullmatch(candidate.name)
    ):
        raise P269Error("unsafe object path")
    return Path(role) / candidate.name


def _gcloud_rows(url: str) -> list[dict[str, Any]]:
    executable = shutil_which("gcloud.cmd")
    if executable is None:
        raise P269Error("installed gcloud.cmd is unavailable")
    completed = subprocess.run(
        [executable, "storage", "ls", "--json", f"{url}**"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    if completed.returncode != 0:
        raise P269Error(f"gcloud metadata listing failed: {completed.stderr.strip()}")
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise P269Error("gcloud metadata listing is not a list")
    return rows


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _remote_metadata(config: dict[str, Any], *, reverse: bool) -> list[dict[str, Any]]:
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    roles = list(config["object_roots"].items())
    if reverse:
        roles.reverse()
    for role, root in roles:
        rows = _gcloud_rows(root)
        if reverse:
            rows.reverse()
        prefix = root.removeprefix("gs://hdrplusdata/")
        for row in rows:
            if row.get("type") != "cloud_object":
                continue
            metadata = row.get("metadata", {})
            name = str(metadata.get("name", ""))
            if not name.startswith(prefix):
                raise P269Error("remote object escaped frozen prefix")
            relative = name[len(prefix) :]
            _safe_relative(role, relative)
            key = (role, relative)
            if key in actual:
                raise P269Error("duplicate remote object")
            actual[key] = {
                "role": role,
                "path": relative,
                "bytes": int(metadata["size"]),
                "generation": str(metadata["generation"]),
                "md5_base64": str(metadata["md5Hash"]),
            }
    expected = {(item["role"], item["path"]): item for item in config["objects"]}
    if set(actual) != set(expected):
        raise P269Error("remote object inventory differs from frozen manifest")
    normalized: list[dict[str, Any]] = []
    for key in sorted(expected):
        frozen = expected[key]
        observed = actual[key]
        if any(
            observed[field] != frozen[field]
            for field in ("bytes", "generation", "md5_base64")
        ):
            raise P269Error("remote object identity differs from frozen manifest")
        normalized.append(observed)
    return normalized


def _source_url(root: str, relative_path: str, generation: str) -> str:
    object_name = root.removeprefix("gs://hdrplusdata/") + relative_path
    encoded = urllib.parse.quote(object_name, safe="/")
    return (
        f"https://storage.googleapis.com/hdrplusdata/{encoded}?generation={generation}"
    )


def _verify_local(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise P269Error("expected local object is missing")
    size = path.stat().st_size
    md5 = _md5_base64(path)
    if size != item["bytes"] or md5 != item["md5_base64"]:
        raise P269Error("local object identity mismatch")
    return {
        "role": item["role"],
        "path": item["path"],
        "bytes": size,
        "md5_base64": md5,
        "sha256": _sha256_file(path),
    }


def _acquire_one(
    destination_root: Path, roots: dict[str, str], item: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    relative = _safe_relative(item["role"], item["path"])
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _verify_local(destination, item), False
    stage = destination.parent / f".{destination.name}.p269-stage-{uuid.uuid4().hex}"
    source = _source_url(roots[item["role"]], item["path"], item["generation"])
    try:
        request = urllib.request.Request(
            source, headers={"User-Agent": "neuro-film-p269/1"}
        )
        with (
            urllib.request.urlopen(request, timeout=180) as response,
            stage.open("xb") as stream,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        verified = _verify_local(stage, item)
        os.rename(stage, destination)
        return verified, True
    finally:
        if stage.exists():
            stage.unlink()


def _acquire_all(
    config: dict[str, Any], destination_root: Path, *, reverse: bool
) -> tuple[list[dict[str, Any]], int]:
    items = list(config["objects"])
    if reverse:
        items.reverse()
    results: list[dict[str, Any]] = []
    acquired = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                _acquire_one, destination_root, config["object_roots"], item
            ): item
            for item in items
        }
        for future in as_completed(futures):
            verified, created = future.result()
            results.append(verified)
            acquired += int(created)
    return sorted(results, key=lambda row: (row["role"], row["path"])), acquired


def _signature(path: Path) -> str:
    header = path.read_bytes()[:16]
    suffix = path.suffix.lower()
    if suffix in {".dng", ".tiff"}:
        if not header.startswith((b"II*\x00", b"MM\x00*")):
            raise P269Error("invalid TIFF/DNG signature")
        return "tiff"
    if suffix == ".jpg":
        if not header.startswith(b"\xff\xd8\xff"):
            raise P269Error("invalid JPEG signature")
        return "jpeg"
    if suffix == ".txt":
        path.read_text(encoding="utf-8")
        return "utf8-text"
    raise P269Error("unsupported frozen object suffix")


def _tag_value(tags: Any, name: str) -> Any:
    tag = tags.get(name)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dng_metadata(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        pages = list(tif.pages)
        selected = max(pages, key=lambda page: int(np.prod(page.shape)))
        tags = selected.tags
        return {
            "page_count": len(pages),
            "shape": list(selected.shape),
            "dtype": str(selected.dtype),
            "compression": str(selected.compression.name),
            "dng_version": _tag_value(tags, "DNGVersion"),
            "make": _tag_value(tags, "Make"),
            "model": _tag_value(tags, "Model"),
            "cfa_repeat_pattern_dim": _tag_value(tags, "CFARepeatPatternDim"),
            "cfa_pattern": _tag_value(tags, "CFAPattern"),
            "white_level": _tag_value(tags, "WhiteLevel"),
            "black_level": _tag_value(tags, "BlackLevel"),
        }


def _input_dng_probe(path: Path, side: int) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        page = max(tif.pages, key=lambda candidate: int(np.prod(candidate.shape)))
        if (
            page.compression.name != "NONE"
            or page.dtype != np.dtype("uint16")
            or page.samplesperpixel != 1
            or len(page.shape) != 2
        ):
            raise P269Error("input DNG is not bounded-readable uncompressed uint16")
        height, width = (int(value) for value in page.shape)
        rows_per_strip_tag = page.tags.get("RowsPerStrip")
        rows_per_strip = int(rows_per_strip_tag.value) if rows_per_strip_tag else height
        if rows_per_strip != 1 or len(page.dataoffsets) != height:
            raise P269Error(
                "input DNG does not expose one independently readable strip per row"
            )
        if side <= 0 or side > height or side > width:
            raise P269Error("bounded input DNG probe side is invalid")
        y0 = max(0, (height - side) // 2)
        x0 = max(0, (width - side) // 2)
        rows: list[np.ndarray] = []
        with path.open("rb") as stream:
            for y in range(y0, y0 + side):
                if int(page.databytecounts[y]) < width * 2:
                    raise P269Error("input DNG row strip is shorter than expected")
                stream.seek(int(page.dataoffsets[y]) + x0 * 2)
                payload = stream.read(side * 2)
                if len(payload) != side * 2:
                    raise P269Error("bounded input DNG probe read is truncated")
                rows.append(np.frombuffer(payload, dtype=f"{tif.byteorder}u2"))
        probe = np.ascontiguousarray(np.stack(rows).astype(np.uint16, copy=False))
        return {
            "raw_height": int(height),
            "raw_width": int(width),
            "raw_dtype": "uint16",
            "cfa_pattern": _tag_value(page.tags, "CFAPattern"),
            "white_level": _tag_value(page.tags, "WhiteLevel"),
            "probe_shape": list(probe.shape),
            "probe_minimum": int(probe.min()),
            "probe_maximum": int(probe.max()),
            "probe_u16le_sha256": _sha256_bytes(probe.astype("<u2").tobytes()),
            "file_payload_bytes_read": int(probe.size * 2),
            "full_raw_plane_decoded": False,
        }


def _inspect(
    config: dict[str, Any], destination_root: Path, local: list[dict[str, Any]]
) -> dict[str, Any]:
    signatures: list[dict[str, str]] = []
    input_dngs: list[dict[str, Any]] = []
    result_dngs: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    side = int(config["budgets"]["maximum_dng_centre_probe_side"])
    for item in local:
        path = destination_root / _safe_relative(item["role"], item["path"])
        signatures.append(
            {"role": item["role"], "path": item["path"], "kind": _signature(path)}
        )
        if item["path"].endswith(".dng"):
            metadata = _dng_metadata(path)
            row = {"role": item["role"], "path": item["path"], "metadata": metadata}
            if item["role"] == "burst":
                row["centre_probe"] = _input_dng_probe(path, side)
                input_dngs.append(row)
            else:
                result_dngs.append(row)
        elif item["path"].endswith(".txt"):
            text = path.read_text(encoding="utf-8").strip()
            sidecars.append(
                {
                    "role": item["role"],
                    "path": item["path"],
                    "utf8_sha256": _sha256_bytes(text.encode("utf-8")),
                    "nonempty": bool(text),
                }
            )
    return {
        "signatures": sorted(signatures, key=lambda row: (row["role"], row["path"])),
        "input_dngs": sorted(input_dngs, key=lambda row: row["path"]),
        "result_dng_metadata_only": sorted(
            result_dngs, key=lambda row: (row["role"], row["path"])
        ),
        "text_sidecars": sorted(sidecars, key=lambda row: (row["role"], row["path"])),
    }


def build_report(config_path: Path, *, acquire: bool, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination_root = ROOT / config["destination_root"]
    expected_total = sum(int(item["bytes"]) for item in config["objects"])
    if expected_total != config["expected_total_bytes"]:
        raise P269Error("frozen total differs from object manifest")
    if expected_total > config["maximum_total_bytes"]:
        raise P269Error("frozen object manifest exceeds budget")
    remote = _remote_metadata(config, reverse=reverse)
    if acquire:
        local, acquired_count = _acquire_all(config, destination_root, reverse=reverse)
    else:
        local = []
        for item in config["objects"]:
            path = destination_root / _safe_relative(item["role"], item["path"])
            local.append(_verify_local(path, item))
        local.sort(key=lambda row: (row["role"], row["path"]))
        acquired_count = 0
    inspection = _inspect(config, destination_root, local)
    scientific = {
        "selected_burst_id": config["selected_burst_id"],
        "remote_inventory": remote,
        "local_inventory": local,
        "local_inventory_sha256": _sha256_bytes(_canonical_bytes(local)),
        "expected_total_bytes": expected_total,
        "inspection": inspection,
        "input_dng_decode_count": len(inspection["input_dngs"]),
        "result_pixel_decode_count": 0,
        "fit_train_inference_score_reads": 0,
    }
    gates = {
        "remote_identity": len(remote) == len(config["objects"]),
        "local_identity": sum(row["bytes"] for row in local) == expected_total,
        "signatures": len(inspection["signatures"]) == len(config["objects"]),
        "input_dng_probes": len(inspection["input_dngs"]) == 8,
        "result_metadata_only": len(inspection["result_dng_metadata_only"]) == 2,
        "sidecars": all(row["nonempty"] for row in inspection["text_sidecars"]),
        "zero_result_and_science_reads": scientific["result_pixel_decode_count"] == 0
        and scientific["fit_train_inference_score_reads"] == 0,
    }
    scientific["gates"] = gates
    status = (
        "PASS_PRIVATE_HDRPLUS_ONE_BURST_SOURCE_LOCK"
        if all(gates.values())
        else "FAIL_CLOSED_HDRPLUS_ONE_BURST_SOURCE_LOCK"
    )
    return {
        "schema": "neuro-film.p269-hdrplus-one-burst-acquisition-result.v1",
        "experiment_id": "P269",
        "status": status,
        "scientific_identity": _sha256_bytes(_canonical_bytes(scientific)),
        "scientific": scientific,
        "execution": {
            "acquire_requested": acquire,
            "reverse_enumeration": reverse,
            "objects_created_this_run": acquired_count,
        },
        "identities": {
            "config_sha256": _sha256_file(config_path),
            "contract_sha256": _sha256_file(
                ROOT / "docs/planning/P269_HDRPLUS_ONE_BURST_ACQUISITION_CONTRACT.md"
            ),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = build_report(args.config, acquire=args.acquire, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(report)
    output.write_bytes(payload)
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    sys.exit(main())
