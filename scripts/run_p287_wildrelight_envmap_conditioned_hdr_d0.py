"""Acquire and audit the bounded P287 WildRelight lake observation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class P287Error(RuntimeError):
    """Raised when a frozen P287 boundary differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_binding(bindings: dict[str, Any], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    body = path.read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256_bytes(
        body
    ) == str(bindings[f"{prefix}_sha256"])


def _data_root(config: dict[str, Any]) -> Path:
    relative = Path(str(config["source"]["local_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise P287Error("source root must remain repository-relative")
    return ROOT / relative


def _exact_source_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = _data_root(config)
    result: list[dict[str, Any]] = []
    for member in config["source"]["members"]:
        path = root / str(member["relative_path"])
        exists = path.is_file()
        size = path.stat().st_size if exists else -1
        digest = _sha256_file(path) if exists and size == int(member["bytes"]) else ""
        result.append(
            {
                "bytes": size,
                "exists": exists,
                "relative_path": str(member["relative_path"]),
                "sha256": digest,
                "valid": bool(
                    exists
                    and size == int(member["bytes"])
                    and digest == str(member["sha256"])
                ),
            }
        )
    return result


def _acquire_member(
    destination: Path, *, url: str, expected_bytes: int, expected_sha256: str
) -> int:
    from src.film_physics.create_only_file import publish_create_only

    if destination.exists():
        if (
            destination.is_file()
            and destination.stat().st_size == expected_bytes
            and _sha256_file(destination) == expected_sha256
        ):
            return 0
        raise P287Error("existing source member differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{os.getpid()}.stage"
    if stage.exists():
        raise P287Error("source stage already exists")
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p287-source-acquisition/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, stage.open(
            "xb"
        ) as stream:
            if int(response.status) != 200:
                raise P287Error("source request did not return HTTP 200")
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > expected_bytes:
                    raise P287Error("source member exceeds frozen size")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise P287Error("downloaded source member differs")
        publish_create_only(stage, destination)
    except (OSError, P287Error, urllib.error.URLError):
        stage.unlink(missing_ok=True)
        raise
    return count


def acquire(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = _data_root(config)
    revision = str(config["source"]["revision"])
    dataset = str(config["source"]["dataset_id"])
    network_bytes = 0
    for member in config["source"]["members"]:
        remote = str(member["remote_path"])
        url = (
            f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
            f"{remote}?download=true"
        )
        network_bytes += _acquire_member(
            root / str(member["relative_path"]),
            url=url,
            expected_bytes=int(member["bytes"]),
            expected_sha256=str(member["sha256"]),
        )
    files = _exact_source_files(config)
    report = {
        "experiment_id": "P287",
        "file_count": len(files),
        "files": files,
        "network_bytes": network_bytes,
        "pixel_decodes": 0,
        "schema": "neuro-film.p287-wildrelight-source-acquisition.v1",
        "source_exact": all(item["valid"] for item in files),
    }
    report["source_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def _decode_rgb(path: Path, openexr: Any) -> tuple[np.ndarray, dict[str, Any]]:
    with openexr.File(str(path)) as exr_file:
        header = dict(exr_file.header())
        channels = exr_file.channels()
        if "RGB" in channels:
            pixels = channels["RGB"].pixels.copy()
        elif all(name in channels for name in ("R", "G", "B")):
            pixels = np.stack(
                [channels[name].pixels.copy() for name in ("R", "G", "B")], axis=-1
            )
        else:
            raise P287Error("EXR does not expose exact RGB channels")
        part_count = len(exr_file.parts)
    values = np.ascontiguousarray(pixels, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise P287Error("EXR RGB payload is malformed or nonfinite")
    return values, {
        "channel_names": sorted(channels),
        "dtype": str(values.dtype),
        "part_count": part_count,
        "shape": list(values.shape),
        "type": str(header.get("type")),
    }


def _spherical_rgb_mean(envmap: np.ndarray) -> np.ndarray:
    if np.min(envmap) < 0.0:
        raise P287Error("environment map contains negative radiance")
    height = envmap.shape[0]
    theta = np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    weights = np.sin(theta)[:, None, None]
    mean = np.sum(envmap.astype(np.float64) * weights, axis=(0, 1)) / (
        float(envmap.shape[1]) * float(np.sum(weights))
    )
    if not np.isfinite(mean).all() or np.min(mean) <= 0.0:
        raise P287Error("environment mean is not positive finite")
    return mean


def _metric(candidate: np.ndarray, target: np.ndarray, source: np.ndarray) -> dict[str, float]:
    if candidate.shape != target.shape or source.shape != target.shape:
        raise P287Error("photo shapes differ")
    height, width, _ = target.shape
    y = int(np.floor(height * 0.05))
    x = int(np.floor(width * 0.05))
    slices = (slice(y, height - y), slice(x, width - x), slice(None))
    candidate = candidate[slices].astype(np.float64)
    target = target[slices].astype(np.float64)
    source = source[slices].astype(np.float64)
    epsilon = float(2.0**-16)
    valid = (
        np.isfinite(candidate)
        & np.isfinite(target)
        & np.isfinite(source)
        & (target > epsilon)
        & (source > epsilon)
    )
    if not np.any(valid):
        raise P287Error("photo row has no valid score support")
    delta = np.log2(candidate[valid] + epsilon) - np.log2(target[valid] + epsilon)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "valid_fraction": float(np.mean(valid)),
    }


def _capture_separations(metadata: dict[str, Any]) -> list[float]:
    photos = {item["time"]: item for item in metadata["photos"]}
    envmaps = {item["time"]: item for item in metadata["envmaps"]}
    if sorted(photos) != sorted(envmaps):
        raise P287Error("photo and envmap time roles differ")
    fmt = "%Y:%m:%d %H:%M:%S"
    return [
        abs(
            (
                datetime.strptime(photos[key]["shooting_time"], fmt).replace(
                    tzinfo=timezone.utc
                )
                - datetime.strptime(envmaps[key]["shooting_time"], fmt).replace(
                    tzinfo=timezone.utc
                )
            ).total_seconds()
        )
        for key in sorted(photos)
    ]


def _worker_execute(config_path: Path, site: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    openexr = importlib.import_module("OpenEXR")
    root = _data_root(config)
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    separations = _capture_separations(metadata)
    envmaps: dict[int, np.ndarray] = {}
    env_means: dict[int, np.ndarray] = {}
    env_headers: list[dict[str, Any]] = []
    for index in range(6):
        value, header = _decode_rgb(root / f"envmap/time{index}_envmap.exr", openexr)
        envmaps[index] = value
        env_means[index] = _spherical_rgb_mean(value)
        env_headers.append(header)
    directed = [(index, index + 1) for index in range(5)] + [
        (index + 1, index) for index in range(5)
    ]
    if reverse:
        directed.reverse()
    rows: list[dict[str, Any]] = []
    photo_headers: dict[str, dict[str, Any]] = {}
    luma = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
    all_finite = True
    all_sign_exact = True
    for source_index, target_index in directed:
        source, source_header = _decode_rgb(
            root / f"photo/time{source_index}_hdr.exr", openexr
        )
        photo_headers[f"time{source_index}"] = source_header
        gain = env_means[target_index] / env_means[source_index]
        scalar_gain = float(
            np.dot(env_means[target_index], luma) / np.dot(env_means[source_index], luma)
        )
        wrong_index = (target_index + 1) % 6
        wrong_gain = env_means[wrong_index] / env_means[source_index]
        candidate = np.ascontiguousarray(source.astype(np.float64) * gain, dtype=np.float32)
        scalar = np.ascontiguousarray(source.astype(np.float64) * scalar_gain, dtype=np.float32)
        cyclic = np.ascontiguousarray(source.astype(np.float64) * wrong_gain, dtype=np.float32)
        frozen_hashes = {
            "candidate": _array_sha256(candidate),
            "cyclic": _array_sha256(cyclic),
            "identity": _array_sha256(source),
            "scalar": _array_sha256(scalar),
        }
        target_reads_before_freeze = 0
        target, target_header = _decode_rgb(
            root / f"photo/time{target_index}_hdr.exr", openexr
        )
        photo_headers[f"time{target_index}"] = target_header
        candidate_metric = _metric(candidate, target, source)
        cyclic_metric = _metric(cyclic, target, source)
        identity_metric = _metric(source, target, source)
        scalar_metric = _metric(scalar, target, source)
        control_rmse = min(identity_metric["rmse"], scalar_metric["rmse"])
        candidate_reduction = 100.0 * (control_rmse - candidate_metric["rmse"]) / control_rmse
        cyclic_reduction = (
            100.0
            * (cyclic_metric["rmse"] - candidate_metric["rmse"])
            / cyclic_metric["rmse"]
        )
        finite = bool(np.isfinite(candidate).all())
        sign_exact = bool(np.array_equal(np.signbit(candidate), np.signbit(source)))
        all_finite = all_finite and finite
        all_sign_exact = all_sign_exact and sign_exact
        rows.append(
            {
                "candidate_reduction_vs_strongest_control_percent": candidate_reduction,
                "candidate_rmse": candidate_metric["rmse"],
                "cyclic_reduction_percent": cyclic_reduction,
                "cyclic_rmse": cyclic_metric["rmse"],
                "finite": finite,
                "frozen_output_hashes": frozen_hashes,
                "gain_rgb": [float(value) for value in gain],
                "identity_rmse": identity_metric["rmse"],
                "row": f"time{source_index}->time{target_index}",
                "scalar_gain": scalar_gain,
                "scalar_rmse": scalar_metric["rmse"],
                "sign_exact": sign_exact,
                "target_reads_before_freeze": target_reads_before_freeze,
                "valid_fraction": candidate_metric["valid_fraction"],
                "wrong_target_envmap": f"time{wrong_index}",
            }
        )
    rows.sort(key=lambda item: item["row"])
    candidate_reductions = np.asarray(
        [item["candidate_reduction_vs_strongest_control_percent"] for item in rows],
        dtype=np.float64,
    )
    cyclic_reductions = np.asarray(
        [item["cyclic_reduction_percent"] for item in rows], dtype=np.float64
    )
    expected = config["expected"]
    gates = {
        "candidate_control_rate": bool(
            np.mean(candidate_reductions > 0.0)
            >= float(expected["minimum_candidate_control_rate"])
        ),
        "candidate_median_reduction": bool(
            np.median(candidate_reductions)
            >= float(expected["minimum_candidate_median_reduction_percent"])
        ),
        "candidate_worst_reduction": bool(
            np.min(candidate_reductions)
            >= float(expected["minimum_candidate_worst_reduction_percent"])
        ),
        "capture_time_separation": max(separations)
        <= float(expected["maximum_capture_time_separation_seconds"]),
        "cyclic_control_rate": bool(
            np.mean(cyclic_reductions > 0.0)
            >= float(expected["minimum_cyclic_control_rate"])
        ),
        "cyclic_median_reduction": bool(
            np.median(cyclic_reductions)
            >= float(expected["minimum_cyclic_median_reduction_percent"])
        ),
        "finite_and_sign_exact": all_finite and all_sign_exact,
        "target_unread_before_freeze": all(
            item["target_reads_before_freeze"] == 0 for item in rows
        ),
        "valid_support": min(item["valid_fraction"] for item in rows)
        >= float(expected["minimum_valid_fraction"]),
    }
    aggregate = {
        "candidate_control_improve_rate": float(np.mean(candidate_reductions > 0.0)),
        "candidate_median_reduction_percent": float(np.median(candidate_reductions)),
        "candidate_worst_reduction_percent": float(np.min(candidate_reductions)),
        "cyclic_improve_rate": float(np.mean(cyclic_reductions > 0.0)),
        "cyclic_median_reduction_percent": float(np.median(cyclic_reductions)),
        "maximum_capture_time_separation_seconds": float(max(separations)),
        "minimum_valid_fraction": float(min(item["valid_fraction"] for item in rows)),
    }
    return {
        "aggregate": aggregate,
        "decision": (
            "PASS_PRIVATE_WILDRELIGHT_ENVMAP_CONDITIONED_HDR_D0"
            if all(gates.values())
            else "FAIL_CLOSED_WILDRELIGHT_ENVMAP_CONDITIONED_HDR_D0"
        ),
        "envmap_headers": env_headers,
        "envmap_mean_rgb": {
            f"time{index}": [float(value) for value in env_means[index]]
            for index in range(6)
        },
        "gates": gates,
        "openexr_version": openexr.__version__,
        "photo_headers": photo_headers,
        "rows": rows,
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    binding_gates = {
        "contract": _verify_binding(bindings, "contract"),
        "runner": _verify_binding(bindings, "runner"),
        "test": _verify_binding(bindings, "test"),
    }
    source_files = _exact_source_files(config)
    wheel = ROOT / str(bindings["openexr_wheel_path"])
    wheel_exact = bool(
        wheel.is_file()
        and wheel.stat().st_size == int(bindings["openexr_wheel_bytes"])
        and _sha256_file(wheel) == str(bindings["openexr_wheel_sha256"])
    )
    if not all(binding_gates.values()) or not all(
        item["valid"] for item in source_files
    ) or not wheel_exact:
        raise P287Error("formal source or local binding differs")
    temporary = Path(tempfile.mkdtemp(prefix="p287-openexr-"))
    worker_report = temporary / "worker.json"
    site = temporary / "site"
    cleanup = False
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
            capture_output=True,
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(config_path),
            "--site",
            str(site),
            "--output",
            str(worker_report),
        ]
        if reverse:
            command.append("--reverse")
        subprocess.run(command, check=True, capture_output=True)
        worker = json.loads(worker_report.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(temporary)
        cleanup = not temporary.exists()
    report = {
        **worker,
        "bindings": {**binding_gates, "openexr_wheel": wheel_exact},
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": "P287",
        "network_bytes": 0,
        "schema": "neuro-film.p287-wildrelight-envmap-conditioned-hdr-d0-result.v1",
        "source_file_count": len(source_files),
        "source_files_exact": all(item["valid"] for item in source_files),
        "temporary_cleanup": cleanup,
    }
    report["gates"]["bindings_source_runtime_cleanup"] = bool(
        all(binding_gates.values())
        and wheel_exact
        and report["source_files_exact"]
        and cleanup
    )
    report["decision"] = (
        "PASS_PRIVATE_WILDRELIGHT_ENVMAP_CONDITIONED_HDR_D0"
        if all(report["gates"].values())
        else "FAIL_CLOSED_WILDRELIGHT_ENVMAP_CONDITIONED_HDR_D0"
    )
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.acquire:
        report = acquire(config_path)
    elif args.worker:
        if args.site is None:
            raise P287Error("worker site is required")
        report = _worker_execute(config_path, args.site.resolve(), reverse=args.reverse)
    else:
        report = execute(config_path, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
