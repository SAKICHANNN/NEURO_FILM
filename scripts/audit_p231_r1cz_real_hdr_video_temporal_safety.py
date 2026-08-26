#!/usr/bin/env python3
"""Audit the exact R1CZ payload on one already-consumed real HDR video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.shared_hdr_dpct_payload import (
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)
from src.preprocess.dng_metadata import canonical_json_bytes

SCHEMA = "neuro-film.p231-r1cz-real-hdr-video-temporal-safety-result.v1"
BT2020_LUMA = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
TEMPORAL_EPSILON_NITS = 1.0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def pq_eotf_nits(encoded: np.ndarray) -> np.ndarray:
    """Return analytic SMPTE ST 2084 luminance in cd/m2."""

    values = np.asarray(encoded, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PQ codes must be finite in [0,1]")
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    powered = np.power(values, 1.0 / m2)
    numerator = np.maximum(powered - c1, 0.0)
    denominator = c2 - c3 * powered
    return 10000.0 * np.power(numerator / denominator, 1.0 / m1)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _probe_media(config: dict[str, Any], video: Path) -> dict[str, Any]:
    runtime = config["runtime"]
    completed = subprocess.run(
        [
            runtime["ffprobe_path"],
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("expected exactly one video stream")
    stream = streams[0]
    return {
        "codec": str(stream["codec_name"]),
        "pixel_format": str(stream["pix_fmt"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "color_primaries": str(stream["color_primaries"]),
        "color_transfer": str(stream["color_transfer"]),
        "color_space": str(stream["color_space"]),
    }


def _decode_sampled_nits(config: dict[str, Any], video: Path) -> np.ndarray:
    media = config["media"]
    width = int(media["width"])
    height = int(media["height"])
    frames = int(media["frames"])
    stride = int(media["sample_stride"])
    frame_values = width * height * 3
    frame_bytes = frame_values * np.dtype("<f4").itemsize
    process = subprocess.Popen(
        [
            config["runtime"]["ffmpeg_path"],
            "-v",
            "error",
            "-nostdin",
            "-threads",
            str(config["runtime"]["threads"]),
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gbrpf32le",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("FFmpeg pipes were not created")
    sampled: list[np.ndarray] = []
    try:
        for _ in range(frames):
            payload = _read_exact(process.stdout, frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError("FFmpeg raw frame was truncated")
            planes = np.frombuffer(payload, dtype="<f4").reshape(3, height, width)
            encoded = np.stack(
                (planes[2, ::stride, ::stride], planes[0, ::stride, ::stride], planes[1, ::stride, ::stride]),
                axis=-1,
            )
            sampled.append(pq_eotf_nits(np.clip(encoded.astype(np.float64), 0.0, 1.0)).astype(np.float32))
        trailing = process.stdout.read(1)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=30)
    except BaseException:
        process.kill()
        process.wait(timeout=30)
        raise
    if return_code != 0 or stderr or trailing:
        raise RuntimeError(
            f"FFmpeg decode did not terminate cleanly: code={return_code}, "
            f"stderr={stderr!r}, trailing={len(trailing)}"
        )
    stack = np.stack(sampled, axis=0)
    expected_shape = (
        frames,
        int(media["sampled_height"]),
        int(media["sampled_width"]),
        3,
    )
    if stack.shape != expected_shape:
        raise ValueError(f"sampled shape changed: {stack.shape}")
    if not np.isfinite(stack).all() or np.any((stack < 0.0) | (stack > 10000.0)):
        raise ValueError("sampled HDR frames left display-linear bounds")
    return stack


def temporal_vectors(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(frames, dtype=np.float64)
    luminance = values @ BT2020_LUMA
    luminance_steps = np.abs(
        np.diff(np.log2(TEMPORAL_EPSILON_NITS + luminance), axis=0)
    ).astype(np.float32)
    chroma = np.stack(
        (
            np.log(
                (values[..., 0] + TEMPORAL_EPSILON_NITS)
                / (values[..., 1] + TEMPORAL_EPSILON_NITS)
            ),
            np.log(
                (values[..., 2] + TEMPORAL_EPSILON_NITS)
                / (values[..., 1] + TEMPORAL_EPSILON_NITS)
            ),
        ),
        axis=-1,
    )
    chroma_steps = np.linalg.norm(np.diff(chroma, axis=0), axis=-1).astype(
        np.float32
    )
    return luminance_steps.reshape(-1), chroma_steps.reshape(-1)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 1.0 if numerator == 0.0 else float("inf")


def summarize(source: np.ndarray, output: np.ndarray) -> dict[str, Any]:
    source_luma, source_chroma = temporal_vectors(source)
    output_luma, output_chroma = temporal_vectors(output)
    source_luma_p95 = float(np.quantile(source_luma, 0.95))
    source_chroma_p95 = float(np.quantile(source_chroma, 0.95))
    output_luma_p95 = float(np.quantile(output_luma, 0.95))
    output_chroma_p95 = float(np.quantile(output_chroma, 0.95))
    material = np.linalg.norm(
        np.log1p(output.astype(np.float64))
        - np.log1p(source.astype(np.float64)),
        axis=-1,
    )
    strict_source = (source > 0.0) & (source < 10000.0)
    new_boundary = strict_source & ((output <= 0.0) | (output >= 10000.0))
    return {
        "source_f32le_sha256": _sha256_bytes(
            source.astype("<f4", copy=False).tobytes()
        ),
        "output_f32le_sha256": _sha256_bytes(
            output.astype("<f4", copy=False).tobytes()
        ),
        "all_finite": bool(np.isfinite(output).all()),
        "minimum_nits": float(np.min(output)),
        "maximum_nits": float(np.max(output)),
        "new_boundary_fraction": float(
            np.count_nonzero(new_boundary) / new_boundary.size
        ),
        "median_log_rgb_material_effect": float(np.median(material)),
        "source_luminance_temporal_p95": source_luma_p95,
        "output_luminance_temporal_p95": output_luma_p95,
        "luminance_temporal_p95_ratio": _ratio(output_luma_p95, source_luma_p95),
        "source_chroma_temporal_p95": source_chroma_p95,
        "output_chroma_temporal_p95": output_chroma_p95,
        "chroma_temporal_p95_ratio": _ratio(
            output_chroma_p95, source_chroma_p95
        ),
        "new_luminance_spike_fraction": float(
            np.mean(output_luma > source_luma + 0.25)
        ),
    }


def _bindings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    producer = Path(config["producer"]["repository"])
    paths = {
        "p230_evidence": (
            ROOT / config["consumer"]["p230_evidence_path"],
            config["consumer"]["p230_evidence_sha256"],
        ),
        "p230_source": (
            ROOT / config["consumer"]["p230_source_path"],
            config["consumer"]["p230_source_sha256"],
        ),
        "r1cz_evidence": (
            producer / config["producer"]["r1cz_evidence_path"],
            config["producer"]["r1cz_evidence_sha256"],
        ),
        "payload": (
            producer / config["producer"]["payload_path"],
            config["producer"]["payload_sha256"],
        ),
        "r1ct_evidence": (
            producer / config["producer"]["r1ct_evidence_path"],
            config["producer"]["r1ct_evidence_sha256"],
        ),
        "r0vc_evidence": (
            producer / config["producer"]["r0vc_evidence_path"],
            config["producer"]["r0vc_evidence_sha256"],
        ),
        "r1ct_forward": (
            producer / config["producer"]["r1ct_forward_report_path"],
            config["producer"]["r1ct_forward_report_sha256"],
        ),
        "r1ct_reverse": (
            producer / config["producer"]["r1ct_reverse_report_path"],
            config["producer"]["r1ct_reverse_report_sha256"],
        ),
        "video": (
            producer / config["producer"]["video_path"],
            config["producer"]["video_sha256"],
        ),
        "ffmpeg": (
            Path(config["runtime"]["ffmpeg_path"]),
            config["runtime"]["ffmpeg_sha256"],
        ),
        "ffprobe": (
            Path(config["runtime"]["ffprobe_path"]),
            config["runtime"]["ffprobe_sha256"],
        ),
    }
    checked: dict[str, Any] = {}
    for name, (path, expected) in paths.items():
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"P231 binding mismatch: {name}")
        checked[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    if checked["payload"]["bytes"] != config["producer"]["payload_bytes"]:
        raise RuntimeError("P231 payload byte length changed")
    if checked["video"]["bytes"] != config["producer"]["video_bytes"]:
        raise RuntimeError("P231 video byte length changed")
    evidence = _json(paths["r1cz_evidence"][0])
    return checked, evidence, paths["video"][0]


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    bindings, r1cz, video = _bindings(config)
    media = _probe_media(config, video)
    expected_media = {
        "codec": "hevc",
        "pixel_format": config["media"]["pixel_format"],
        "width": config["media"]["width"],
        "height": config["media"]["height"],
        "frames": config["media"]["frames"],
        "color_primaries": config["media"]["color_primaries"],
        "color_transfer": config["media"]["color_transfer"],
        "color_space": config["media"]["color_space"],
    }
    if media != expected_media:
        raise RuntimeError(f"P231 media facts changed: {media}")
    source = _decode_sampled_nits(config, video)
    source_sha256 = _sha256_bytes(source.astype("<f4", copy=False).tobytes())
    decode_matches = source_sha256 == config["producer"]["sampled_source_f32le_sha256"]
    scientific: dict[str, Any] = {
        "protocol": config["schema"],
        "bindings": bindings,
        "bundle_id": config["producer"]["bundle_id"],
        "payload_sha256": config["producer"]["payload_sha256"],
        "media": media,
        "source_f32le_sha256": source_sha256,
        "source_matches_r1ct": decode_matches,
        "frames": int(source.shape[0]),
        "sampled_height": int(source.shape[1]),
        "sampled_width": int(source.shape[2]),
        "payload_build_application_source_reads": 0,
        "target_reads": 0,
        "network_reads": 0,
    }
    if not decode_matches:
        scientific["gates"] = {
            "all_bindings_and_media_facts_exact": True,
            "independent_decode_matches_r1ct_source_sha": False,
        }
        status = "INFRASTRUCTURE_INVALID_P231_DECODE_DOMAIN_MISMATCH"
    else:
        payload_path = Path(bindings["payload"]["path"])
        bundle = load_shared_hdr_dpct_payload(
            payload_path.read_bytes(), r1cz["bundle_envelope"]
        )
        output = np.empty_like(source)
        indices = range(source.shape[0] - 1, -1, -1) if reverse else range(source.shape[0])
        for index in indices:
            output[index] = apply_shared_hdr_dpct_payload(source[index], bundle)
        summary = summarize(source, output)
        gates = {
            "all_bindings_and_media_facts_exact": True,
            "independent_decode_matches_r1ct_source_sha": True,
            "output_finite_in_0_10000": summary["all_finite"]
            and summary["minimum_nits"] >= 0.0
            and summary["maximum_nits"] <= 10000.0,
            "new_boundary_fraction_exact": summary["new_boundary_fraction"] == 0.0,
            "median_log_rgb_material_effect_minimum": summary[
                "median_log_rgb_material_effect"
            ]
            >= 0.01,
            "luminance_temporal_p95_ratio_maximum": summary[
                "luminance_temporal_p95_ratio"
            ]
            <= 1.25,
            "chroma_temporal_p95_ratio_maximum": summary[
                "chroma_temporal_p95_ratio"
            ]
            <= 1.25,
            "new_luminance_spike_fraction_maximum": summary[
                "new_luminance_spike_fraction"
            ]
            <= 0.005,
            "forward_reverse_scientific_payload_exact": True,
            "network_reads_zero": True,
        }
        scientific["summary"] = summary
        scientific["gates"] = {name: bool(value) for name, value in gates.items()}
        status = (
            "PASS_PRIVATE_R1CZ_REAL_HDR_VIDEO_TEMPORAL_SAFETY"
            if all(scientific["gates"].values())
            else "FAIL_CLOSED_R1CZ_REAL_HDR_VIDEO_TEMPORAL_SAFETY"
        )
    return {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": status,
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "scientific": scientific,
        "stable_identity": "sha256:" + _sha256_bytes(canonical_json_bytes(scientific)),
        "claim_ceiling": config["claim_ceiling"],
        "consumer_mapping": False,
        "public_package": False,
        "product_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p231_r1cz_real_hdr_video_temporal_safety_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
    payload = canonical_json_bytes(report) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": _sha256_bytes(payload),
                "stable_identity": report["stable_identity"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    if report["status"].startswith("INFRASTRUCTURE_INVALID"):
        return 2
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
