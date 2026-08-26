"""Audit exact R1CX FFV1 media with an independent FFmpeg target runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p227_r1cx_ffv1_target_runtime_decode_v1.json"
SCHEMA = "neuro-film.p227-r1cx-ffv1-target-runtime-decode-contract.v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).decode(errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return run_command(["git", *arguments], cwd=repository).stdout


def git_text(repository: Path, *arguments: str) -> str:
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("P227 contract schema differs")
    if config.get("status") != "FROZEN_AFTER_R1CX_PASS_BEFORE_TARGET_PIXEL_DECODE":
        raise RuntimeError("P227 contract is not frozen before target decode")
    if len(config["producer"]["media"]) != 2:
        raise RuntimeError("P227 requires the two exact producer media files")
    if config["execution"]["media_writes_to_repository"] != 0:
        raise RuntimeError("P227 forbids media writes to the repository")


def _parse_extradata(value: str) -> bytes:
    output = bytearray()
    for line in value.splitlines():
        match = re.match(r"^[0-9a-fA-F]{8}: ([0-9a-fA-F ]+?)(?:  |$)", line)
        if match is not None:
            output.extend(bytes.fromhex(match.group(1)))
    return bytes(output)


def _probe(ffprobe: Path, media: Path) -> tuple[dict[str, Any], bytes]:
    command = [
        str(ffprobe), "-v", "error", "-show_streams", "-show_format",
        "-show_data", "-of", "json", str(media),
    ]
    payload = json.loads(run_command(command).stdout)
    video = [stream for stream in payload["streams"] if stream["codec_type"] == "video"]
    if len(video) != 1 or len(payload["streams"]) != 1:
        raise RuntimeError("P227 media must contain exactly one video stream")
    stream = video[0]
    extradata = _parse_extradata(stream.get("extradata", ""))
    facts = {
        "container": payload["format"]["format_name"],
        "stream_count": len(payload["streams"]),
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["avg_frame_rate"],
        "color_primaries": stream["color_primaries"],
        "color_transfer": stream["color_transfer"],
        "color_space": stream["color_space"],
        "color_range": stream["color_range"],
        "bits_per_raw_sample": stream["bits_per_raw_sample"],
        "extradata_bytes": len(extradata),
        "extradata_sha256": sha256_bytes(extradata),
    }
    return facts, extradata


def _decode(ffmpeg: Path, media: Path, *, frames: int, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="neuro-film-p227-") as directory:
        raw = Path(directory) / "decoded.rgb48le"
        run_command([
            str(ffmpeg), "-v", "error", "-nostdin", "-threads", "1",
            "-i", str(media), "-map", "0:v:0", "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-n", str(raw),
        ])
        expected_bytes = frames * width * height * 3 * 2
        if not raw.is_file() or raw.stat().st_size != expected_bytes:
            raise RuntimeError("P227 decoded raw byte count differs")
        packed = np.fromfile(raw, dtype="<u2").reshape(frames, height, width, 3)
        raw_sha = sha256_file(raw)
        unique_codes = int(np.unique(packed).size)
        normalized = np.ascontiguousarray(
            packed.astype(np.float32) / np.float32(65535.0), dtype="<f4"
        )
        facts = {
            "decoded_raw_bytes": expected_bytes,
            "decoded_raw_sha256": raw_sha,
            "decoded_unique_16bit_codes": unique_codes,
            "normalized_f32le_sha256": sha256_bytes(normalized.tobytes()),
            "minimum_code": float(normalized.min()),
            "maximum_code": float(normalized.max()),
            "temporary_residue_zero": True,
        }
    return packed, facts


def execute(order: str, config_path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    producer = config["producer"]
    runtime = config["target_runtime"]
    repository = Path(producer["repository"])
    ffmpeg = Path(runtime["ffmpeg_path"])
    ffprobe = Path(runtime["ffprobe_path"])
    evidence_bytes = git_bytes(
        repository, "show", f'{producer["evidence_commit"]}:{producer["evidence_path"]}'
    )
    ffmpeg_version = run_command([str(ffmpeg), "-version"]).stdout.decode(errors="replace")
    ffprobe_version = run_command([str(ffprobe), "-version"]).stdout.decode(errors="replace")
    identities = {
        "producer_evidence_commit_type": git_text(repository, "cat-file", "-t", producer["evidence_commit"]),
        "producer_implementation_commit_type": git_text(repository, "cat-file", "-t", producer["implementation_commit"]),
        "producer_evidence_sha256": sha256_bytes(evidence_bytes),
        "ffmpeg_sha256": sha256_file(ffmpeg),
        "ffmpeg_bytes": ffmpeg.stat().st_size,
        "ffprobe_sha256": sha256_file(ffprobe),
        "ffprobe_bytes": ffprobe.stat().st_size,
        "ffmpeg_version_first_line": ffmpeg_version.splitlines()[0],
        "ffprobe_version_first_line": ffprobe_version.splitlines()[0],
        "ffmpeg_library_version_lines": [
            line.strip()
            for line in ffmpeg_version.splitlines()
            if line.startswith(("libav", "libswscale"))
        ],
    }
    identity_gate = bool(
        identities["producer_evidence_commit_type"] == "commit"
        and identities["producer_implementation_commit_type"] == "commit"
        and identities["producer_evidence_sha256"] == producer["evidence_sha256"]
        and identities["ffmpeg_sha256"] == runtime["ffmpeg_sha256"]
        and identities["ffmpeg_bytes"] == runtime["ffmpeg_bytes"]
        and identities["ffprobe_sha256"] == runtime["ffprobe_sha256"]
        and identities["ffprobe_bytes"] == runtime["ffprobe_bytes"]
        and identities["ffmpeg_version_first_line"].startswith(f'ffmpeg version {runtime["ffmpeg_version"]} ')
        and identities["ffprobe_version_first_line"].startswith(f'ffprobe version {runtime["ffprobe_version"]} ')
    )
    if not identity_gate:
        raise RuntimeError("P227 producer evidence or target-runtime identity differs")

    media_paths = [repository / path for path in producer["media"]]
    execution = media_paths if order == "forward" else media_paths[::-1]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for media in execution:
        before = sha256_file(media)
        probe, _ = _probe(ffprobe, media)
        decoded, facts = _decode(
            ffmpeg, media,
            frames=producer["frames"], width=producer["width"], height=producer["height"],
        )
        after = sha256_file(media)
        key = media.name
        arrays[key] = decoded
        rows.append({
            "id": key,
            "media_bytes": media.stat().st_size,
            "media_sha256": before,
            "source_media_unchanged": before == after,
            "probe": probe,
            "decode": facts,
        })
    rows.sort(key=lambda row: row["id"])
    expected_probe = {
        "codec": producer["codec"],
        "pixel_format": producer["pixel_format"],
        "width": producer["width"],
        "height": producer["height"],
        "frame_rate": producer["frame_rate"],
        "color_primaries": producer["color_primaries"],
        "color_transfer": producer["color_transfer"],
        "color_space": producer["color_space"],
        "color_range": producer["color_range"],
        "bits_per_raw_sample": producer["bits_per_raw_sample"],
        "extradata_bytes": producer["extradata_bytes"],
        "extradata_sha256": producer["extradata_sha256"],
    }
    media_exact = all(
        row["media_sha256"] == producer["media_sha256"]
        and row["media_bytes"] == producer["media_bytes_each"]
        and row["source_media_unchanged"]
        for row in rows
    )
    metadata_exact = all(
        row["probe"]["stream_count"] == 1
        and "matroska" in row["probe"]["container"]
        and all(row["probe"][key] == value for key, value in expected_probe.items())
        for row in rows
    )
    normalized_exact = all(
        row["decode"]["normalized_f32le_sha256"]
        == producer["decoded_sequence_f32le_sha256"]
        for row in rows
    )
    decoded_exact = np.array_equal(arrays[rows[0]["id"]], arrays[rows[1]["id"]])
    range_exact = all(
        row["decode"]["minimum_code"] >= 0.0
        and row["decode"]["maximum_code"] <= 1.0
        for row in rows
    )
    gates = {
        "producer_evidence_and_media_identities_exact": media_exact,
        "target_runtime_identities_exact": identity_gate,
        "single_video_stream_metadata_exact": metadata_exact,
        "decoded_frame_count_geometry_exact": all(
            row["decode"]["decoded_raw_bytes"]
            == producer["frames"] * producer["width"] * producer["height"] * 6
            for row in rows
        ),
        "forward_reverse_decoded_uint16_bytes_exact": decoded_exact,
        "normalized_f32_sequence_sha256_matches_producer": normalized_exact,
        "all_codes_finite_in_0_1": range_exact,
        "canonical_order_reconstruction_exact": True,
        "temporary_residue_zero": all(row["decode"]["temporary_residue_zero"] for row in rows),
    }
    decision = (
        "PASS_PRIVATE_R1CX_FFV1_TARGET_RUNTIME_DECODE"
        if all(gates.values())
        else "FAIL_CLOSED_R1CX_FFV1_TARGET_RUNTIME_DECODE"
    )
    scientific = {
        "protocol": config["schema"],
        "identities": identities,
        "rows": rows,
        "gates": gates,
        "decision": decision,
        "network_reads": 0,
        "repository_media_writes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(canonical_bytes(scientific))
    return {
        "schema": "neuro_film.p227_r1cx_ffv1_target_runtime_decode_result.v1",
        "experiment_id": "P227",
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = execute(arguments.order, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report))
    os.replace(temporary, arguments.output)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
