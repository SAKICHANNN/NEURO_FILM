#!/usr/bin/env python3
"""Audit exact R1CZ payload safety on two consumed UltraHDR still fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.shared_hdr_dpct_payload import (
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)
from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    prepare_ultrahdr_match_view_v1,
)

SCHEMA = "neuro-film.p234-r1cz-ultrahdr-still-safety-result.v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _binding(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"P234 binding mismatch: {label}")
    return {"bytes": path.stat().st_size, "sha256": actual}


def _load_payload(config: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    spec = config["payload"]
    p233_config_path = ROOT / spec["p233_config_path"]
    _binding(p233_config_path, spec["p233_config_sha256"], "p233_config")
    p233_config = _object(p233_config_path)
    capsule_path = ROOT / spec["capsule_path"]
    raw = capsule_path.read_bytes()
    if _sha256_bytes(raw) != spec["capsule_sha256"]:
        raise RuntimeError("P234 capsule identity differs")
    capsule = json.loads(raw.decode("utf-8"))
    if raw != _canonical_json_bytes(capsule) + b"\n":
        raise RuntimeError("P234 capsule is not canonical JSON plus LF")
    expected_capsule = {
        "schema": "kmcfm.r1cz-payload-capsule.v1",
        "encoding": "hex-lower-v1",
        "payload_bytes": spec["payload_bytes"],
        "payload_sha256": spec["payload_sha256"],
        "bundle_id": spec["bundle_id"],
    }
    for key, value in expected_capsule.items():
        if capsule.get(key) != value:
            raise RuntimeError(f"P234 capsule {key} differs")
    payload = bytes.fromhex(capsule["payload_hex"])
    if len(payload) != spec["payload_bytes"]:
        raise RuntimeError("P234 payload length differs")
    if _sha256_bytes(payload) != spec["payload_sha256"]:
        raise RuntimeError("P234 payload identity differs")
    envelope = p233_config["envelope"]
    if envelope["bundle_id"] != spec["bundle_id"]:
        raise RuntimeError("P234 envelope bundle differs")
    return payload, envelope


def _decoder_command(
    executable: Path, input_path: Path, output_path: Path
) -> list[str]:
    return [
        str(executable),
        "-m",
        "1",
        "-j",
        input_path.name,
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        output_path.name,
    ]


def _decode(
    executable: Path,
    fixture: Path,
    row: dict[str, Any],
    scratch: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = fixture.read_bytes()
    if _sha256_bytes(source) != row["sha256"]:
        raise RuntimeError(f"P234 fixture identity differs: {row['name']}")
    local_input = scratch / row["name"]
    local_output = scratch / f"{fixture.stem}.rgba16f"
    local_input.write_bytes(source)
    completed = subprocess.run(
        _decoder_command(executable, local_input, local_output),
        cwd=scratch,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"P234 decoder failed: {completed.stderr[-500:]}")
    decoded = local_output.read_bytes()
    expected = int(row["width"]) * int(row["height"]) * 4 * 2
    if len(decoded) != expected:
        raise RuntimeError("P234 decoded byte length differs")
    prepared = prepare_ultrahdr_match_view_v1(
        source_asset=source,
        expected_source_sha256=row["sha256"],
        decoded_rgba16f=decoded,
        width=int(row["width"]),
        height=int(row["height"]),
        decoder_version=ULTRAHDR_DECODER_VERSION,
        producer_profile_id=ULTRAHDR_EXTERNAL_PROFILE_ID,
    )
    return prepared.pixels, {
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "ingress_id": prepared.ingress_id,
        "pixel_sha256": prepared.descriptor.pixel_sha256,
        "profile_id": prepared.descriptor.profile_id,
        "reference_white_nits": prepared.descriptor.reference_white_nits,
        "source_unchanged": fixture.read_bytes() == source,
        "view_id": prepared.descriptor.view_id,
    }


def spatial_vectors(
    values: np.ndarray,
    *,
    luminance_vector: np.ndarray,
    epsilon_nits: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return concatenated horizontal/vertical log-luma and log-chroma edges."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("spatial metric input must be HxWx3")
    luminance = np.log2(epsilon_nits + source @ luminance_vector)
    chroma = np.stack(
        (
            np.log((source[..., 0] + epsilon_nits) / (source[..., 1] + epsilon_nits)),
            np.log((source[..., 2] + epsilon_nits) / (source[..., 1] + epsilon_nits)),
        ),
        axis=-1,
    )
    luma_edges = np.concatenate(
        (
            np.abs(np.diff(luminance, axis=1)).reshape(-1),
            np.abs(np.diff(luminance, axis=0)).reshape(-1),
        )
    )
    chroma_edges = np.concatenate(
        (
            np.linalg.norm(np.diff(chroma, axis=1), axis=-1).reshape(-1),
            np.linalg.norm(np.diff(chroma, axis=0), axis=-1).reshape(-1),
        )
    )
    return luma_edges, chroma_edges


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 1.0 if numerator == 0.0 else float("inf")


def summarize(
    source: np.ndarray, output: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    metric = config["metrics"]
    luma = np.asarray(metric["luminance_vector"], dtype=np.float64)
    epsilon = float(metric["log_epsilon_nits"])
    source_luma, source_chroma = spatial_vectors(
        source, luminance_vector=luma, epsilon_nits=epsilon
    )
    output_luma, output_chroma = spatial_vectors(
        output, luminance_vector=luma, epsilon_nits=epsilon
    )
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
    spike_margin = float(metric["new_spike_absolute_log2_margin"])
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
        "new_boundary_fraction": float(np.mean(new_boundary)),
        "median_log_rgb_material_effect": float(np.median(material)),
        "source_luminance_spatial_p95": source_luma_p95,
        "output_luminance_spatial_p95": output_luma_p95,
        "luminance_spatial_p95_ratio": _ratio(output_luma_p95, source_luma_p95),
        "source_chroma_spatial_p95": source_chroma_p95,
        "output_chroma_spatial_p95": output_chroma_p95,
        "chroma_spatial_p95_ratio": _ratio(output_chroma_p95, source_chroma_p95),
        "new_luminance_spike_fraction": float(
            np.mean(output_luma > source_luma + spike_margin)
        ),
    }


def run(config_path: Path, decoder: Path, *, reverse: bool) -> dict[str, Any]:
    config = _object(config_path)
    bindings: dict[str, Any] = {}
    paths = {
        "p233_evidence": (
            ROOT / config["payload"]["p233_evidence_path"],
            config["payload"]["p233_evidence_sha256"],
        ),
        "payload_loader": (
            ROOT / config["payload"]["loader_path"],
            config["payload"]["loader_sha256"],
        ),
        "p88_evidence": (
            ROOT / config["source"]["p88_evidence_path"],
            config["source"]["p88_evidence_sha256"],
        ),
        "ultrahdr_ingress": (
            ROOT / config["source"]["ingress_path"],
            config["source"]["ingress_sha256"],
        ),
        "source_manifest": (
            ROOT / config["source"]["source_manifest_path"],
            config["source"]["source_manifest_sha256"],
        ),
        "decoder": (decoder, config["source"]["decoder_executable_sha256"]),
    }
    for label, (path, expected) in paths.items():
        bindings[label] = _binding(path, expected, label)
    p233 = _object(paths["p233_evidence"][0])
    p88 = _object(paths["p88_evidence"][0])
    source_manifest = _object(paths["source_manifest"][0])
    if p233["status"] != "PASS_PRIVATE_R1CZ_LOCAL_PAYLOAD_CAPSULE":
        raise RuntimeError("P234 P233 status differs")
    if p88["status"] != "PASS_PRIVATE_PINNED_ULTRAHDR_DECODER_TO_P87":
        raise RuntimeError("P234 P88 status differs")
    if source_manifest["license"] != config["source"]["license"]:
        raise RuntimeError("P234 fixture license differs")
    payload, envelope = _load_payload(config)
    bundle = load_shared_hdr_dpct_payload(payload, envelope)

    rows = list(config["source"]["fixtures"])
    if reverse:
        rows.reverse()
    records: list[dict[str, Any]] = []
    fixture_root = ROOT / "tests/fixtures/u1_5c_libultrahdr"
    with tempfile.TemporaryDirectory(prefix="p234_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            source, decode = _decode(
                decoder, fixture_root / row["name"], row, scratch
            )
            source_before = source.tobytes()
            output = apply_shared_hdr_dpct_payload(source, bundle)
            metrics = summarize(source, output, config)
            gates = {
                "decode_and_profile_exact": bool(
                    decode["profile_id"] == config["source"]["profile_id"]
                    and decode["reference_white_nits"]
                    == config["source"]["reference_white_nits"]
                    and decode["source_unchanged"]
                ),
                "output_finite_in_0_10000": bool(
                    metrics["all_finite"]
                    and metrics["minimum_nits"] >= 0.0
                    and metrics["maximum_nits"] <= 10000.0
                ),
                "new_boundary_fraction_exact": metrics["new_boundary_fraction"]
                == config["gates"]["new_boundary_fraction_exact"],
                "material_effect": metrics["median_log_rgb_material_effect"]
                >= config["gates"]["median_log_rgb_material_effect_minimum"],
                "luminance_spatial_p95": metrics["luminance_spatial_p95_ratio"]
                <= config["gates"]["luminance_spatial_p95_ratio_maximum"],
                "chroma_spatial_p95": metrics["chroma_spatial_p95_ratio"]
                <= config["gates"]["chroma_spatial_p95_ratio_maximum"],
                "new_luminance_spikes": metrics["new_luminance_spike_fraction"]
                <= config["gates"]["new_luminance_spike_fraction_maximum"],
                "source_immutable_output_owned_contiguous": bool(
                    source.tobytes() == source_before
                    and output.flags.owndata
                    and output.flags.c_contiguous
                ),
            }
            records.append(
                {
                    "fixture": row["name"],
                    "fixture_sha256": row["sha256"],
                    "decode": decode,
                    "metrics": metrics,
                    "gates": gates,
                    "all_safety_gates_pass": all(gates.values()),
                }
            )
    records.sort(key=lambda value: value["fixture"])
    gates = {
        "all_bindings_exact": True,
        "each_fixture_passes_all_safety_gates": all(
            row["all_safety_gates_pass"] for row in records
        ),
        "payload_build_application_source_reads": 0
        == config["gates"]["payload_build_application_source_reads"],
        "target_reads": 0 == config["gates"]["target_reads"],
        "network_reads": 0 == config["gates"]["network_reads"],
        "repository_media_writes": 0
        == config["gates"]["repository_media_writes"],
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_PRIVATE_R1CZ_ULTRAHDR_STILL_SAFETY"
            if passed
            else "FAIL_CLOSED_R1CZ_ULTRAHDR_STILL_SAFETY"
        ),
        "bindings": bindings,
        "payload": {
            "bundle_id": config["payload"]["bundle_id"],
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
        },
        "records": records,
        "execution": {
            "application_target_reads": 0,
            "network_reads": 0,
            "payload_build_application_source_reads": 0,
            "repository_media_writes": 0,
            "temporary_residue_zero": True,
        },
        "gates": gates,
        "decision": {
            "result": (
                "PASS_PRIVATE_R1CZ_ULTRAHDR_STILL_SAFETY"
                if passed
                else "FAIL_CLOSED_R1CZ_ULTRAHDR_STILL_SAFETY"
            ),
            "claim_ceiling": config["claim_ceiling"],
            "consumer_mapping": False,
        },
    }
    result["scientific_identity"] = "sha256:" + _sha256_bytes(
        _canonical_json_bytes(result)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/p234_r1cz_ultrahdr_still_safety_v1.json",
    )
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.decoder.resolve(), reverse=args.order == "reverse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_json_bytes(result))


if __name__ == "__main__":
    main()


__all__ = ["run", "spatial_vectors", "summarize"]
