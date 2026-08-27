#!/usr/bin/env python3
"""Audit exact P290 gain-map AVIF through the frozen P283/P284 PQ rail."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_p289_libavif_gainmap_encoder_d0 as p289
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from src.color_match.libavif_gainmap_ingress import (
    prepare_libavif_gainmap_match_view_v1,
)
from src.preprocess.color_management import linear_rgb_matrix
from src.preprocess.libavif_gainmap_encoder import (
    LibavifGainMapEncoderError,
    encode_gainmap_avif_create_only_v1,
)
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.rec2100_pq_transfer import (
    absolute_rec2020_cdm2_to_pq_rgb16,
    pq_to_absolute_rec2020_cdm2,
    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
)

REPORT_SCHEMA = "neuro-film.p291-libavif-gainmap-encode-pq-roundtrip-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _bound(binding: list[Any]) -> Path:
    path = ROOT / str(binding[0])
    if (
        not path.is_file()
        or path.stat().st_size != int(binding[1])
        or _sha256_file(path) != str(binding[2])
    ):
        raise ValueError(f"bound file identity mismatch: {binding[0]}")
    return path


def code_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return deterministic uint16 absolute-error summaries."""

    if reference.dtype != np.uint16 or candidate.dtype != np.uint16:
        raise TypeError("code-error inputs must be uint16")
    if reference.shape != candidate.shape:
        raise ValueError("code-error input shapes differ")
    difference = np.abs(reference.astype(np.int32) - candidate.astype(np.int32))
    return {
        "maximum": float(np.max(difference)),
        "median": float(np.median(difference)),
        "p95": float(np.percentile(difference, 95.0)),
    }


def direct_bt709_pq_to_rec2020_pq_rgb16(encoded_rgb16: np.ndarray) -> np.ndarray:
    """Independent explicit oracle for the frozen P283/P284 colour path."""

    if encoded_rgb16.dtype != np.uint16:
        raise TypeError("direct oracle input must be uint16")
    if encoded_rgb16.ndim != 3 or encoded_rgb16.shape[2] != 3:
        raise ValueError("direct oracle input must be HxWx3")
    absolute_srgb = pq_to_absolute_rec2020_cdm2(
        encoded_rgb16.astype(np.float64) / 65535.0
    )
    matrix = linear_rgb_matrix("linear_srgb", "linear_rec2020")
    absolute_rec2020 = np.matmul(absolute_srgb, matrix.T)
    return absolute_rec2020_cdm2_to_pq_rgb16(absolute_rec2020)


def _rejected(callable_: Any, output: Path) -> bool:
    before = output.read_bytes() if output.exists() else None
    try:
        callable_()
    except (
        FileExistsError,
        LibavifGainMapEncoderError,
        ReferenceMatchContractError,
        ValueError,
    ):
        return (
            output.read_bytes() == before if before is not None else not output.exists()
        )
    return False


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load(config_path)
    for binding in (
        config["contract"],
        *config["parents"].values(),
        *config["implementation"].values(),
        *config.get("execution_bindings", {}).values(),
    ):
        _bound(binding)

    parent = _load(ROOT / config["parents"]["p289_config"][0])
    avifdec, gainmaputil = p289._validate_runtime(parent)
    source = p289._validate_source(parent)
    source_before = _sha256_file(source)
    runtime_before = {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    profile = config["profile"]
    fixture = config["fixture"]
    limits = config["gates"]
    control_order = [
        "foreign-destination",
        "wrong-media-hash",
        "wrong-role",
        "nonfinite",
    ]
    if reverse:
        control_order.reverse()

    with tempfile.TemporaryDirectory(prefix="p291_", dir=ROOT / "tmp") as raw:
        scratch = Path(raw)
        hdr, sdr, hdr_bgr, _direct_sdr_bgr, endpoint_hashes = p289._build_endpoints(
            parent, source, avifdec, gainmaputil, scratch
        )
        endpoint_gates = {
            "hdr-pixel": endpoint_hashes["hdr_pixel_sha256"]
            == fixture["hdr_pixel_sha256"],
            "hdr-png": endpoint_hashes["hdr_png_sha256"] == fixture["hdr_png_sha256"],
            "sdr-pixel": endpoint_hashes["sdr_pixel_sha256"]
            == fixture["sdr_pixel_sha256"],
            "sdr-png": endpoint_hashes["sdr_png_sha256"] == fixture["sdr_png_sha256"],
        }
        if not all(endpoint_gates.values()):
            raise ValueError("P289 endpoint identity drift")

        candidate = scratch / "candidate.avif"
        encode_arguments: dict[str, Any] = {
            "hdr_endpoint": hdr.resolve(),
            "sdr_endpoint": sdr.resolve(),
            "destination": candidate.resolve(),
            "expected_hdr_sha256": fixture["hdr_png_sha256"],
            "expected_sdr_sha256": fixture["sdr_png_sha256"],
            "runtime": gainmaputil.resolve(),
            "expected_runtime_sha256": runtime_before[gainmaputil.name],
            "decoder": avifdec.resolve(),
            "expected_decoder_sha256": runtime_before[avifdec.name],
            "base_cicp": tuple(profile["base_cicp"]),
            "alternate_cicp": tuple(profile["alternate_cicp"]),
            "base_headroom": float(profile["base_headroom"]),
            "alternate_headroom": float(profile["alternate_headroom"]),
        }
        receipt = encode_gainmap_avif_create_only_v1(**encode_arguments)
        candidate_bytes = candidate.read_bytes()
        candidate_sha256 = _sha256_bytes(candidate_bytes)
        receipt_sha256 = _sha256_bytes(_canonical(receipt.to_dict()))

        decoded_hdr_path = scratch / "decoded_hdr.png"
        decoded_sdr_path = scratch / "decoded_sdr.png"
        decoded_hdr_process = p289._run(
            p289._expand(
                parent["commands"]["hdr_endpoint"],
                avifdec,
                {"source": candidate, "hdr_png": decoded_hdr_path},
            )
        )
        decoded_sdr_process = p289._run(
            p289._expand(
                parent["commands"]["sdr_endpoint"],
                gainmaputil,
                {"source": candidate, "sdr_png": decoded_sdr_path},
            )
        )
        if decoded_hdr_process.returncode != 0 or decoded_sdr_process.returncode != 0:
            raise ValueError("candidate endpoint decode failed")
        decoded_hdr_bgr = p289._decode_png(decoded_hdr_path)
        decoded_sdr_bgr = p289._decode_png(decoded_sdr_path)
        decoded_hdr_rgb = np.ascontiguousarray(decoded_hdr_bgr[:, :, ::-1])
        direct_hdr_rgb = np.ascontiguousarray(hdr_bgr[:, :, ::-1])
        decoded_vs_direct = code_error(direct_hdr_rgb, decoded_hdr_rgb)

        prepared = prepare_libavif_gainmap_match_view_v1(
            source_asset=candidate_bytes,
            expected_source_sha256=candidate_sha256,
            encoded_rgb16=decoded_hdr_rgb,
            decoder_version=str(profile["decoder_version"]),
            source_color_primaries=int(profile["base_cicp"][0]),
            source_transfer_characteristics=int(profile["base_cicp"][1]),
            source_full_range=True,
            higher_rendition_role=str(profile["higher_rendition_role"]),
        )
        output = scratch / "candidate.pq.png"
        repeat = scratch / "candidate.repeat.pq.png"
        png_sha256, pq_samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
            prepared.pixels, output, row_count=int(profile["png_row_count"])
        )
        repeat_sha256, repeat_samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
            prepared.pixels, repeat, row_count=int(profile["png_row_count"])
        )
        expected_from_decoded = direct_bt709_pq_to_rec2020_pq_rgb16(decoded_hdr_rgb)
        expected_from_direct = direct_bt709_pq_to_rec2020_pq_rgb16(direct_hdr_rgb)
        pq_vs_decoded = code_error(expected_from_decoded, pq_samples)
        pq_vs_direct = code_error(expected_from_direct, pq_samples)

        controls: dict[str, bool] = {}
        for role in control_order:
            control_output = scratch / f"control-{role}.bin"
            if role == "foreign-destination":
                control_output.write_bytes(b"P291 foreign destination\n")
                arguments = {
                    **encode_arguments,
                    "destination": control_output.resolve(),
                }
                controls[role] = _rejected(
                    lambda arguments=arguments: encode_gainmap_avif_create_only_v1(
                        **arguments
                    ),
                    control_output,
                )
            elif role == "wrong-media-hash":
                controls[role] = _rejected(
                    lambda: prepare_libavif_gainmap_match_view_v1(
                        source_asset=candidate_bytes,
                        expected_source_sha256="0" * 64,
                        encoded_rgb16=decoded_hdr_rgb,
                        decoder_version=str(profile["decoder_version"]),
                        source_color_primaries=1,
                        source_transfer_characteristics=16,
                        source_full_range=True,
                        higher_rendition_role="base",
                    ),
                    control_output,
                )
            elif role == "wrong-role":
                controls[role] = _rejected(
                    lambda: prepare_libavif_gainmap_match_view_v1(
                        source_asset=candidate_bytes,
                        expected_source_sha256=candidate_sha256,
                        encoded_rgb16=decoded_hdr_rgb,
                        decoder_version=str(profile["decoder_version"]),
                        source_color_primaries=1,
                        source_transfer_characteristics=16,
                        source_full_range=True,
                        higher_rendition_role="lower",
                    ),
                    control_output,
                )
            elif role == "nonfinite":
                nonfinite_output = control_output.with_suffix(".png")
                controls[role] = _rejected(
                    lambda nonfinite_output=nonfinite_output: (
                        save_absolute_rec2020_cdm2_to_pq_rgb16_png(
                            np.full((1, 1, 3), np.nan), nonfinite_output
                        )
                    ),
                    nonfinite_output,
                )
            else:
                raise AssertionError(role)

        create_only_unchanged = _rejected(
            lambda: save_absolute_rec2020_cdm2_to_pq_rgb16_png(prepared.pixels, output),
            output,
        )
        candidate_gates = {
            "decoded-hdr-exact-parent": _sha256_bytes(decoded_hdr_bgr.tobytes())
            == fixture["decoded_hdr_pixel_sha256"],
            "decoded-sdr-exact-parent": _sha256_bytes(decoded_sdr_bgr.tobytes())
            == fixture["decoded_sdr_pixel_sha256"],
            "media-bytes-exact-parent": len(candidate_bytes)
            == int(fixture["media_bytes"]),
            "media-sha-exact-parent": candidate_sha256 == fixture["media_sha256"],
            "receipt-exact-parent": receipt_sha256 == fixture["receipt_sha256"],
        }
        error_gates = {
            "decoded-vs-direct-hdr": decoded_vs_direct["median"]
            <= float(limits["decoded_vs_direct_hdr_median_code_error_max"])
            and decoded_vs_direct["p95"]
            <= float(limits["decoded_vs_direct_hdr_p95_code_error_max"])
            and decoded_vs_direct["maximum"]
            <= float(limits["decoded_vs_direct_hdr_max_code_error_max"]),
            "pq-vs-decoded-hdr": pq_vs_decoded["median"]
            <= float(limits["pq_vs_decoded_hdr_median_code_error_max"])
            and pq_vs_decoded["p95"]
            <= float(limits["pq_vs_decoded_hdr_p95_code_error_max"])
            and pq_vs_decoded["maximum"]
            <= float(limits["pq_vs_decoded_hdr_max_code_error_max"]),
            "pq-vs-direct-hdr": pq_vs_direct["median"]
            <= float(limits["pq_vs_direct_hdr_median_code_error_max"])
            and pq_vs_direct["p95"]
            <= float(limits["pq_vs_direct_hdr_p95_code_error_max"])
            and pq_vs_direct["maximum"]
            <= float(limits["pq_vs_direct_hdr_max_code_error_max"]),
        }
        publication_gates = {
            "create-only-atomic": create_only_unchanged,
            "input-ownership": prepared.pixels.flags.owndata
            and prepared.pixels.flags.c_contiguous
            and not prepared.pixels.flags.writeable,
            "invalid-controls-atomic": all(controls.values()),
            "profile-geometry": prepared.descriptor.profile_id
            == MATCH_PROFILE_ABSOLUTE_REC2020
            and prepared.descriptor.reference_white_nits
            == float(profile["reference_white_nits"])
            and prepared.pixels.shape
            == (int(fixture["height"]), int(fixture["width"]), 3),
            "range-finite": np.isfinite(prepared.pixels).all()
            and float(np.min(prepared.pixels)) >= 0.0
            and float(np.max(prepared.pixels)) <= 10000.0,
            "repeat-png-exact": png_sha256 == repeat_sha256
            and output.read_bytes() == repeat.read_bytes()
            and np.array_equal(pq_samples, repeat_samples),
            "sample-readback-exact": sha256_rec2100_pq_rgb16_png_samples(
                output, width=int(fixture["width"]), height=int(fixture["height"])
            )
            == _sha256_bytes(pq_samples.tobytes()),
        }
        scientific: dict[str, Any] = {
            "candidate_count": "2/3",
            "candidate_gates": dict(sorted(candidate_gates.items())),
            "controls": dict(sorted(controls.items())),
            "decoded_hdr_pixel_sha256": _sha256_bytes(decoded_hdr_bgr.tobytes()),
            "decoded_sdr_pixel_sha256": _sha256_bytes(decoded_sdr_bgr.tobytes()),
            "decoded_vs_direct_hdr_code_error": decoded_vs_direct,
            "endpoint_gates": dict(sorted(endpoint_gates.items())),
            "error_gates": dict(sorted(error_gates.items())),
            "ingress_id": prepared.ingress_id,
            "media_bytes": len(candidate_bytes),
            "media_sha256": candidate_sha256,
            "png_file_sha256": png_sha256,
            "png_sample_sha256": _sha256_bytes(pq_samples.tobytes()),
            "pq_vs_decoded_hdr_code_error": pq_vs_decoded,
            "pq_vs_direct_hdr_code_error": pq_vs_direct,
            "publication_gates": dict(sorted(publication_gates.items())),
            "receipt_sha256": receipt_sha256,
        }
        candidate.unlink()
        output.unlink()
        repeat.unlink()
        scientific["zero_persistent_media"] = not any(
            path.exists() for path in (candidate, output, repeat)
        )

    immutable = source_before == _sha256_file(source) and runtime_before == {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    gates = {
        "all-candidate-identities-exact": all(candidate_gates.values()),
        "all-endpoint-identities-exact": all(endpoint_gates.values()),
        "all-error-bounds-pass": all(error_gates.values()),
        "all-publication-gates-pass": all(publication_gates.values()),
        "source-runtime-immutable": immutable,
        "zero-persistent-media": bool(scientific["zero_persistent_media"]),
    }
    scientific["gates"] = dict(sorted(gates.items()))
    status = (
        "PASS_PRIVATE_LIBAVIF_GAINMAP_ENCODE_PQ_INTEGRATION"
        if all(gates.values())
        else "FAIL_CLOSED_LIBAVIF_GAINMAP_ENCODE_PQ_INTEGRATION"
    )
    return {
        "claim_ceiling": config["claim_ceiling"],
        "schema": REPORT_SCHEMA,
        "scientific": scientific,
        "stable_identity": "sha256:" + _sha256_bytes(_canonical(scientific)),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(
        (ROOT / arguments.config).resolve(), reverse=arguments.order == "reverse"
    )
    output = (ROOT / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
