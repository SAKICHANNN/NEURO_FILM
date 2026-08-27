from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_p289_libavif_gainmap_encoder_d0 as p289
from src.preprocess.libavif_gainmap_encoder import (
    LibavifGainMapEncoderError,
    encode_gainmap_avif_create_only_v1,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _expect_rejection(arguments: dict[str, Any], destination: Path) -> bool:
    before = destination.read_bytes() if destination.exists() else None
    try:
        encode_gainmap_avif_create_only_v1(**arguments)
    except LibavifGainMapEncoderError:
        if before is not None:
            return destination.read_bytes() == before
        return not destination.exists()
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
    p289_config_path = ROOT / config["parents"]["p289_config"][0]
    parent = _load(p289_config_path)
    avifdec, gainmaputil = p289._validate_runtime(parent)
    source = p289._validate_source(parent)
    source_before = _sha256_file(source)
    runtime_before = {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    fixture = config["fixture"]
    profile = config["profile"]
    controls = [
        "foreign-destination",
        "missing-runtime",
        "nonfinite-headroom",
        "wrong-cicp",
        "wrong-decoder-hash",
        "wrong-dimensions",
        "wrong-endpoint-hash",
        "wrong-layout",
        "wrong-runtime-hash",
        "forced-encoder-failure",
    ]
    if reverse:
        controls.reverse()

    with tempfile.TemporaryDirectory(prefix="p290_", dir=ROOT / "tmp") as raw:
        scratch = Path(raw)
        hdr, sdr, _hdr_pixels, _sdr_pixels, endpoint_hashes = (
            p289._build_endpoints(parent, source, avifdec, gainmaputil, scratch)
        )
        endpoint_gates = {
            "hdr-pixel": endpoint_hashes["hdr_pixel_sha256"]
            == fixture["hdr_pixel_sha256"],
            "hdr-png": endpoint_hashes["hdr_png_sha256"]
            == fixture["hdr_png_sha256"],
            "sdr-pixel": endpoint_hashes["sdr_pixel_sha256"]
            == fixture["sdr_pixel_sha256"],
            "sdr-png": endpoint_hashes["sdr_png_sha256"]
            == fixture["sdr_png_sha256"],
        }
        if not all(endpoint_gates.values()):
            raise ValueError("P289 endpoint identity drift")
        destination = scratch / "published.avif"
        common: dict[str, Any] = {
            "hdr_endpoint": hdr.resolve(),
            "sdr_endpoint": sdr.resolve(),
            "destination": destination.resolve(),
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
        receipt = encode_gainmap_avif_create_only_v1(**common)
        media_bytes = destination.read_bytes()
        media_sha256 = _sha256_bytes(media_bytes)

        decoded_hdr = scratch / "decoded_hdr.png"
        decoded_sdr = scratch / "decoded_sdr.png"
        base_result = p289._run(
            p289._expand(
                parent["commands"]["hdr_endpoint"],
                avifdec,
                {"source": destination, "hdr_png": decoded_hdr},
            )
        )
        tone_result = p289._run(
            p289._expand(
                parent["commands"]["sdr_endpoint"],
                gainmaputil,
                {"source": destination, "sdr_png": decoded_sdr},
            )
        )
        if base_result.returncode != 0 or tone_result.returncode != 0:
            raise ValueError("published endpoint decode failed")
        decoded_hdr_pixels = p289._decode_png(decoded_hdr)
        decoded_sdr_pixels = p289._decode_png(decoded_sdr)

        uint8_path = scratch / "uint8.png"
        small_path = scratch / "small.png"
        fake_runtime = scratch / "fake.exe"
        assert cv2.imwrite(str(uint8_path), np.zeros((2, 2, 3), dtype=np.uint8))
        assert cv2.imwrite(str(small_path), np.zeros((2, 2, 3), dtype=np.uint16))
        fake_runtime.write_bytes(b"P290 forced encoder failure\n")
        uint8_hash = _sha256_file(uint8_path)
        small_hash = _sha256_file(small_path)
        fake_hash = _sha256_file(fake_runtime)
        control_results: dict[str, bool] = {}
        for index, role in enumerate(controls):
            control_destination = scratch / f"control_{index}.avif"
            arguments = {**common, "destination": control_destination.resolve()}
            if role == "foreign-destination":
                control_destination.write_bytes(b"P290 foreign destination\n")
            elif role == "missing-runtime":
                arguments["runtime"] = (scratch / "missing.exe").resolve()
            elif role == "nonfinite-headroom":
                arguments["base_headroom"] = float("nan")
            elif role == "wrong-cicp":
                arguments["base_cicp"] = (9, 16, 0)
            elif role == "wrong-decoder-hash":
                arguments["expected_decoder_sha256"] = "0" * 64
            elif role == "wrong-dimensions":
                arguments["hdr_endpoint"] = small_path.resolve()
                arguments["expected_hdr_sha256"] = small_hash
            elif role == "wrong-endpoint-hash":
                arguments["expected_hdr_sha256"] = "0" * 64
            elif role == "wrong-layout":
                arguments["hdr_endpoint"] = uint8_path.resolve()
                arguments["expected_hdr_sha256"] = uint8_hash
            elif role == "wrong-runtime-hash":
                arguments["expected_runtime_sha256"] = "0" * 64
            elif role == "forced-encoder-failure":
                arguments["runtime"] = fake_runtime.resolve()
                arguments["expected_runtime_sha256"] = fake_hash
            else:
                raise AssertionError(role)
            control_results[role] = _expect_rejection(
                arguments, control_destination
            )

        receipt_dict = receipt.to_dict()
        receipt_sha256 = _sha256_bytes(_canonical(receipt_dict))
        sibling_stages = sorted(
            path.name for path in scratch.iterdir() if path.name.endswith(".stage")
        )
        candidate_gates = {
            "decoded-hdr-exact-p289": _sha256_bytes(decoded_hdr_pixels.tobytes())
            == fixture["expected_decoded_hdr_pixel_sha256"],
            "decoded-sdr-exact-p289": _sha256_bytes(decoded_sdr_pixels.tobytes())
            == fixture["expected_decoded_sdr_pixel_sha256"],
            "media-bytes-exact-p289": len(media_bytes)
            == int(fixture["expected_output_bytes"]),
            "media-sha-exact-p289": media_sha256
            == fixture["expected_output_sha256"],
            "receipt-output-exact": receipt.bytes == len(media_bytes)
            and receipt.output_sha256 == media_sha256,
            "zero-sibling-stage-residue": not sibling_stages,
        }
        destination.unlink()
        scientific = {
            "candidate_count": "2/3",
            "candidate_gates": dict(sorted(candidate_gates.items())),
            "control_results": dict(sorted(control_results.items())),
            "decoded_hdr_pixel_sha256": _sha256_bytes(
                decoded_hdr_pixels.tobytes()
            ),
            "decoded_sdr_pixel_sha256": _sha256_bytes(
                decoded_sdr_pixels.tobytes()
            ),
            "endpoint_gates": dict(sorted(endpoint_gates.items())),
            "media_bytes": len(media_bytes),
            "media_sha256": media_sha256,
            "receipt": receipt_dict,
            "receipt_sha256": receipt_sha256,
            "zero_network_reads": True,
            "zero_persistent_media": not destination.exists(),
        }

    source_runtime_immutable = source_before == _sha256_file(source) and runtime_before == {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    gates = {
        "all-invalid-controls-atomic": all(control_results.values()),
        "candidate-exact-p289": all(candidate_gates.values()),
        "endpoints-exact-p289": all(endpoint_gates.values()),
        "source-runtime-immutable": source_runtime_immutable,
        "zero-persistent-media": scientific["zero_persistent_media"],
    }
    scientific["gates"] = dict(sorted(gates.items()))
    status = (
        "PASS_PRIVATE_LIBAVIF_GAINMAP_CREATE_ONLY_API"
        if all(gates.values())
        else "FAIL_CLOSED_LIBAVIF_GAINMAP_CREATE_ONLY_API"
    )
    stable_identity = "sha256:" + _sha256_bytes(_canonical(scientific))
    return {
        "claim_ceiling": config["claim_ceiling"],
        "schema": "neuro-film.p290-libavif-gainmap-create-only-api-result.v1",
        "scientific": scientific,
        "stable_identity": stable_identity,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run((ROOT / arguments.config).resolve(), reverse=arguments.order == "reverse")
    output = (ROOT / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
