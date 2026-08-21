#!/usr/bin/env python3
"""Formal P89 pinned Ultra HDR to absolute Rec.2020 PQ PNG audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    prepare_ultrahdr_match_view_v1,
)
from src.color_match.ultrahdr_pq import publish_ultrahdr_match_view_pq_png_v1
from src.preprocess import load_working_image
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.rec2100_pq_transfer import (
    PQ_C1,
    PQ_C2,
    PQ_C3,
    PQ_M1,
    PQ_M2,
    PQ_RGB16_MAXIMUM,
    absolute_rec2020_cdm2_to_pq,
    absolute_rec2020_cdm2_to_pq_rgb16,
    pq_to_absolute_rec2020_cdm2,
)

REPORT_SCHEMA = "neuro-film.p89-ultrahdr-absolute-rec2020-pq-png-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _decoder_command(executable: Path, input_name: str, output_name: str) -> list[str]:
    return [
        str(executable),
        "-m",
        "1",
        "-j",
        input_name,
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        output_name,
    ]


def _run_decoder(
    executable: Path,
    input_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _decoder_command(executable, input_path.name, output_path.name),
        cwd=input_path.parent,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _scalar_pq_sample(value: float) -> int:
    normalized = value / 10000.0
    powered = math.pow(normalized, 2610.0 / 16384.0)
    encoded = math.pow(
        ((3424.0 / 4096.0) + ((2413.0 / 4096.0) * 32.0) * powered)
        / (1.0 + ((2392.0 / 4096.0) * 32.0) * powered),
        (2523.0 / 4096.0) * 128.0,
    )
    return math.floor(encoded * 65535.0 + 0.5)


def _scalar_oracle_samples(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.fromiter(
        (_scalar_pq_sample(float(value)) for value in flat),
        dtype=np.uint16,
        count=flat.size,
    )
    return result.reshape(values.shape)


def _expected_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _production_rejection(path: Path, expected: str) -> str:
    try:
        load_working_image(path)
    except ValueError as error:
        message = str(error)
        if expected not in message:
            raise AssertionError("production loader failed for another reason") from error
        return message
    raise AssertionError("production loader accepted unsupported HDR input")


def _validate_bindings(
    config: dict[str, Any],
    config_path: Path,
    decoder_app: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    bindings = config["bindings"]
    paths = {
        "p87_evidence_sha256": ROOT
        / "docs/evidence/P87_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_RESULT.json",
        "p88_evidence_sha256": ROOT
        / "docs/evidence/P88_ULTRAHDR_PINNED_DECODER_CONSUMPTION_RESULT.json",
        "u1_4g_evidence_sha256": ROOT
        / "docs/evidence/U1_4G_REC2100_PQ_PNG_RAIL_RESULT.json",
        "p88_config_sha256": ROOT
        / "configs/p88_ultrahdr_pinned_decoder_consumption_v1.json",
    }
    actual = {
        key: _expected_hash(path, bindings[key], key) for key, path in paths.items()
    }
    p88_config = _load_object(paths["p88_config_sha256"])
    actual["decoder_executable_sha256"] = _expected_hash(
        decoder_app,
        p88_config["official"]["decoder_executable_sha256"],
        "decoder executable",
    )
    actual["p89_config_sha256"] = _sha256_file(config_path)
    return actual, p88_config


def _probe(config: dict[str, Any]) -> dict[str, Any]:
    count = int(config["arithmetic"]["dense_probe_samples"])
    scalar = np.linspace(0.0, 10000.0, count, dtype=np.float64)
    values = np.repeat(scalar[:, None], 3, axis=1)
    encoded = absolute_rec2020_cdm2_to_pq(values)[:, 0]
    samples = absolute_rec2020_cdm2_to_pq_rgb16(values)[:, 0]
    oracle = _scalar_oracle_samples(values)[:, 0]
    decoded = pq_to_absolute_rec2020_cdm2(
        np.repeat((samples.astype(np.float64) / PQ_RGB16_MAXIMUM)[:, None], 3, axis=1)
    )[:, 0]
    return {
        "decoded_maximum_nits": float(decoded.max()),
        "decoded_minimum_nits": float(decoded.min()),
        "encoded_monotone": bool(np.all(np.diff(encoded) > 0.0)),
        "endpoint_codes_exact": bool(samples[0] == 0 and samples[-1] == 65535),
        "maximum_code_quantization_error": float(
            np.max(np.abs(encoded - samples.astype(np.float64) / PQ_RGB16_MAXIMUM))
        ),
        "sample_count": count,
        "sample_monotone": bool(np.all(np.diff(samples.astype(np.int64)) >= 0)),
        "scalar_oracle_exact": bool(np.array_equal(samples, oracle)),
    }


def run(config_path: Path, decoder_app: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    bindings, p88_config = _validate_bindings(config, config_path, decoder_app)
    fixture_root = ROOT / "tests/fixtures/u1_5c_libultrahdr"
    rows = list(config["fixtures"])
    if reverse:
        rows.reverse()
    records: list[dict[str, Any]] = []
    invalid_atomic = True
    with tempfile.TemporaryDirectory(prefix="p89_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            fixture = fixture_root / row["name"]
            source_bytes = fixture.read_bytes()
            if _sha256_bytes(source_bytes) != row["sha256"]:
                raise ValueError(f"fixture SHA-256 mismatch: {row['name']}")
            local_input = scratch / row["name"]
            decoded_path = scratch / f"{Path(row['name']).stem}.rgba16f"
            output_path = scratch / f"{Path(row['name']).stem}.pq.png"
            local_input.write_bytes(source_bytes)
            completed = _run_decoder(decoder_app, local_input, decoded_path)
            if completed.returncode != 0:
                raise RuntimeError(f"decoder failed: {row['name']}")
            decoded = decoded_path.read_bytes()
            prepared = prepare_ultrahdr_match_view_v1(
                source_asset=source_bytes,
                expected_source_sha256=row["sha256"],
                decoded_rgba16f=decoded,
                width=int(row["width"]),
                height=int(row["height"]),
                decoder_version=ULTRAHDR_DECODER_VERSION,
                producer_profile_id=ULTRAHDR_EXTERNAL_PROFILE_ID,
            )
            png_sha, samples = publish_ultrahdr_match_view_pq_png_v1(
                prepared,
                output_path,
                row_count=64,
            )
            encoded = absolute_rec2020_cdm2_to_pq(prepared.pixels)
            decoded_absolute = pq_to_absolute_rec2020_cdm2(
                samples.astype(np.float64) / PQ_RGB16_MAXIMUM
            )
            sample_sha = _sha256_bytes(samples.tobytes())
            records.append(
                {
                    "decoded_payload_sha256": prepared.decoded_payload_sha256,
                    "fixture_name": row["name"],
                    "fixture_sha256": row["sha256"],
                    "maximum_absolute_roundtrip_error_nits": float(
                        np.max(np.abs(decoded_absolute - prepared.pixels))
                    ),
                    "maximum_code_quantization_error": float(
                        np.max(
                            np.abs(
                                encoded
                                - samples.astype(np.float64) / PQ_RGB16_MAXIMUM
                            )
                        )
                    ),
                    "maximum_input_nits": float(prepared.pixels.max()),
                    "minimum_input_nits": float(prepared.pixels.min()),
                    "png_file_sha256": png_sha,
                    "profile_id": prepared.descriptor.profile_id,
                    "production_pq_rejection": _production_rejection(
                        output_path,
                        "unsupported PNG cICP color/dynamic-range signalling",
                    ),
                    "production_source_rejection": _production_rejection(
                        fixture,
                        "HDR/gain-map reconstruction is not implemented",
                    ),
                    "sample_readback_exact": sha256_rec2100_pq_rgb16_png_samples(
                        output_path,
                        width=int(row["width"]),
                        height=int(row["height"]),
                    )
                    == sample_sha,
                    "sample_sha256": sample_sha,
                    "scalar_oracle_exact": bool(
                        np.array_equal(samples, _scalar_oracle_samples(prepared.pixels))
                    ),
                    "source_unchanged": fixture.read_bytes() == source_bytes,
                }
            )

        invalid_cases = [
            np.full((1, 1, 3), np.nan),
            np.full((1, 1, 3), -0.01),
            np.full((1, 1, 3), 10000.01),
            np.zeros((2, 2), dtype=np.float64),
        ]
        for index, values in enumerate(invalid_cases):
            invalid_path = scratch / f"invalid_{index}.png"
            try:
                from src.preprocess.rec2100_pq_transfer import (
                    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
                )

                save_absolute_rec2020_cdm2_to_pq_rgb16_png(values, invalid_path)
            except ValueError:
                pass
            else:
                invalid_atomic = False
            invalid_atomic = invalid_atomic and not invalid_path.exists()

    records.sort(key=lambda value: value["fixture_name"])
    probe = _probe(config)
    limit = (
        float(config["arithmetic"]["maximum_code_quantization_error"])
        + float(config["arithmetic"]["quantization_epsilon"])
    )
    gates = {
        "bindings_exact": len(bindings) == 6,
        "all_profiles_exact": all(
            row["profile_id"] == MATCH_PROFILE_ABSOLUTE_REC2020 for row in records
        ),
        "all_scalar_oracles_exact": all(row["scalar_oracle_exact"] for row in records),
        "all_code_quantization_errors_bounded": all(
            row["maximum_code_quantization_error"] <= limit for row in records
        ),
        "all_sample_readbacks_exact": all(row["sample_readback_exact"] for row in records),
        "all_sources_unchanged": all(row["source_unchanged"] for row in records),
        "invalid_inputs_atomic": invalid_atomic,
        "probe_endpoint_codes_exact": probe["endpoint_codes_exact"],
        "probe_encoded_monotone": probe["encoded_monotone"],
        "probe_samples_monotone": probe["sample_monotone"],
        "probe_scalar_oracle_exact": probe["scalar_oracle_exact"],
        "production_loader_still_rejects": all(
            "unsupported PNG cICP color/dynamic-range signalling"
            in row["production_pq_rejection"]
            and "HDR/gain-map reconstruction is not implemented"
            in row["production_source_rejection"]
            for row in records
        ),
    }
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "constants": {
            "c1": PQ_C1,
            "c2": PQ_C2,
            "c3": PQ_C3,
            "m1": PQ_M1,
            "m2": PQ_M2,
        },
        "contract_id": config["contract_id"],
        "gates": gates,
        "p88_decoder_commit": p88_config["official"]["commit"],
        "probe": probe,
        "records": records,
        "schema": REPORT_SCHEMA,
        "status": (
            "PASS_PRIVATE_ULTRAHDR_ABSOLUTE_REC2020_PQ_PNG"
            if all(gates.values())
            else "FAIL_CLOSED_ULTRAHDR_ABSOLUTE_REC2020_PQ_PNG"
        ),
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p89_ultrahdr_absolute_rec2020_pq_png_v1.json"),
    )
    parser.add_argument("--decoder-app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config, args.decoder_app.resolve(), reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": _sha256_bytes(payload),
                "stable_evidence_id": report["stable_evidence_id"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
