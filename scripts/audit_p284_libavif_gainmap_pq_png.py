#!/usr/bin/env python3
"""Audit one-way P283/P87 to P89 PQ PNG publication."""

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

from scripts.audit_p282_libavif_gainmap_windows_runtime import (
    build_base_command,
    build_info_command,
    build_tonemap_command,
)
from scripts.audit_p283_libavif_gainmap_p87_bridge import (
    _decode_rgb16,
    _run,
    parse_chosen_cicp,
)
from src.color_match.libavif_gainmap_ingress import (
    prepare_libavif_gainmap_match_view_v1,
)
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.pipeline import load_working_image
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.rec2100_pq_transfer import (
    PQ_RGB16_MAXIMUM,
    absolute_rec2020_cdm2_to_pq,
    pq_to_absolute_rec2020_cdm2,
    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
)

REPORT_SCHEMA = "neuro-film.p284-libavif-gainmap-pq-png-result.v1"


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


def _validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "p282_evidence",
        "p283_config",
        "p283_evidence",
        "p89_evidence",
        "u1_4g_evidence",
    ):
        path = ROOT / config["bindings"][name]
        actual = _sha256_file(path)
        if actual != config["bindings"][f"{name}_sha256"]:
            raise ValueError(f"binding mismatch: {name}")
        result[f"{name}_sha256"] = actual
    source_bindings = {
        "pq_transfer_sha256": ROOT / "src/preprocess/rec2100_pq_transfer.py",
        "png_stream_sha256": ROOT / "src/preprocess/png_stream.py",
    }
    for name, path in source_bindings.items():
        actual = _sha256_file(path)
        if actual != config["bindings"][name]:
            raise ValueError(f"source binding mismatch: {name}")
        result[name] = actual
    p282 = _load_object(ROOT / config["bindings"]["p282_evidence"])
    p283 = _load_object(ROOT / config["bindings"]["p283_evidence"])
    p89 = _load_object(ROOT / config["bindings"]["p89_evidence"])
    if p282["status"] != "PASS_PRIVATE_LIBAVIF_GAINMAP_WINDOWS_RUNTIME":
        raise ValueError("P282 parent status mismatch")
    if p283["status"] != "FAIL_CLOSED_LIBAVIF_GAINMAP_P87_BRIDGE":
        raise ValueError("P283 parent status mismatch")
    if p89["status"] != "PASS_PRIVATE_ULTRAHDR_ABSOLUTE_REC2020_PQ_PNG":
        raise ValueError("P89 parent status mismatch")
    return result


def _production_rejection(path: Path) -> bool:
    try:
        load_working_image(path)
    except ValueError:
        return True
    return False


def invalid_publication_controls(scratch: Path) -> dict[str, bool]:
    cases = {
        "nonfinite": np.full((1, 1, 3), np.nan),
        "negative": np.full((1, 1, 3), -0.01),
        "above_maximum": np.full((1, 1, 3), 10000.01),
        "wrong_shape": np.zeros((2, 2), dtype=np.float64),
    }
    results: dict[str, bool] = {}
    for name, values in cases.items():
        output = scratch / f"invalid_{name}.png"
        try:
            save_absolute_rec2020_cdm2_to_pq_rgb16_png(values, output)
        except ValueError:
            results[name] = not output.exists()
        else:
            results[name] = False
    return results


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    bindings = _validate_bindings(config)
    p283_config = _load_object(ROOT / config["bindings"]["p283_config"])
    runtime_root = ROOT / p283_config["runtime"]["root"]
    avifdec = runtime_root / "avifdec.exe"
    gainmaputil = runtime_root / "avifgainmaputil.exe"
    source_root = ROOT / p283_config["source_root"]
    rows = list(p283_config["rows"])
    if reverse:
        rows.reverse()
    source_before = {
        row["name"]: _sha256_file(source_root / row["name"]) for row in rows
    }
    runtime_before = {
        "avifdec.exe": _sha256_file(avifdec),
        "avifgainmaputil.exe": _sha256_file(gainmaputil),
    }
    records: list[dict[str, Any]] = []
    create_only_atomic = True
    with tempfile.TemporaryDirectory(prefix="p284_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            source = source_root / row["name"]
            source_bytes = source.read_bytes()
            if _sha256_bytes(source_bytes) != row["source_sha256"]:
                raise ValueError(f"source identity mismatch: {row['name']}")
            info = _run(build_info_command(avifdec, source))
            if info.returncode != 0:
                raise RuntimeError(f"avifdec info failed: {row['name']}")
            cicp = parse_chosen_cicp(info.stdout, row["higher_role"])
            decoded_path = scratch / f"{row['name']}.higher.png"
            decode_command = (
                build_base_command(avifdec, source, decoded_path)
                if row["higher_role"] == "base"
                else build_tonemap_command(
                    gainmaputil,
                    source,
                    decoded_path,
                    float(row["alternate_headroom"]),
                )
            )
            decoded_process = _run(decode_command)
            if decoded_process.returncode != 0 or not decoded_path.is_file():
                raise RuntimeError(f"higher rendition failed: {row['name']}")
            rgb16, bgr_sha = _decode_rgb16(decoded_path)
            if bgr_sha != row["p282_bgr16_sample_sha256"]:
                raise ValueError(f"P282 sample identity mismatch: {row['name']}")
            input_before = rgb16.copy()
            prepared = prepare_libavif_gainmap_match_view_v1(
                source_asset=source_bytes,
                expected_source_sha256=row["source_sha256"],
                encoded_rgb16=rgb16,
                decoder_version=p283_config["runtime"]["decoder_version"],
                source_color_primaries=cicp[0],
                source_transfer_characteristics=cicp[1],
                source_full_range=True,
                higher_rendition_role=row["higher_role"],
            )
            output = scratch / f"{row['name']}.pq.png"
            repeat = scratch / f"{row['name']}.repeat.pq.png"
            png_sha, samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
                prepared.pixels,
                output,
                row_count=int(config["publication"]["row_count"]),
            )
            repeat_sha, repeat_samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
                prepared.pixels,
                repeat,
                row_count=int(config["publication"]["row_count"]),
            )
            try:
                save_absolute_rec2020_cdm2_to_pq_rgb16_png(prepared.pixels, output)
            except FileExistsError:
                pass
            else:
                create_only_atomic = False
            encoded = absolute_rec2020_cdm2_to_pq(prepared.pixels)
            decoded_absolute = pq_to_absolute_rec2020_cdm2(
                samples.astype(np.float64) / PQ_RGB16_MAXIMUM
            )
            sample_sha = _sha256_bytes(samples.tobytes())
            records.append(
                {
                    "absolute_light_maximum_error_nits": float(
                        np.max(np.abs(decoded_absolute - prepared.pixels))
                    ),
                    "chosen_cicp": list(cicp),
                    "code_quantization_maximum_error": float(
                        np.max(
                            np.abs(
                                encoded - samples.astype(np.float64) / PQ_RGB16_MAXIMUM
                            )
                        )
                    ),
                    "higher_role": row["higher_role"],
                    "input_unchanged": bool(np.array_equal(rgb16, input_before)),
                    "maximum_nits": float(np.max(prepared.pixels)),
                    "minimum_nits": float(np.min(prepared.pixels)),
                    "name": row["name"],
                    "png_file_sha256": png_sha,
                    "png_repeat_exact": bool(
                        png_sha == repeat_sha
                        and np.array_equal(samples, repeat_samples)
                        and output.read_bytes() == repeat.read_bytes()
                    ),
                    "production_output_rejection": _production_rejection(output),
                    "sample_readback_exact": sha256_rec2100_pq_rgb16_png_samples(
                        output, width=400, height=300
                    )
                    == sample_sha,
                    "sample_sha256": sample_sha,
                    "source_sha256": row["source_sha256"],
                }
            )
        invalid_controls = invalid_publication_controls(scratch)

    records.sort(key=lambda value: value["name"])
    source_after = {name: _sha256_file(source_root / name) for name in source_before}
    runtime_after = {
        "avifdec.exe": _sha256_file(avifdec),
        "avifgainmaputil.exe": _sha256_file(gainmaputil),
    }
    code_limit = float(
        config["publication"]["maximum_code_quantization_error"]
    ) + float(config["publication"]["quantization_epsilon"])
    gates = {
        "all_absolute_light_errors_bounded": all(
            row["absolute_light_maximum_error_nits"]
            <= float(config["gates"]["maximum_absolute_light_error_nits"])
            for row in records
        ),
        "all_cicp_exact": all(row["chosen_cicp"] == [1, 16, 6] for row in records),
        "all_code_quantization_errors_bounded": all(
            row["code_quantization_maximum_error"] <= code_limit for row in records
        ),
        "all_inputs_unchanged": all(row["input_unchanged"] for row in records),
        "all_png_repeats_exact": all(row["png_repeat_exact"] for row in records),
        "all_production_outputs_reject": all(
            row["production_output_rejection"] for row in records
        ),
        "all_sample_readbacks_exact": all(
            row["sample_readback_exact"] for row in records
        ),
        "all_rows_present": len(records) == int(config["gates"]["required_row_count"]),
        "create_only_atomic": create_only_atomic,
        "invalid_inputs_atomic": all(invalid_controls.values()),
        "runtime_immutable": runtime_before == runtime_after,
        "source_immutable": source_before == source_after,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha256_file(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "invalid_controls": invalid_controls,
        "records": records,
        "runtime_network_requests": 0,
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_LIBAVIF_GAINMAP_PQ_PNG"
        if passed
        else "FAIL_CLOSED_LIBAVIF_GAINMAP_PQ_PNG",
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p284_libavif_gainmap_pq_png_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
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
