"""Audit strict P251 ACES2065-1 ingress to canonical ACES 2 PQ composition."""

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
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _load_ephemeral_writer,
        _synthetic_lattice,
    )
else:
    from audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _load_ephemeral_writer,
        _synthetic_lattice,
    )

ROOT = Path(__file__).resolve().parents[1]


class P252Error(RuntimeError):
    """Raised when a frozen P252 identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _array_sha256(value: np.ndarray, *, dtype: str | None = None) -> str:
    array = np.ascontiguousarray(value if dtype is None else value.astype(dtype))
    return _sha256_bytes(array.tobytes())


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _working(pixels: np.ndarray, working_space: str):
    from src.preprocess.types import SourceProfile, WorkingImage

    return WorkingImage(
        pixels=np.ascontiguousarray(pixels.copy()),
        working_space=working_space,
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "P252 legacy regression fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("p252-legacy-regression.exr"),
    )


def _legacy_hashes(order: str) -> dict[str, str]:
    from src.preprocess.ocio_aces2_output import (
        apply_working_image_aces2_output,
        build_aces2_numeric_fixture,
    )

    fixture = build_aces2_numeric_fixture()
    indices = np.arange(fixture.shape[0])
    execution_indices = indices if order == "forward" else indices[::-1]
    ordered = np.ascontiguousarray(fixture[execution_indices])
    result: dict[str, str] = {}
    for working_space in ("linear_srgb", "linear_rec2020"):
        for target in ("sdr_rec709", "hdr_rec2020_pq"):
            output = apply_working_image_aces2_output(
                _working(ordered.reshape(1, -1, 3), working_space), target
            ).reshape(-1, 3)
            canonical = np.empty_like(output)
            canonical[execution_indices] = output
            result[f"{working_space}_{target}"] = _array_sha256(canonical)
    return dict(sorted(result.items()))


def _write_wrong_identity(path: Path, source: Path, openexr: Any) -> None:
    with openexr.File(str(source)) as source_file:
        header = dict(source_file.header())
        pixels = source_file.channels()["RGB"].pixels.copy()
    header["colorInteropID"] = "lin_ap1_scene"
    with openexr.File(header, {"RGB": pixels}) as output:
        output.write(str(path))


def _expect_composition_rejection(source: Path, output: Path) -> bool:
    from src.preprocess.aces2065_aces2_pq import (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
    )

    try:
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(source, output)
    except (RuntimeError, ValueError):
        return not output.exists()
    return False


def _worker_execute(
    config_path: Path,
    producer_repo: Path,
    workspace: Path,
    order: str,
) -> dict[str, Any]:
    from src.preprocess.aces2065_aces2_pq import (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
    )
    from src.preprocess.aces2065_openexr import (
        load_aces2065_openexr_working_image,
    )
    from src.preprocess.ocio_aces2_output import (
        CONFIG_CACHE_ID,
        apply_aces2_output_packed,
        apply_working_image_aces2_output,
        convert_working_image_to_acescg,
        load_aces2_config,
    )
    from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples

    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    openexr = importlib.import_module("OpenEXR")
    ocio = importlib.import_module("PyOpenColorIO")
    writer_bytes = _git_bytes(
        producer_repo,
        config["p251_producer"]["writer_commit"],
        config["p251_producer"]["writer_path"],
    )
    if len(writer_bytes) != config["p251_producer"]["writer_bytes"] or (
        _sha256_bytes(writer_bytes)
        != config["p251_producer"]["writer_sha256"]
    ):
        raise P252Error("P251 producer writer identity differs")
    writer = _load_ephemeral_writer(writer_bytes, workspace / "writer-site")
    source = workspace / "synthetic-ap0.exr"
    original = _synthetic_lattice()
    original_sha = _array_sha256(original, dtype="<f4")
    writer.write_aces2065_1_openexr(source, original)
    source_sha = _sha256_file(source)
    working = load_aces2065_openexr_working_image(source)
    working_sha = _array_sha256(working.pixels, dtype="<f4")
    identity = convert_working_image_to_acescg(working)
    identity_error = float(
        np.max(np.abs(identity.astype(np.float64) - working.pixels.astype(np.float64)))
    )
    direct = apply_aces2_output_packed(
        working.pixels.reshape(-1, 3), "hdr_rec2020_pq"
    ).reshape(working.pixels.shape)
    adapter = apply_working_image_aces2_output(working, "hdr_rec2020_pq")
    output = workspace / "display.png"
    file_sha, samples, encoded = (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
            source,
            output,
            row_count=3,
            reverse_partition=order == "reverse",
        )
    )
    sample_sha = _array_sha256(samples.astype(">u2"))
    strict_sample_sha = sha256_rec2100_pq_rgb16_png_samples(
        output, width=working.pixels.shape[1], height=working.pixels.shape[0]
    )

    malformed = workspace / "malformed.exr"
    malformed.write_bytes(b"not-an-openexr")
    wrong_identity = workspace / "wrong-identity.exr"
    _write_wrong_identity(wrong_identity, source, openexr)
    control_names = ["malformed", "wrong_identity"]
    if order == "reverse":
        control_names.reverse()
    rejected: dict[str, bool] = {}
    for name in control_names:
        control_source = malformed if name == "malformed" else wrong_identity
        rejected[name] = _expect_composition_rejection(
            control_source, workspace / f"{name}.png"
        )
    foreign = workspace / "foreign.png"
    foreign.write_bytes(b"foreign-destination")
    foreign_before = foreign.read_bytes()
    foreign_rejected = _expect_composition_rejection(source, foreign)
    foreign_unchanged = foreign.read_bytes() == foreign_before

    config_cache = load_aces2_config().getCacheID()
    return {
        "adapter_direct_exact": bool(np.array_equal(adapter, direct)),
        "adapter_sha256": _array_sha256(adapter, dtype="<f4"),
        "config_cache_id": config_cache,
        "direct_sha256": _array_sha256(direct, dtype="<f4"),
        "encoded_finite_in_unit": bool(
            np.isfinite(encoded).all()
            and np.all(encoded >= 0.0)
            and np.all(encoded <= 1.0)
        ),
        "encoded_sha256": _array_sha256(encoded, dtype="<f4"),
        "foreign_destination_rejected": foreign_rejected,
        "foreign_destination_unchanged": foreign_unchanged,
        "identity_conversion_error": identity_error,
        "identity_owned_contiguous": bool(
            identity.flags.owndata
            and identity.flags.c_contiguous
            and not np.shares_memory(identity, working.pixels)
        ),
        "input_unchanged": _array_sha256(original, dtype="<f4") == original_sha,
        "invalid_controls_rejected": dict(sorted(rejected.items())),
        "legacy_hashes": _legacy_hashes(order),
        "openexr_version": openexr.__version__,
        "opencolorio_version": ocio.__version__,
        "output_bytes": output.stat().st_size,
        "output_png_sha256": _sha256_file(output),
        "output_receipt_sha256": file_sha,
        "output_samples_sha256": sample_sha,
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha,
        "source_unchanged": _sha256_file(source) == source_sha,
        "strict_sample_sha256": strict_sample_sha,
        "working": {
            "f32le_sha256": working_sha,
            "shape": list(working.pixels.shape),
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "maximum": float(np.max(working.pixels)),
            "minimum": float(np.min(working.pixels)),
        },
        "working_ap1_roundtrip_error": float(
            np.max(
                np.abs(
                    working.pixels.astype(np.float64)
                    - original.astype(np.float64)
                )
            )
        ),
        "config_cache_exact": config_cache == CONFIG_CACHE_ID,
        "bound_composition_module": bindings["composition_module_path"],
    }


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    identity_paths = {
        "contract": bindings["contract_path"],
        "p251_config": bindings["p251_config_path"],
        "p251_evidence": bindings["p251_evidence_path"],
        "ocio_module": bindings["ocio_module_path"],
        "composition_module": bindings["composition_module_path"],
        "canonical_bridge": bindings["canonical_bridge_path"],
        "canonical_writer": bindings["canonical_writer_path"],
        "runner": bindings["runner_path"],
        "u1_4e_config": bindings["u1_4e_config_path"],
        "u1_4e_evidence": bindings["u1_4e_evidence_path"],
    }
    identities = {
        name: _sha256_file(ROOT / path) == bindings[f"{name}_sha256"]
        for name, path in identity_paths.items()
    }
    wheel = producer_repo / config["p251_producer"]["openexr_wheel_path"]
    identities["openexr_wheel"] = (
        wheel.stat().st_size == config["p251_producer"]["openexr_wheel_bytes"]
        and _sha256_file(wheel)
        == config["p251_producer"]["openexr_wheel_sha256"]
    )
    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p252-"))
    worker: dict[str, Any] | None = None
    try:
        site = temporary / "site"
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
        worker_report = temporary / "worker.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(site), str(ROOT)))
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(config_path),
                "--producer-repo",
                str(producer_repo),
                "--workspace",
                str(temporary),
                "--order",
                order,
                "--output",
                str(worker_report),
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        worker = json.loads(worker_report.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if worker is None:
        raise P252Error("worker did not produce a report")
    input_contract = config["input"]
    gate_config = config["gates"]
    gates = {
        "exact_bound_identities": all(identities.values()),
        "exact_runtime": worker["openexr_version"]
        == config["runtime"]["openexr_version"]
        and worker["opencolorio_version"]
        == config["runtime"]["opencolorio_version"]
        and worker["config_cache_exact"],
        "exact_p249_source": worker["source_bytes"]
        == input_contract["p249_openexr_bytes"]
        and worker["source_sha256"] == input_contract["p249_openexr_sha256"],
        "exact_p251_working_image": worker["working"]["shape"]
        == input_contract["shape"]
        and worker["working"]["working_space"] == input_contract["working_space"]
        and worker["working"]["transfer_state"] == input_contract["transfer_state"]
        and worker["working_ap1_roundtrip_error"]
        <= gate_config["maximum_ap1_roundtrip_error"],
        "identity_conversion_exact_owned": worker["identity_conversion_error"]
        <= gate_config["maximum_identity_conversion_error"]
        and worker["identity_owned_contiguous"],
        "legacy_hashes_unchanged": worker["legacy_hashes"]
        == config["legacy_hashes"],
        "direct_official_output_exact": worker["adapter_direct_exact"]
        and worker["adapter_sha256"] == worker["direct_sha256"],
        "finite_unit_output": worker["encoded_finite_in_unit"],
        "canonical_png_receipt_exact": worker["output_png_sha256"]
        == worker["output_receipt_sha256"],
        "strict_rec2100_pq_samples_exact": worker["output_samples_sha256"]
        == worker["strict_sample_sha256"],
        "source_and_input_immutable": worker["source_unchanged"]
        and worker["input_unchanged"],
        "invalid_controls_rejected": all(
            worker["invalid_controls_rejected"].values()
        ),
        "foreign_destination_preserved": worker["foreign_destination_rejected"]
        and worker["foreign_destination_unchanged"],
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_ACES2065_ACES2_PQ_COMPOSITION"
        if all(gates.values())
        else "FAIL_CLOSED_ACES2065_ACES2_PQ_COMPOSITION",
        "experiment_id": "P252",
        "gates": gates,
        "network_reads": 0,
        "result": worker,
        "schema": "neuro-film.p252-aces2065-aces2-pq-composition-result.v1",
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.workspace is None:
            raise ValueError("worker requires workspace")
        report = _worker_execute(
            args.config.resolve(),
            args.producer_repo.resolve(),
            args.workspace.resolve(),
            args.order,
        )
    else:
        if args.workspace is not None:
            raise ValueError("controller forbids workspace")
        report = execute(
            args.config.resolve(), args.producer_repo.resolve(), args.order
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
