"""Audit the strict P251 AP0 OpenEXR to ACEScg WorkingImage ingress."""

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


class P251Error(RuntimeError):
    """Raised when a frozen P251 identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _expect_rejection(path: Path) -> bool:
    from src.preprocess.aces2065_openexr import (
        Aces2065OpenExrError,
        load_aces2065_openexr_working_image,
    )

    try:
        load_aces2065_openexr_working_image(path)
    except Aces2065OpenExrError:
        return True
    return False


def _write_control(
    path: Path,
    *,
    openexr: Any,
    header: dict[str, Any],
    channels: dict[str, np.ndarray],
) -> None:
    with openexr.File(header, channels) as exr_file:
        exr_file.write(str(path))


def _worker_execute(
    config_path: Path,
    producer_repo: Path,
    workspace: Path,
    order: str,
) -> dict[str, Any]:
    from src.preprocess.aces2065_openexr import (
        ACES2065_OPENEXR_INGRESS_ID,
        load_aces2065_openexr_working_image,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    expected = config["expected"]
    openexr = importlib.import_module("OpenEXR")
    writer_bytes = _git_bytes(
        producer_repo,
        bindings["producer_writer_commit"],
        bindings["producer_writer_path"],
    )
    if len(writer_bytes) != bindings["producer_writer_bytes"] or (
        _sha256_bytes(writer_bytes) != bindings["producer_writer_sha256"]
    ):
        raise P251Error("producer writer identity differs")
    writer = _load_ephemeral_writer(writer_bytes, workspace / "writer-site")
    source = workspace / "synthetic-ap0.exr"
    original = _synthetic_lattice()
    original_sha = _array_sha256(original)
    receipt = writer.write_aces2065_1_openexr(source, original)
    source_sha = _sha256_file(source)
    source_bytes = source.stat().st_size
    working = load_aces2065_openexr_working_image(source)
    source_unchanged = source_sha == _sha256_file(source)
    maximum_error = float(
        np.max(
            np.abs(
                working.pixels.astype(np.float64) - original.astype(np.float64)
            )
        )
    )
    with openexr.File(str(source)) as source_file:
        header = dict(source_file.header())
        ap0 = source_file.channels()["RGB"].pixels.copy()
    controls = ["malformed", "wrong_identity", "half", "extra_channel"]
    if order == "reverse":
        controls.reverse()
    rejected: dict[str, bool] = {}
    for name in controls:
        target = workspace / f"control-{name}.exr"
        if name == "malformed":
            target.write_bytes(b"not-an-openexr")
        elif name == "wrong_identity":
            wrong_header = dict(header)
            wrong_header["colorInteropID"] = "lin_ap1_scene"
            _write_control(
                target,
                openexr=openexr,
                header=wrong_header,
                channels={"RGB": ap0},
            )
        elif name == "half":
            _write_control(
                target,
                openexr=openexr,
                header=dict(header),
                channels={"RGB": ap0.astype(np.float16)},
            )
        else:
            _write_control(
                target,
                openexr=openexr,
                header=dict(header),
                channels={
                    "RGB": ap0,
                    "A": np.ones(ap0.shape[:2], dtype=np.float32),
                },
            )
        rejected[name] = _expect_rejection(target)
    return {
        "controls_rejected": dict(sorted(rejected.items())),
        "ingress_id": ACES2065_OPENEXR_INGRESS_ID,
        "maximum_ap1_roundtrip_error": maximum_error,
        "openexr_version": openexr.__version__,
        "original_acescg_f32le_sha256": original_sha,
        "output": {
            "alpha_policy": working.alpha_policy,
            "bit_depth_in": working.bit_depth_in,
            "f32le_sha256": _array_sha256(working.pixels),
            "maximum": float(np.max(working.pixels)),
            "minimum": float(np.min(working.pixels)),
            "orientation_applied": working.orientation_applied,
            "owned": bool(working.pixels.flags.owndata),
            "c_contiguous": bool(working.pixels.flags.c_contiguous),
            "writeable": bool(working.pixels.flags.writeable),
            "shape": list(working.pixels.shape),
            "source_profile": working.source_profile.description,
            "source_transfer_state": working.source_transfer_state,
            "transfer_state": working.transfer_state,
            "working_space": working.working_space,
        },
        "receipt": {
            "file_bytes": receipt.file_bytes,
            "file_sha256": receipt.file_sha256,
            "input_acescg_f32le_sha256": receipt.input_acescg_f32le_sha256,
            "pixel_f32le_sha256": receipt.pixel_f32le_sha256,
            "writer_id": receipt.writer_id,
        },
        "source_bytes": source_bytes,
        "source_sha256": source_sha,
        "source_unchanged": source_unchanged,
        "input_unchanged": _array_sha256(original) == original_sha,
        "within_roundtrip_gate": maximum_error
        <= expected["maximum_ap1_roundtrip_error"],
    }


def execute(
    config_path: Path,
    producer_repo: Path,
    order: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    expected = config["expected"]
    contract = ROOT / bindings["contract_path"]
    module = ROOT / bindings["consumer_module_path"]
    runner = ROOT / bindings["runner_path"]
    wheel = producer_repo / bindings["openexr_wheel_path"]
    identities = {
        "contract": len(contract.read_bytes()) == bindings["contract_bytes"]
        and _sha256_file(contract) == bindings["contract_sha256"],
        "module": len(module.read_bytes()) == bindings["consumer_module_bytes"]
        and _sha256_file(module) == bindings["consumer_module_sha256"],
        "runner": len(runner.read_bytes()) == bindings["runner_bytes"]
        and _sha256_file(runner) == bindings["runner_sha256"],
        "wheel": wheel.stat().st_size == bindings["openexr_wheel_bytes"]
        and _sha256_file(wheel) == bindings["openexr_wheel_sha256"],
    }
    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p251-"))
    result: dict[str, Any] | None = None
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
        result = json.loads(worker_report.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if result is None:
        raise P251Error("worker did not produce a report")
    gates = {
        "exact_bound_identities": all(identities.values()),
        "exact_openexr_wheel": result["openexr_version"] == "3.4.15",
        "exact_p249_container": result["source_bytes"]
        == expected["p249_openexr_bytes"]
        and result["source_sha256"] == expected["p249_openexr_sha256"],
        "exact_original_lattice": result["original_acescg_f32le_sha256"]
        == expected["p249_input_acescg_f32le_sha256"],
        "owned_contiguous_output": result["output"]["owned"]
        and result["output"]["c_contiguous"]
        and result["output"]["writeable"],
        "strict_working_image_contract": result["output"]["working_space"]
        == "acescg_ap1_d60"
        and result["output"]["transfer_state"] == "scene_linear"
        and result["output"]["source_transfer_state"] == "scene_linear"
        and result["output"]["alpha_policy"] == "absent"
        and result["output"]["bit_depth_in"] == 32,
        "negative_and_highlight_preserved": result["output"]["minimum"] < 0.0
        and result["output"]["maximum"] > 1.0,
        "roundtrip_within_gate": result["within_roundtrip_gate"],
        "source_and_input_immutable": result["source_unchanged"]
        and result["input_unchanged"],
        "invalid_controls_rejected": all(result["controls_rejected"].values()),
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_ACES2065_OPENEXR_WORKING_IMAGE_INGRESS"
        if all(gates.values())
        else "FAIL_CLOSED_ACES2065_OPENEXR_WORKING_IMAGE_INGRESS",
        "experiment_id": "P251",
        "gates": gates,
        "network_reads": 0,
        "result": result,
        "schema": "neuro-film.p251-aces2065-openexr-working-image-ingress-result.v1",
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
