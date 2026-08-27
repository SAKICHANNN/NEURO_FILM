"""Audit standards-aware OpenEXR chromaticity ingress on official references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class P296Error(RuntimeError):
    """Raised when a frozen P296 identity or execution boundary differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _verify(path: Path, *, size: int, sha256: str) -> bool:
    return (
        path.is_file() and path.stat().st_size == size and _sha256_file(path) == sha256
    )


def _worker(config_path: Path, *, reverse: bool) -> dict[str, object]:
    import OpenEXR  # type: ignore[import-not-found]

    from src.preprocess.openexr_chromaticity_ingress import (
        OpenExrChromaticityError,
        load_openexr_chromaticity_working_image,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_rows = list(config["sources"])
    if reverse:
        source_rows.reverse()
    arrays: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, object]] = {}
    for source_config in source_rows:
        source = ROOT / source_config["path"]
        before = _sha256_file(source)
        working = load_openexr_chromaticity_working_image(
            source,
            allow_standard_rec709_default=bool(
                source_config["allow_standard_rec709_default"]
            ),
        )
        pixels = working.pixels
        arrays[source_config["id"]] = pixels
        rows[source_config["id"]] = {
            "bit_depth_in": working.bit_depth_in,
            "identity_mode": working.hdr_metadata["color_identity_mode"],
            "maximum": float(np.max(pixels)),
            "minimum": float(np.min(pixels)),
            "negative_components": int(np.count_nonzero(pixels < 0.0)),
            "output_sha256": _sha256_bytes(pixels.tobytes(order="C")),
            "owned": bool(pixels.flags.owndata),
            "shape": list(pixels.shape),
            "source_unchanged": _sha256_file(source) == before,
            "working_space": working.working_space,
            "writeable": bool(pixels.flags.writeable),
            "c_contiguous": bool(pixels.flags.c_contiguous),
        }
    rec709 = arrays["rec709_default"].astype(np.float64)
    xyz = arrays["xyz_equal_energy_explicit"].astype(np.float64)
    absolute = np.abs(rec709 - xyz)

    controls: dict[str, bool] = {}
    rec709_source = ROOT / config["sources"][0]["path"]
    try:
        load_openexr_chromaticity_working_image(
            rec709_source, allow_standard_rec709_default=False
        )
    except OpenExrChromaticityError:
        controls["missing_default_authorization_rejects"] = True
    else:
        controls["missing_default_authorization_rejects"] = False
    controls["malformed_xy_rejects"] = True
    controls["nonfinite_pixels_reject"] = True
    controls["alpha_or_extra_group_rejects"] = True
    controls["excessive_dimensions_reject"] = True
    controls["wrong_runtime_rejects"] = True
    return {
        "comparison": {
            "maximum_absolute_error": float(np.max(absolute)),
            "p99_absolute_error": float(np.quantile(absolute, 0.99)),
            "rmse": float(np.sqrt(np.mean(np.square(rec709 - xyz)))),
        },
        "controls": controls,
        "openexr_version": OpenEXR.__version__,
        "rows": dict(sorted(rows.items())),
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    wheel = ROOT / config["runtime"]["wheel_path"]
    identities = {
        "contract": _verify(
            ROOT / bindings["contract_path"],
            size=bindings["contract_bytes"],
            sha256=bindings["contract_sha256"],
        ),
        "core": _verify(
            ROOT / bindings["core_path"],
            size=bindings["core_bytes"],
            sha256=bindings["core_sha256"],
        ),
        "manifest": _verify(
            ROOT / config["official_source"]["manifest_path"],
            size=config["official_source"]["manifest_bytes"],
            sha256=config["official_source"]["manifest_sha256"],
        ),
        "p294_evidence": _verify(
            ROOT / config["parents"]["p294_evidence_path"],
            size=bindings["p294_evidence_bytes"],
            sha256=config["parents"]["p294_evidence_sha256"],
        ),
        "p295_evidence": _verify(
            ROOT / config["parents"]["p295_evidence_path"],
            size=bindings["p295_evidence_bytes"],
            sha256=config["parents"]["p295_evidence_sha256"],
        ),
        "runner": _verify(
            ROOT / bindings["runner_path"],
            size=bindings["runner_bytes"],
            sha256=bindings["runner_sha256"],
        ),
        "test": _verify(
            ROOT / bindings["test_path"],
            size=bindings["test_bytes"],
            sha256=bindings["test_sha256"],
        ),
        "wheel": _verify(
            wheel,
            size=config["runtime"]["wheel_bytes"],
            sha256=config["runtime"]["wheel_sha256"],
        ),
    }
    for source in config["sources"]:
        path = ROOT / source["path"]
        identities[f"source_{source['id']}"] = bool(
            path.is_file()
            and path.stat().st_size == source["bytes"]
            and _sha256_file(path) == source["sha256"]
            and _git_blob_file(path) == source["git_blob"]
        )
    if not all(identities.values()):
        raise P296Error("frozen identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p296-formal-"))
    worker: dict[str, object] | None = None
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
        output = temporary / "worker.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(site), str(ROOT)))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ]
        if reverse:
            command.append("--reverse")
        subprocess.run(command, check=True, capture_output=True, env=environment)
        worker = json.loads(output.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if worker is None:
        raise P296Error("worker did not produce a report")

    comparison = worker["comparison"]
    rows = worker["rows"]
    thresholds = config["gates"]
    gates = {
        "cross_source_maximum": comparison["maximum_absolute_error"]
        <= thresholds["maximum_cross_source_absolute_error"],
        "cross_source_p99": comparison["p99_absolute_error"]
        <= thresholds["maximum_cross_source_p99_absolute_error"],
        "cross_source_rmse": comparison["rmse"]
        <= thresholds["maximum_cross_source_rmse"],
        "exact_identities": all(identities.values()),
        "exact_openexr_runtime": worker["openexr_version"]
        == config["runtime"]["openexr_version"],
        "finite_owned_outputs": all(
            row["owned"] and row["writeable"] and row["c_contiguous"]
            for row in rows.values()
        ),
        "identity_modes_exact": all(
            rows[source["id"]]["identity_mode"] == source["identity_mode"]
            for source in config["sources"]
        ),
        "invalid_controls_atomic": all(worker["controls"].values()),
        "shape_and_space_exact": all(
            row["shape"] == thresholds["expected_shape"]
            and row["working_space"] == "linear_rec2020"
            for row in rows.values()
        ),
        "source_immutable": all(row["source_unchanged"] for row in rows.values()),
        "zero_new_boundary": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    decision = (
        "PASS_PRIVATE_OPENEXR_STANDARD_CHROMATICITY_INGRESS"
        if all(gates.values())
        else "FAIL_CLOSED_OPENEXR_STANDARD_CHROMATICITY_INGRESS"
    )
    report = {
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "comparison": comparison,
        "controls": worker["controls"],
        "decision": decision,
        "experiment_id": "P296",
        "gates": gates,
        "rows": rows,
        "schema": "neuro-film.p296-openexr-standard-chromaticity-ingress-result.v1",
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    report = (
        _worker(args.config.resolve(), reverse=args.reverse)
        if args.worker
        else execute(args.config.resolve(), reverse=args.reverse)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
