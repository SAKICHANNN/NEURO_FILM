"""Audit unchanged P251 strict ingress on the exact P255 24MP AP0 master."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

if __package__:
    from scripts.audit_p248_acescg_openexr_scanline_streaming import (
        _canonical_bytes,
        _generate_block,
        _install_openexr_wheel,
        _process_tree_rss,
        _sha256_file,
    )
    from scripts.audit_p255_aces2065_openexr_scanline_streaming import (
        _build,
        _run_native,
    )
else:
    from audit_p248_acescg_openexr_scanline_streaming import (
        _canonical_bytes,
        _generate_block,
        _install_openexr_wheel,
        _process_tree_rss,
        _sha256_file,
    )
    from audit_p255_aces2065_openexr_scanline_streaming import _build, _run_native

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neuro-film.p256-aces2065-openexr-24mp-ingress-resource-result.v1"


class P256Error(RuntimeError):
    """Raised when a frozen P256 identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _worker(config_path: Path, source: Path, site: Path) -> dict[str, Any]:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    from src.preprocess.aces2065_openexr import (
        ACES2065_OPENEXR_INGRESS_ID,
        load_aces2065_openexr_working_image,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = config["input"]
    expected = config["expected_output"]
    before = _sha256_file(source)
    working = load_aces2065_openexr_working_image(source)
    pixels = working.pixels
    digest = hashlib.sha256()
    maximum_error = 0.0
    new_boundary = 0
    block = int(frozen["generation_row_block"])
    for y0 in range(0, int(frozen["height"]), block):
        rows = min(block, int(frozen["height"]) - y0)
        reference = _generate_block(
            y0, rows, int(frozen["width"]), int(frozen["coordinate_period"])
        )
        observed = np.ascontiguousarray(pixels[y0 : y0 + rows])
        digest.update(observed.tobytes())
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(observed.astype(np.float64) - reference))),
        )
        new_boundary += int(
            np.count_nonzero(
                ((observed == np.float32(0.0)) | (observed == np.float32(1.0)))
                & ~(
                    (reference == np.float32(0.0))
                    | (reference == np.float32(1.0))
                )
            )
        )
    result = {
        "alpha_policy": working.alpha_policy,
        "bit_depth_in": working.bit_depth_in,
        "c_contiguous": bool(pixels.flags.c_contiguous),
        "finite": bool(np.isfinite(pixels).all()),
        "ingress_id": ACES2065_OPENEXR_INGRESS_ID,
        "maximum": float(np.max(pixels)),
        "maximum_ap1_roundtrip_error": maximum_error,
        "minimum": float(np.min(pixels)),
        "new_boundary_count": new_boundary,
        "orientation_applied": working.orientation_applied,
        "output_f32le_sha256": digest.hexdigest(),
        "owned": bool(pixels.flags.owndata),
        "shape": list(pixels.shape),
        "source_immutable": before == _sha256_file(source),
        "source_profile": working.source_profile.description,
        "source_transfer_state": working.source_transfer_state,
        "transfer_state": working.transfer_state,
        "working_space": working.working_space,
        "writeable": bool(pixels.flags.writeable),
        "within_numeric_gate": maximum_error
        <= float(expected["maximum_ap1_roundtrip_error"]),
    }
    return result


def _fresh_reader(
    config_path: Path,
    source: Path,
    site: Path,
    workspace: Path,
    label: str,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    report = workspace / f"{label}.json"
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(config_path),
            "--source",
            str(source),
            "--site",
            str(site),
            "--output",
            str(report),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process = psutil.Process(child.pid)
    peak = 0
    samples = 0
    started = time.perf_counter()
    timed_out = False
    while child.poll() is None:
        peak = max(peak, _process_tree_rss(process))
        samples += 1
        if time.perf_counter() - started > timeout:
            timed_out = True
            child.kill()
            break
        time.sleep(interval)
    stdout, stderr = child.communicate()
    wall = time.perf_counter() - started
    if timed_out or child.returncode != 0 or not report.is_file():
        raise P256Error(
            f"reader {label} failed (timeout={timed_out}, code={child.returncode}): "
            + stderr.decode("utf-8", errors="replace")
            + stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_text(encoding="utf-8"))
    report.unlink()
    value["resource"] = {
        "peak_process_tree_rss_bytes": peak,
        "rss_samples": samples,
        "sampling_interval_seconds": interval,
        "wall_seconds": wall,
    }
    return value


def execute(config_path: Path, producer_repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    frozen = config["input"]
    limits = config["gates"]
    names = (
        "contract",
        "p251_module",
        "p251_config",
        "p251_evidence",
        "p255_config",
        "p255_evidence",
        "p255_native_source",
        "p255_runner",
        "runner",
    )
    identities = {
        name: _identity(ROOT / bindings[f"{name}_path"]) for name in names
    }
    identities["config_sha256"] = _sha256_file(config_path)
    identity_exact = all(
        identities[name]["bytes"] == bindings[f"{name}_bytes"]
        and identities[name]["sha256"] == bindings[f"{name}_sha256"]
        for name in names
    )
    wheel = producer_repo / bindings["producer_openexr_wheel_path"]
    wheel_exact = (
        wheel.stat().st_size == bindings["producer_openexr_wheel_bytes"]
        and _sha256_file(wheel) == bindings["producer_openexr_wheel_sha256"]
    )
    workspace = Path(tempfile.mkdtemp(prefix="neuro-film-p256-"))
    try:
        p255_config = json.loads(
            (ROOT / bindings["p255_config_path"]).read_text(encoding="utf-8")
        )
        binary, build = _build(p255_config, workspace)
        site = workspace / "site"
        _install_openexr_wheel(wheel, site)
        source = workspace / "p255-ap0-24mp.exr"
        completed = _run_native(
            binary,
            source,
            width=int(frozen["width"]),
            height=int(frozen["height"]),
            row_block=int(frozen["write_row_block"]),
            period=int(frozen["coordinate_period"]),
        )
        if completed.returncode != 0:
            raise P256Error(completed.stderr.decode("utf-8", errors="replace"))
        source_exact = (
            source.stat().st_size == int(frozen["openexr_bytes"])
            and _sha256_file(source) == frozen["openexr_sha256"]
        )
        if not source_exact:
            raise P256Error("P255 24MP source identity differs before reader launch")
        interval = float(limits["maximum_sampling_interval_seconds"])
        timeout = float(limits["maximum_reader_wall_seconds"])
        readers = [
            _fresh_reader(
                config_path, source, site, workspace, label, interval, timeout
            )
            for label in ("reader-a", "reader-b")
        ]
        source_immutable = _sha256_file(source) == frozen["openexr_sha256"]
        source.unlink()
    finally:
        shutil.rmtree(workspace, ignore_errors=False)

    expected = config["expected_output"]
    stable_output = len({row["output_f32le_sha256"] for row in readers}) == 1
    numeric = all(row["within_numeric_gate"] for row in readers)
    contract = all(
        row["shape"] == [int(frozen["height"]), int(frozen["width"]), 3]
        and row["working_space"] == expected["working_space"]
        and row["transfer_state"] == expected["transfer_state"]
        and row["source_transfer_state"] == expected["source_transfer_state"]
        and row["alpha_policy"] == expected["alpha_policy"]
        and row["bit_depth_in"] == expected["bit_depth_in"]
        and row["owned"]
        and row["c_contiguous"]
        and row["writeable"]
        for row in readers
    )
    range_pass = all(
        row["finite"]
        and row["minimum"] < 0.0
        and row["maximum"] > 1.0
        and row["new_boundary_count"] == 0
        for row in readers
    )
    resources = all(
        row["resource"]["peak_process_tree_rss_bytes"]
        <= limits["maximum_reader_process_tree_rss_bytes"]
        and row["resource"]["wall_seconds"] <= limits["maximum_reader_wall_seconds"]
        for row in readers
    )
    gates = {
        "identities": identity_exact and wheel_exact,
        "numeric": numeric,
        "output-repeat": stable_output,
        "range-and-boundary": range_pass,
        "resource": resources,
        "source-identity-and-immutability": source_exact
        and source_immutable
        and all(row["source_immutable"] for row in readers),
        "strict-working-image-contract": contract,
        "workspace-clean": not workspace.exists(),
    }
    stable_scientific = {
        "gates": gates,
        "maximum_ap1_roundtrip_error": max(
            row["maximum_ap1_roundtrip_error"] for row in readers
        ),
        "new_boundary_count": max(row["new_boundary_count"] for row in readers),
        "output_f32le_sha256": readers[0]["output_f32le_sha256"],
        "source_bytes": int(frozen["openexr_bytes"]),
        "source_sha256": frozen["openexr_sha256"],
        "stable_reader_facts": {
            key: readers[0][key]
            for key in (
                "alpha_policy",
                "bit_depth_in",
                "finite",
                "ingress_id",
                "maximum",
                "minimum",
                "orientation_applied",
                "shape",
                "source_profile",
                "source_transfer_state",
                "transfer_state",
                "working_space",
            )
        },
        "network_requests": 0,
        "external_or_project_pixel_reads": 0,
        "owned_workspace_residue": 0,
    }
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_PRIVATE_ACES2065_OPENEXR_24MP_INGRESS_RESOURCE"
            if all(gates.values())
            else "FAIL_CLOSED_ACES2065_OPENEXR_24MP_INGRESS_RESOURCE"
        ),
        "identities": identities,
        "build": build,
        "readers": readers,
        "stable_scientific": stable_scientific,
        "stable_identity": _sha256_bytes(_canonical_bytes(stable_scientific)),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.source is None or args.site is None:
            raise P256Error("worker requires --source and --site")
        value = _worker(args.config.resolve(), args.source.resolve(), args.site.resolve())
    else:
        if args.producer_repo is None:
            raise P256Error("controller requires --producer-repo")
        value = execute(args.config.resolve(), args.producer_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(value))
    return 0 if value.get("status", "PASS").startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
