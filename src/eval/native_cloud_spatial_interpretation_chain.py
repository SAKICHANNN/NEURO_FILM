"""P4FA correct-domain native stochastic-cloud and spatial chain."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_msvc import find_msvc_installation, sha256_file

SOURCES = tuple(
    "native/film_physics/" + name
    for name in (
        "nf_cloud_spatial_interpretation_chain_f32_v1.c",
        "nf_cloud_spatial_interpretation_chain_probe_v1.c",
        "nf_sensitometry_cloud_bridge_f32_v1.c",
        "nf_physical_domains_f32_v1.c",
        "nf_gaussian_rgb_f32_v1.c",
        "nf_bounded_adjacency_f32_v1.c",
        "nf_conditioned_cloud_row_chain_f32_v1.c",
        "nf_conditioned_cloud_row_chain_f32_v2.c",
        "nf_density_conditioned_poisson_u16_v3.c",
        "nf_cloud_spatial_response_f32_v2.c",
        "nf_cloud_attenuation_f32_v1.c",
        "nf_deterministic_log10_f32_v1.c",
    )
)


def _build(root: Path, output: Path, llvm: Path | None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    exe = output / ("p4fa-llvm.exe" if llvm else "p4fa-msvc.exe")
    sources = [str((root / item).resolve()) for item in SOURCES]
    if llvm:
        result = subprocess.run(
            [str(llvm), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2",
             "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", *sources,
             "-o", str(exe), "-Wl,--no-insert-timestamp"],
            capture_output=True, check=False, timeout=120,
        )
    else:
        vcvars = find_msvc_installation() / "Common7/Tools/VsDevCmd.bat"
        batch = output / "build.bat"
        quoted = " ".join(f'"{item}"' for item in sources)
        batch.write_text(
            "@echo off\r\n"
            + f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
            + "if errorlevel 1 exit /b %errorlevel%\r\n"
            + f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX {quoted} '
              f'/link /Brepro /OUT:"{exe}"\r\n',
            encoding="ascii", newline="",
        )
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(batch)], cwd=output,
            capture_output=True, check=False, timeout=120,
        )
    if result.returncode or not exe.is_file():
        raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
    return exe


def evaluate(root: Path, contract_path: Path, output: Path, llvm: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    parent = root / contract["parent"]["path"]
    if (
        sha256_file(parent) != contract["parent"]["sha256"]
        or json.loads(parent.read_text())["decision"]
        != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4FA parent drift")
    builds = {
        "msvc": _build(root, output / "msvc", None),
        "llvm": _build(root, output / "llvm", llvm),
    }
    rows: dict[str, dict] = {}
    for name, exe in builds.items():
        paths = [output / f"{name}-{index}.f32" for index in range(2)]
        runs = []
        for path in paths:
            run = subprocess.run(
                [str(exe), str(path)], capture_output=True,
                check=False, timeout=120,
            )
            if run.returncode:
                raise RuntimeError((run.stdout + run.stderr).decode(errors="replace"))
            runs.append(run)
        values = np.fromfile(paths[0], dtype="<f4")
        rows[name] = {
            "scan_sha256": sha256_file(paths[0]),
            "manual": all(b"manual_exact=1" in run.stdout for run in runs),
            "stages_nonzero": all(b"stages_nonzero=1" in run.stdout for run in runs),
            "atomic": all(b"invalid=" in run.stdout and b"invalid=0" not in run.stdout for run in runs),
            "repeat": sha256_file(paths[0]) == sha256_file(paths[1]),
            "domain": bool(np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1))),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    gates = {
        "manual": all(row["manual"] for row in rows.values()),
        "compiler": rows["msvc"]["scan_sha256"] == rows["llvm"]["scan_sha256"],
        "repeat": all(row["repeat"] for row in rows.values()),
        "atomic": all(row["atomic"] for row in rows.values()),
        "domain": all(row["domain"] for row in rows.values()),
        "stages_nonzero": all(row["stages_nonzero"] for row in rows.values()),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(root / SOURCES[0]),
        "header_sha256": sha256_file(
            root / "native/film_physics/nf_cloud_spatial_interpretation_chain_f32_v1.h"
        ),
        "results": rows,
        "gates": gates,
        "decision": contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4fa_native_cloud_spatial_interpretation_chain.v1",
        "automatic_pass": all(gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
