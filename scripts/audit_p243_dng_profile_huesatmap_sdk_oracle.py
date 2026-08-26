"""Build Adobe's SDK reference oracle and score the frozen P243 arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_profile_huesatmap import apply_profile_huesatmap
from src.preprocess.dng_profile_huesatmap_audit import parse_profile_huesatmap_exif


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _tool_paths() -> tuple[Path, Path, Path]:
    base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "18" / "BuildTools"
    return (
        base / "Common7" / "Tools" / "VsDevCmd.bat",
        base / "MSBuild" / "Current" / "Bin" / "MSBuild.exe",
        base / "VC" / "Tools" / "MSVC" / "14.50.35717" / "bin" / "Hostx64" / "x64" / "cl.exe",
    )


def _build_oracle(config: dict[str, Any], root: Path) -> Path:
    authority = config["authority"]
    sdk_zip = ROOT / authority["sdk_zip_path"]
    with zipfile.ZipFile(sdk_zip) as archive:
        archive.extractall(root)
    project = root / authority["sdk_project_member"]
    source_root = project.parents[3] / "source"
    oracle = ROOT / "tests/native/p243_dng_huesatmap_sdk_oracle.cpp"
    shutil.copyfile(oracle, source_root / oracle.name)
    text = project.read_text(encoding="utf-8-sig").replace(
        r"..\..\..\source\dng_validate.cpp", rf"..\..\..\source\{oracle.name}"
    )
    project.write_text(text, encoding="utf-8")
    dev, _, _ = _tool_paths()
    solution = root / authority["sdk_solution_member"]
    build_script = root / "build_oracle.cmd"
    build_script.write_text(
        f'@call "{dev}" -arch=x64 -host_arch=x64 >nul\n'
        f'@msbuild "{solution}" /m /nologo /v:quiet '
        f'/p:Configuration="Validate Release" /p:Platform=x64 /p:PlatformToolset=v145\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["cmd", "/d", "/c", str(build_script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"SDK build failed: {completed.stderr[-2000:]}")
    executable = root / "dng_sdk_1_7_1/dng_sdk/targets/win/release64_x64/dng_validate.exe"
    if not executable.is_file():
        raise RuntimeError("SDK oracle executable missing")
    return executable


def _probes(levels: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, levels, dtype=np.float32)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _oracle_apply(executable: Path, table: np.ndarray, rgb: np.ndarray, root: Path) -> np.ndarray:
    v, h, s, _ = table.shape
    source = root / "input.bin"
    target = root / "output.bin"
    source.write_bytes(b"P243HS01" + struct.pack("<4I", h, s, v, len(rgb)) + table.astype("<f4").tobytes() + rgb.astype("<f4").tobytes())
    subprocess.run([str(executable), str(source), str(target)], check=True, timeout=60)
    payload = target.read_bytes()
    if payload[:8] != b"P243HO01" or struct.unpack_from("<I", payload, 8)[0] != len(rgb):
        raise RuntimeError("invalid SDK oracle output")
    return np.frombuffer(payload, dtype="<f4", offset=12).reshape(-1, 3).copy()


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authority = config["authority"]
    sdk_zip = ROOT / authority["sdk_zip_path"]
    dev, msbuild, compiler = _tool_paths()
    if _sha(sdk_zip) != authority["sdk_zip_sha256"] or _sha(msbuild) != authority["msbuild_sha256"] or _sha(compiler) != authority["cl_sha256"] or not dev.is_file():
        raise ValueError("authority or toolchain identity mismatch")
    probes = _probes(int(config["probe"]["levels"]))
    rows = list(config["source_rows"])
    if reverse:
        rows.reverse()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p243_", dir=ROOT / "tmp") as temp:
        build_root = Path(temp)
        executable = _build_oracle(config, build_root)
        for row in rows:
            exif = ROOT / row["exif_path"]
            if _sha(exif) != row["exif_sha256"]:
                raise ValueError("EXIF identity mismatch")
            parsed = parse_profile_huesatmap_exif(exif.read_text(encoding="utf-8"))
            for name, table in (("data1", parsed.data1), ("data2", parsed.data2)):
                candidate = apply_profile_huesatmap(probes, table)
                oracle = _oracle_apply(executable, table, probes, build_root)
                error = np.abs(candidate.astype(np.float64) - oracle.astype(np.float64))
                results.append({"id": row["id"], "table": name, "max_abs_error": float(error.max()), "rmse": float(np.sqrt(np.mean(error**2))), "output_min": float(candidate.min()), "output_max": float(candidate.max()), "finite": bool(np.isfinite(candidate).all())})
        identity = np.ones((1, 6, 6, 3), dtype=np.float32)
        identity[..., 0] = 0.0
        identity_oracle = _oracle_apply(executable, identity, probes, build_root)
        identity_error = float(np.max(np.abs(apply_profile_huesatmap(probes, identity) - identity_oracle)))
        executable_sha = _sha(executable)
        executable_bytes = executable.stat().st_size
    limits = config["gates"]
    gates = {
        "all_outputs_finite": all(row["finite"] for row in results),
        "max_abs_error": max(row["max_abs_error"] for row in results) <= limits["max_abs_error_at_most"],
        "rmse": max(row["rmse"] for row in results) <= limits["rmse_at_most"],
        "identity": identity_error <= limits["identity_map_max_abs_error_at_most"],
        "output_range": min(row["output_min"] for row in results) >= 0.0 and max(row["output_max"] for row in results) <= 1.0,
        "zero_raw_pixel_quality_reads": True,
    }
    status = "PASS_PRIVATE_DNG_PROFILE_HUESATMAP_SDK_ARITHMETIC" if all(gates.values()) else "FAIL_CLOSED_DNG_PROFILE_HUESATMAP_SDK_ARITHMETIC"
    return {"schema": "neuro-film.p243-dng-profile-huesatmap-sdk-oracle-report.v1", "experiment_id": "P243", "status": status, "bindings": {"config_sha256": _sha(config_path), "oracle_source_sha256": _sha(ROOT / "tests/native/p243_dng_huesatmap_sdk_oracle.cpp")}, "oracle_build": {"executable_bytes": executable_bytes, "executable_sha256": executable_sha}, "probe_count": len(probes), "rows": sorted(results, key=lambda x: (x["id"], x["table"])), "aggregate": {"max_abs_error": max(row["max_abs_error"] for row in results), "max_rmse": max(row["rmse"] for row in results), "identity_max_abs_error": identity_error}, "gates": gates, "information_flow": {"raw_downloads": 0, "pixel_decodes": 0, "image_quality_scores": 0}, "claim_ceiling": config["claim_ceiling"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
