from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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


def _validate_bound_file(binding: list[Any]) -> Path:
    path = ROOT / str(binding[0])
    if not path.is_file():
        raise ValueError(f"bound file is absent: {binding[0]}")
    if path.stat().st_size != int(binding[1]):
        raise ValueError(f"bound file size mismatch: {binding[0]}")
    if _sha256_file(path) != str(binding[2]):
        raise ValueError(f"bound file SHA-256 mismatch: {binding[0]}")
    return path


def _validate_runtime(config: dict[str, Any]) -> tuple[Path, Path]:
    runtime = config["runtime"]
    _validate_bound_file(runtime["manifest"])
    root = ROOT / runtime["root"]
    paths: list[Path] = []
    for key in ("avifdec", "avifgainmaputil"):
        name, size, digest = runtime[key]
        path = root / name
        if path.stat().st_size != int(size) or _sha256_file(path) != digest:
            raise ValueError(f"runtime identity mismatch: {name}")
        paths.append(path)
    return paths[0], paths[1]


def _validate_source(config: dict[str, Any]) -> Path:
    source = config["source"]
    _validate_bound_file(source["manifest"])
    path = ROOT / source["root"] / source["name"]
    if path.stat().st_size != int(source["bytes"]):
        raise ValueError("source size mismatch")
    if _sha256_file(path) != source["sha256"]:
        raise ValueError("source SHA-256 mismatch")
    return path


def _expand(
    tokens: list[str], executable: Path, replacements: dict[str, Path]
) -> list[str]:
    command = [str(executable)]
    for token in tokens[1:]:
        value = token
        for key, path in replacements.items():
            value = value.replace("{" + key + "}", str(path))
        command.append(value)
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def _decode_png(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"PNG decode failed: {path.name}")
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint16:
        raise ValueError(f"unexpected PNG array: {value.shape} {value.dtype}")
    return np.ascontiguousarray(value)


def _to_12bit(value: np.ndarray) -> np.ndarray:
    scaled = np.rint(value.astype(np.float64) * (4095.0 / 65535.0))
    return np.clip(scaled, 0.0, 4095.0).astype(np.int16)


def _error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("endpoint shape mismatch")
    difference = np.abs(
        _to_12bit(reference).astype(np.int32)
        - _to_12bit(candidate).astype(np.int32)
    )
    return {
        "maximum": float(np.max(difference)),
        "median": float(np.median(difference)),
        "p95": float(np.percentile(difference, 95.0)),
    }


def _headroom(text: str, role: str) -> float:
    match = re.search(rf"\* {role} headroom:\s+([-+0-9.eE]+)", text)
    if match is None:
        raise ValueError(f"missing {role.lower()} headroom")
    return float(match.group(1))


def _cicp(info: str, *, alternate: bool) -> tuple[int, int, int]:
    section = info
    if alternate:
        marker = " * Alternate image:"
        if marker not in info:
            raise ValueError("missing alternate-image info")
        section = info.split(marker, 1)[1]
    else:
        section = info.split(" * Alternate image:", 1)[0]
    patterns = (
        r"Color Primaries\s*:\s*(\d+)",
        r"Transfer Char\.\s*:\s*(\d+)",
        r"Matrix Coeffs\.\s*:\s*(\d+)",
    )
    values: list[int] = []
    for pattern in patterns:
        match = re.search(pattern, section)
        if match is None:
            raise ValueError("missing CICP field")
        values.append(int(match.group(1)))
    return values[0], values[1], values[2]


def _publish_combine_create_only(
    command: list[str], destination: Path
) -> subprocess.CompletedProcess[str]:
    if destination.exists():
        raise FileExistsError(destination)
    return _run(command)


def _build_endpoints(
    config: dict[str, Any],
    source: Path,
    avifdec: Path,
    gainmaputil: Path,
    scratch: Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray, dict[str, Any]]:
    hdr = scratch / "endpoint_hdr.png"
    sdr = scratch / "endpoint_sdr.png"
    replacements = {"source": source, "hdr_png": hdr, "sdr_png": sdr}
    hdr_result = _run(
        _expand(config["commands"]["hdr_endpoint"], avifdec, replacements)
    )
    sdr_result = _run(
        _expand(config["commands"]["sdr_endpoint"], gainmaputil, replacements)
    )
    if hdr_result.returncode != 0 or sdr_result.returncode != 0:
        raise ValueError("endpoint materialization failed")
    hdr_pixels = _decode_png(hdr)
    sdr_pixels = _decode_png(sdr)
    frozen = {
        "hdr_pixel_sha256": _sha256_bytes(hdr_pixels.tobytes()),
        "hdr_png_sha256": _sha256_file(hdr),
        "sdr_pixel_sha256": _sha256_bytes(sdr_pixels.tobytes()),
        "sdr_png_sha256": _sha256_file(sdr),
    }
    return hdr, sdr, hdr_pixels, sdr_pixels, frozen


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load(config_path)
    for binding in config.get("execution_bindings", {}).values():
        _validate_bound_file(binding)
    _validate_bound_file(config["parents"]["p282_config"])
    _validate_bound_file(config["parents"]["p282_evidence"])
    contract = config["contract"]
    contract_path = ROOT / contract["path"]
    if (
        contract_path.stat().st_size != int(contract["bytes"])
        or _sha256_file(contract_path) != contract["sha256"]
    ):
        raise ValueError("contract identity mismatch")
    avifdec, gainmaputil = _validate_runtime(config)
    source = _validate_source(config)
    source_before = _sha256_file(source)
    runtime_before = {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    controls = ["foreign", "missing", "truncated"]
    if reverse:
        controls.reverse()

    with tempfile.TemporaryDirectory(prefix="p289_", dir=ROOT / "tmp") as raw:
        scratch = Path(raw)
        hdr, sdr, hdr_pixels, sdr_pixels, endpoints = _build_endpoints(
            config, source, avifdec, gainmaputil, scratch
        )
        candidate = scratch / "candidate.avif"
        replacements = {
            "source": source,
            "hdr_png": hdr,
            "sdr_png": sdr,
            "candidate_avif": candidate,
        }
        combine_command = _expand(
            config["commands"]["combine"], gainmaputil, replacements
        )
        combined = _publish_combine_create_only(combine_command, candidate)
        candidate_exists = candidate.is_file()

        metadata = _run([str(gainmaputil), "printmetadata", str(candidate), "--jobs", "1"])
        info = _run([str(avifdec), "--info", str(candidate)])
        metadata_text = metadata.stdout + "\n" + metadata.stderr
        info_text = info.stdout + "\n" + info.stderr

        decoded_hdr = scratch / "decoded_hdr.png"
        decoded_sdr = scratch / "decoded_sdr.png"
        base_result = _run(
            _expand(
                config["commands"]["hdr_endpoint"],
                avifdec,
                {"source": candidate, "hdr_png": decoded_hdr},
            )
        )
        tone_result = _run(
            _expand(
                config["commands"]["sdr_endpoint"],
                gainmaputil,
                {"source": candidate, "sdr_png": decoded_sdr},
            )
        )
        decoded_hdr_pixels = _decode_png(decoded_hdr)
        decoded_sdr_pixels = _decode_png(decoded_sdr)
        hdr_error = _error(hdr_pixels, decoded_hdr_pixels)
        sdr_error = _error(sdr_pixels, decoded_sdr_pixels)
        hdr_12 = _to_12bit(hdr_pixels)
        sdr_12 = _to_12bit(sdr_pixels)
        material_fraction = float(np.count_nonzero(hdr_12 != sdr_12) / hdr_12.size)

        control_results: dict[str, bool] = {}
        for role in controls:
            output = scratch / f"control_{role}.avif"
            if role == "foreign":
                payload = b"P289 foreign destination\n"
                output.write_bytes(payload)
                try:
                    _publish_combine_create_only(
                        _expand(
                            config["commands"]["combine"],
                            gainmaputil,
                            {
                                "hdr_png": hdr,
                                "sdr_png": sdr,
                                "candidate_avif": output,
                            },
                        ),
                        output,
                    )
                    rejected = False
                except FileExistsError:
                    rejected = output.read_bytes() == payload
                control_results[role] = rejected
            else:
                invalid = scratch / f"{role}.png"
                if role == "truncated":
                    invalid.write_bytes(hdr.read_bytes()[:64])
                result = _run(
                    _expand(
                        config["commands"]["combine"],
                        gainmaputil,
                        {
                            "hdr_png": invalid,
                            "sdr_png": sdr,
                            "candidate_avif": output,
                        },
                    )
                )
                control_results[role] = result.returncode != 0 and not output.exists()

        base_headroom = _headroom(metadata_text, "Base")
        alternate_headroom = _headroom(metadata_text, "Alternate")
        base_cicp = _cicp(info_text, alternate=False)
        alternate_cicp = _cicp(info_text, alternate=True)
        dimensions = tuple(int(value) for value in config["gates"]["require_dimensions"])
        gates = {
            "atomic-invalid-controls": all(control_results.values()),
            "base-cicp-exact": base_cicp == (1, 16, 0),
            "candidate-created": combined.returncode == 0 and candidate_exists,
            "candidate-gain-map-recognized": metadata.returncode == 0
            and info.returncode == 0
            and "Gain map" in info_text,
            "dimensions-and-endpoint-types-exact": hdr_pixels.shape[:2]
            == (dimensions[1], dimensions[0])
            and sdr_pixels.shape == hdr_pixels.shape,
            "endpoint-material-difference": material_fraction
            >= float(config["gates"]["minimum_material_endpoint_difference_fraction"]),
            "headrooms-exact": base_headroom
            == float(config["gates"]["require_base_headroom"])
            and alternate_headroom
            == float(config["gates"]["require_alternate_headroom"]),
            "hdr-endpoint-error": hdr_error["median"]
            <= float(config["gates"]["endpoint_median_abs_error_12bit_max"])
            and hdr_error["p95"]
            <= float(config["gates"]["endpoint_p95_abs_error_12bit_max"])
            and hdr_error["maximum"]
            <= float(config["gates"]["endpoint_max_abs_error_12bit_max"]),
            "sdr-endpoint-error": sdr_error["median"]
            <= float(config["gates"]["endpoint_median_abs_error_12bit_max"])
            and sdr_error["p95"]
            <= float(config["gates"]["endpoint_p95_abs_error_12bit_max"])
            and sdr_error["maximum"]
            <= float(config["gates"]["endpoint_max_abs_error_12bit_max"]),
            "runtime-decodes-succeed": base_result.returncode == 0
            and tone_result.returncode == 0,
            "alternate-cicp-exact": alternate_cicp == (1, 13, 0),
        }
        scientific = {
            "alternate_cicp": alternate_cicp,
            "alternate_headroom": alternate_headroom,
            "base_cicp": base_cicp,
            "base_headroom": base_headroom,
            "candidate_avif_bytes": candidate.stat().st_size,
            "candidate_avif_sha256": _sha256_file(candidate),
            "candidate_decoded_hdr_pixel_sha256": _sha256_bytes(
                decoded_hdr_pixels.tobytes()
            ),
            "candidate_decoded_sdr_pixel_sha256": _sha256_bytes(
                decoded_sdr_pixels.tobytes()
            ),
            "control_results": dict(sorted(control_results.items())),
            "endpoints_frozen_before_combine": endpoints,
            "gates": dict(sorted(gates.items())),
            "hdr_error_12bit": hdr_error,
            "material_endpoint_difference_fraction": material_fraction,
            "sdr_error_12bit": sdr_error,
            "source_sha256": source_before,
            "runtime_sha256": runtime_before,
            "zero_network_reads": True,
            "zero_persistent_media": True,
        }
        required_pass = all(gates.values())
        status = (
            "PASS_PRIVATE_LIBAVIF_GAINMAP_ENCODER_D0"
            if required_pass
            else "FAIL_CLOSED_LIBAVIF_GAINMAP_ENCODER_D0"
        )
        stable_identity = "sha256:" + _sha256_bytes(_canonical(scientific))

    source_after = _sha256_file(source)
    runtime_after = {
        avifdec.name: _sha256_file(avifdec),
        gainmaputil.name: _sha256_file(gainmaputil),
    }
    scientific["source-runtime-immutable"] = (
        source_after == source_before and runtime_after == runtime_before
    )
    scientific["gates"]["source-runtime-immutable"] = scientific[
        "source-runtime-immutable"
    ]
    if not scientific["source-runtime-immutable"]:
        status = "FAIL_CLOSED_LIBAVIF_GAINMAP_ENCODER_D0"
    stable_identity = "sha256:" + _sha256_bytes(_canonical(scientific))
    return {
        "claim_ceiling": config["claim_ceiling"],
        "schema": "neuro-film.p289-libavif-gainmap-encoder-d0-result.v1",
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
    result = run((ROOT / arguments.config).resolve(), reverse=arguments.order == "reverse")
    output = (ROOT / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical(result))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
