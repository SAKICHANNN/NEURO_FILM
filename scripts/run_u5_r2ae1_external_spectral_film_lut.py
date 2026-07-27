"""Run the pinned external spectral_film_lut bank without copying its source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATHS = {
    "KODAK_EKTAR_100": "src/spectral_film_lut/negative_film/kodak_ektar_100.py",
    "KODAK_PORTRA_400": "src/spectral_film_lut/negative_film/kodak_portra_400.py",
    "KODAK_5203": "src/spectral_film_lut/negative_film/kodak_5203.py",
    "KODAK_5207": "src/spectral_film_lut/negative_film/kodak_5207.py",
    "KODAK_5213": "src/spectral_film_lut/negative_film/kodak_5213.py",
    "KODAK_5219": "src/spectral_film_lut/negative_film/kodak_5219.py",
    "KODAK_ENDURA_PREMIER": "src/spectral_film_lut/print_film/kodak_endura_premier.py",
    "KODAK_2383": "src/spectral_film_lut/print_film/kodak_2383.py",
    "KODAK_EKTACHROME_100D": "src/spectral_film_lut/reversal_film/kodak_ektachrome_100d.py",
    "KODACHROME_64": "src/spectral_film_lut/reversal_film/kodachrome_64.py",
}
MANIFEST_SCHEMA = "u5-r2ae1-external-spectral-bank-manifest-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _verify_source(external_root: Path, config: dict[str, Any]) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["external_source"]["revision"]:
        raise RuntimeError("external source revision mismatch")
    for symbol, expected in config["source_profile_hashes"].items():
        observed = _sha256_file(external_root / PROFILE_PATHS[symbol])
        if observed != expected:
            raise RuntimeError(f"external profile hash mismatch: {symbol}")


def _package_versions(config: dict[str, Any]) -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name)
        for name in config["runtime"]["package_versions"]
    }
    if observed != config["runtime"]["package_versions"]:
        raise RuntimeError("external runtime package versions mismatch")
    return observed


def _pipeline_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config["pipeline"]
    red, green, blue = pipeline["printer_lights_rgb"]
    return {
        "mode": pipeline["mode"],
        "input_colorspace": config["synthetic_population"]["input_gamut"],
        "exp_kelvin": pipeline["exp_kelvin"],
        "tint": pipeline["tint"],
        "exp_comp": pipeline["exp_comp"],
        "adx_coding": pipeline["adx_coding"],
        "adx_scaling": pipeline["adx_scaling"],
        "red_light": red,
        "green_light": green,
        "blue_light": blue,
        "projector_kelvin": pipeline["projector_kelvin"],
        "white_clip": pipeline["white_clip"],
        "white_balance": pipeline["white_balance"],
        "output_gamut": config["synthetic_population"]["output_gamut"],
        "sat_adjust": pipeline["sat_adjust"],
        "shadow_comp": pipeline["shadow_comp"],
        "gamma_func": config["synthetic_population"]["output_encoding"],
        "push_pull": pipeline["push_pull"],
        "inversion": pipeline["inversion"],
        "idealized_curve": pipeline["idealized_curve"],
        "apd_intermediate": pipeline["apd_intermediate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if platform.python_version() != config["runtime"]["python_version"]:
        raise RuntimeError("external runner Python version mismatch")
    _verify_source(args.external_root, config)
    versions = _package_versions(config)

    source_path = str((args.external_root / "src").resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import spectral_film_lut as stock_module
    from spectral_film_lut.film_spectral import FilmSpectral
    from spectral_film_lut.utils import create_lut, film_conversion

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = _pipeline_kwargs(config)
    size = int(config["synthetic_population"]["cube_size"])
    neutral_count = int(config["synthetic_population"]["neutral_samples"])
    neutral_axis = np.linspace(0.0, 1.0, neutral_count, dtype=np.float32)
    neutral_input = np.repeat(neutral_axis[:, None], 3, axis=1)
    records = []
    rendered: dict[str, np.ndarray] = {}
    for chain in config["chains"]:
        negative_data = getattr(stock_module, chain["negative_symbol"])
        print_data = (
            None
            if chain["print_symbol"] is None
            else getattr(stock_module, chain["print_symbol"])
        )
        negative = FilmSpectral(negative_data)
        print_film = None if print_data is None else FilmSpectral(print_data)
        output = np.asarray(
            create_lut(
                negative,
                print_film,
                lut_size=size,
                cube=False,
                **kwargs,
            ),
            dtype="<f4",
        )
        neutral = np.asarray(
            film_conversion(
                neutral_input.copy(), negative, print_film, **kwargs
            ),
            dtype="<f4",
        )
        output_path = args.output_dir / f"{chain['id']}.npy"
        neutral_path = args.output_dir / f"{chain['id']}_neutral.npy"
        np.save(output_path, output, allow_pickle=False)
        np.save(neutral_path, neutral, allow_pickle=False)
        rendered[chain["id"]] = output
        records.append(
            {
                "chain_id": chain["id"],
                "family": chain["family"],
                "negative_symbol": chain["negative_symbol"],
                "print_symbol": chain["print_symbol"],
                "output_path": str(output_path.resolve()),
                "neutral_path": str(neutral_path.resolve()),
                "output_file_sha256": _sha256_file(output_path),
                "neutral_file_sha256": _sha256_file(neutral_path),
                "output_array_sha256": _array_sha256(output),
                "neutral_array_sha256": _array_sha256(neutral),
            }
        )

    duplicate_id = config["controls"]["exact_duplicate_chain_id"]
    duplicate_chain = next(row for row in config["chains"] if row["id"] == duplicate_id)
    negative = FilmSpectral(getattr(stock_module, duplicate_chain["negative_symbol"]))
    print_film = FilmSpectral(getattr(stock_module, duplicate_chain["print_symbol"]))
    duplicate = np.asarray(
        create_lut(negative, print_film, lut_size=size, cube=False, **kwargs),
        dtype="<f4",
    )
    duplicate_path = args.output_dir / "duplicate_control.npy"
    np.save(duplicate_path, duplicate, allow_pickle=False)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "external_revision": config["external_source"]["revision"],
        "python_version": platform.python_version(),
        "package_versions": versions,
        "config_sha256": _canonical_sha256(config),
        "records": records,
        "duplicate_control": {
            "chain_id": duplicate_id,
            "output_path": str(duplicate_path.resolve()),
            "output_file_sha256": _sha256_file(duplicate_path),
            "output_array_sha256": _array_sha256(duplicate),
            "matches_primary_in_process": bool(
                np.array_equal(duplicate, rendered[duplicate_id])
            ),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

