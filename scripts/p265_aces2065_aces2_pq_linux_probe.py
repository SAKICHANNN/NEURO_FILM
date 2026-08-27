"""Isolated Linux probe for the strict P251/P252 AP0 to PQ path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _array_sha256(value: Any, numpy: Any, *, dtype: str | None = None) -> str:
    array = numpy.ascontiguousarray(
        value if dtype is None else value.astype(dtype)
    )
    return _sha256_bytes(array.tobytes(order="C"))


def _install_exact_package_roots(root: Path) -> dict[str, int]:
    """Expose exact submodules without executing broad preprocess initializers."""

    source_package = types.ModuleType("src")
    source_package.__path__ = [str(root / "src")]  # type: ignore[attr-defined]
    preprocess_package = types.ModuleType("src.preprocess")
    preprocess_package.__path__ = [  # type: ignore[attr-defined]
        str(root / "src" / "preprocess")
    ]
    sys.modules["src"] = source_package
    sys.modules["src.preprocess"] = preprocess_package

    sentinel_calls = {"srgb_icc_profile": 0}
    output_encode = types.ModuleType("src.preprocess.output_encode")

    def forbidden_sdr_icc_profile() -> bytes:
        sentinel_calls["srgb_icc_profile"] += 1
        raise RuntimeError("SDR ICC dependency is outside the frozen PQ path")

    output_encode.srgb_icc_profile = forbidden_sdr_icc_profile  # type: ignore[attr-defined]
    sys.modules["src.preprocess.output_encode"] = output_encode
    return sentinel_calls


def _load_writer(path: Path) -> types.ModuleType:
    specification = importlib.util.spec_from_file_location(
        "p265_bound_r1dt_writer", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load the bound R1DT writer")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _write_wrong_identity(source: Path, output: Path, openexr: Any) -> None:
    with openexr.File(str(source)) as source_file:
        header = dict(source_file.header())
        pixels = source_file.channels()["RGB"].pixels.copy()
    header["colorInteropID"] = "lin_ap1_scene"
    with openexr.File(header, {"RGB": pixels}) as destination:
        destination.write(str(output))


def _expect_rejection(source: Path, output: Path) -> bool:
    from src.preprocess.aces2065_aces2_pq import (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
    )

    try:
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(source, output)
    except (RuntimeError, ValueError):
        return not output.exists()
    return False


def execute(
    *,
    root: Path,
    workspace: Path,
    writer_source: Path,
    input_pixels: Path,
    order: str,
) -> dict[str, Any]:
    import numpy as np

    sentinel_calls = _install_exact_package_roots(root)
    from src.preprocess.aces2065_aces2_pq import (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
    )
    from src.preprocess.aces2065_openexr import (
        load_aces2065_openexr_working_image,
    )
    from src.preprocess.ocio_aces2_output import load_aces2_config
    from src.preprocess.png_stream import (
        sha256_rec2100_pq_rgb16_png_samples,
    )

    openexr = importlib.import_module("OpenEXR")
    ocio = importlib.import_module("PyOpenColorIO")
    writer = _load_writer(writer_source)
    pixels = np.fromfile(input_pixels, dtype="<f4").reshape(7, 9, 3)
    pixels_before = _array_sha256(pixels, np, dtype="<f4")
    source = workspace / "p249-ap0.exr"
    writer.write_aces2065_1_openexr(source, pixels)
    source_sha = _sha256_file(source)
    working = load_aces2065_openexr_working_image(source)
    output = workspace / "p252-linux.png"
    receipt_sha, samples, encoded = (
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
            source,
            output,
            row_count=3,
            reverse_partition=order == "reverse",
        )
    )

    malformed = workspace / "malformed.exr"
    malformed.write_bytes(b"not-an-openexr")
    wrong_identity = workspace / "wrong-identity.exr"
    _write_wrong_identity(source, wrong_identity, openexr)
    control_sources = {
        "malformed": malformed,
        "wrong_identity": wrong_identity,
    }
    control_names = list(control_sources)
    if order == "reverse":
        control_names.reverse()
    rejected: dict[str, bool] = {}
    for name in control_names:
        rejected[name] = _expect_rejection(
            control_sources[name], workspace / f"{name}.png"
        )

    foreign = workspace / "foreign.png"
    foreign.write_bytes(b"foreign-destination")
    foreign_before = foreign.read_bytes()
    try:
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(source, foreign)
    except ValueError:
        foreign_rejected = True
    else:
        foreign_rejected = False

    config = load_aces2_config()
    result = {
        "config_cache_id": config.getCacheID(),
        "encoded_f32le_sha256": _array_sha256(encoded, np, dtype="<f4"),
        "encoded_finite_in_unit": bool(
            np.isfinite(encoded).all()
            and np.all(encoded >= 0.0)
            and np.all(encoded <= 1.0)
        ),
        "foreign_destination_rejected": foreign_rejected,
        "foreign_destination_unchanged": foreign.read_bytes() == foreign_before,
        "input_pixels_unchanged": _array_sha256(pixels, np, dtype="<f4")
        == pixels_before,
        "invalid_controls_rejected": dict(sorted(rejected.items())),
        "numpy_version": np.__version__,
        "openexr_version": openexr.__version__,
        "opencolorio_version": ocio.__version__,
        "output_bytes": output.stat().st_size,
        "output_png_sha256": _sha256_file(output),
        "output_receipt_sha256": receipt_sha,
        "output_samples_sha256": _array_sha256(samples, np),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha,
        "source_unchanged": _sha256_file(source) == source_sha,
        "strict_sample_sha256": sha256_rec2100_pq_rgb16_png_samples(
            output, width=9, height=7
        ),
        "unused_sdr_dependency_calls": dict(sentinel_calls),
        "working": {
            "f32le_sha256": _array_sha256(working.pixels, np, dtype="<f4"),
            "maximum": float(np.max(working.pixels)),
            "minimum": float(np.min(working.pixels)),
            "shape": list(working.pixels.shape),
            "transfer_state": working.transfer_state,
            "working_space": working.working_space,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--writer-source", type=Path, required=True)
    parser.add_argument("--input-pixels", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        root=args.root.resolve(),
        workspace=args.workspace.resolve(),
        writer_source=args.writer_source.resolve(),
        input_pixels=args.input_pixels.resolve(),
        order=args.order,
    )
    args.output.write_bytes(_canonical_bytes(result))


if __name__ == "__main__":
    main()
