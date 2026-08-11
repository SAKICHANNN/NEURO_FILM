"""U6.P8CR canonical profile-ingress audit for the Thomas PNG core."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _parent_payloads,
    _profiles,
)
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import (
    _decode_rgb16,
    _icc_payload,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_png_conformance import (
    ByteSink,
    build_msvc,
    load_library,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.native_gauge_profile import native_gauge_profile_struct
from src.film_physics.native_granularity_amplitude import (
    NativeGranularityAmplitudeProfileV1,
    compile_native_granularity_amplitude_profile,
)
from src.film_physics.native_thomas_export_profile import (
    canonical_profile_bytes,
    compile_native_thomas_export_profile,
    reconstruct_native_thomas_export_profile,
    validate_native_thomas_export_profile,
)
from src.film_physics.native_thomas_field import NativeThomasFieldProfileV1
from src.preprocess.output_encode import srgb_icc_profile


class NativeThomasExportProfileError(RuntimeError):
    """Raised when the frozen P8CR contract or native replay drifts."""


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NativeThomasExportProfileError("P8CR JSON must be an object")
    return payload


def _struct_bytes(value: ctypes.Structure) -> bytes:
    return ctypes.string_at(ctypes.byref(value), ctypes.sizeof(value))


def _validate_contract(root: Path, contract: dict[str, Any]) -> None:
    if (
        contract.get("schema") != "neuro_film.u6_p8cr_thomas_export_profile_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or int(contract["fixture"]["runs"]) != 2
    ):
        raise NativeThomasExportProfileError("P8CR contract drift")
    for name, binding in contract["parents"].items():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise NativeThomasExportProfileError(f"P8CR parent drift: {name}")
    p8cq = _json(root / contract["parents"]["p8cq_export_core"]["path"])
    if (
        p8cq.get("decision")
        != contract["parents"]["p8cq_export_core"]["required_decision"]
    ):
        raise NativeThomasExportProfileError("P8CR P8CQ decision drift")


def _configure_parallel(library: ctypes.CDLL) -> None:
    library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1.restype = (
        ctypes.c_int
    )
    library.nf_thomas_rgb16_png_cached_parallel_apply_v1.argtypes = [
        ctypes.POINTER(NativeGranularityAmplitudeProfileV1),
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ByteSink,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_thomas_rgb16_png_cached_parallel_apply_v1.restype = ctypes.c_int


def _run(
    library: ctypes.CDLL,
    amplitude: NativeGranularityAmplitudeProfileV1,
    fields: tuple[
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
    ],
    gauge: ctypes.Structure,
    exposure: np.ndarray,
    *,
    row_partition: int,
) -> dict[str, Any]:
    height, width = exposure.shape[1:]
    workspace_bytes = ctypes.c_size_t()
    status = library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1(
        height, width, row_partition, ctypes.byref(workspace_bytes)
    )
    if status != 0:
        raise NativeThomasExportProfileError("P8CR workspace request failed")
    workspace = ctypes.create_string_buffer(workspace_bytes.value)
    field_array = (NativeThomasFieldProfileV1 * 3)(*fields)
    chunks: list[bytes] = []

    @ByteSink
    def sink(
        _context: int,
        values: ctypes.POINTER(ctypes.c_uint8),
        count: int,
    ) -> int:
        chunks.append(ctypes.string_at(values, count))
        return 1

    means = (ctypes.c_double * 3)(-13.0, -13.0, -13.0)
    status = library.nf_thomas_rgb16_png_cached_parallel_apply_v1(
        ctypes.byref(amplitude),
        field_array,
        ctypes.byref(gauge),
        height,
        width,
        row_partition,
        exposure.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        exposure.size,
        workspace,
        workspace_bytes.value,
        sink,
        None,
        means,
    )
    if status != 0:
        raise NativeThomasExportProfileError(f"P8CR native apply failed: {status}")
    png = b"".join(chunks)
    decoded = _decode_rgb16(png)
    icc = _icc_payload(png)
    return {
        "png": png,
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
        "icc_sha256": hashlib.sha256(icc).hexdigest(),
        "icc_exact": icc == srgb_icc_profile(),
        "raw_field_means": [float(value) for value in means],
        "workspace_bytes": workspace_bytes.value,
    }


def evaluate(*, root: Path, contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    _validate_contract(root, contract)
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    fields = _profiles(
        _json(root / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    gauge_payload = _gauge_payload()
    parent_bindings = {
        name: binding["sha256"] for name, binding in contract["parents"].items()
    }
    parent_bindings.update(
        {
            "p4bw_profile": sha256_file(
                root / "outputs/experiments/"
                "u6_p4bw_density_conditioned_thomas_profile_v1/run_a/bundle.json"
            ),
            "p2q_characteristic": sha256_file(
                root / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json"
            ),
        }
    )
    profile = compile_native_thomas_export_profile(
        amplitude,
        fields,
        gauge_payload,
        source_bindings=parent_bindings,
    )
    encoded = canonical_profile_bytes(profile)
    decoded = json.loads(encoded)
    profile_sha = validate_native_thomas_export_profile(decoded)
    rebuilt_amplitude, rebuilt_fields, rebuilt_gauge = (
        reconstruct_native_thomas_export_profile(decoded)
    )
    original_fields = tuple(profile_row.as_abi() for profile_row in fields)
    original_gauge = native_gauge_profile_struct(gauge_payload)
    abi_exact = {
        "amplitude": _struct_bytes(amplitude) == _struct_bytes(rebuilt_amplitude),
        "fields": all(
            _struct_bytes(expected) == _struct_bytes(actual)
            for expected, actual in zip(original_fields, rebuilt_fields)
        ),
        "gauge": _struct_bytes(original_gauge) == _struct_bytes(rebuilt_gauge),
    }
    fixture = contract["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    rng = np.random.default_rng(int(fixture["exposure_seed"]))
    exposure = np.empty((3, height, width), dtype=np.float32)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        margin = (upper - lower) * 0.01
        exposure[channel] = rng.uniform(
            lower + margin, upper - margin, size=(height, width)
        ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    build = build_msvc(root, output_dir / "build")
    library = load_library(Path(build["dll_path"]))
    build_evidence = dict(build)
    build_evidence.pop("dll_path")
    _configure_parallel(library)
    original = _run(
        library,
        amplitude,
        original_fields,
        original_gauge,
        exposure,
        row_partition=int(fixture["row_partition"]),
    )
    rebuilt_runs = [
        _run(
            library,
            rebuilt_amplitude,
            rebuilt_fields,
            rebuilt_gauge,
            exposure,
            row_partition=int(fixture["row_partition"]),
        )
        for _ in range(int(fixture["runs"]))
    ]
    tampered = json.loads(encoded)
    tampered["fields"][0]["mean_offspring"] += 0.25
    invalid_rejected = False
    try:
        validate_native_thomas_export_profile(tampered)
    except ValueError:
        invalid_rejected = True
    stable_runs = [
        {key: value for key, value in row.items() if key != "png"}
        for row in rebuilt_runs
    ]
    gates = {
        "canonical_roundtrip_exact": canonical_profile_bytes(decoded) == encoded,
        "abi_struct_bytes_exact": all(abi_exact.values()),
        "png_bytes_exact": all(row["png"] == original["png"] for row in rebuilt_runs),
        "decoded_rgb16_exact": all(
            row["decoded_sha256"] == original["decoded_sha256"] for row in rebuilt_runs
        ),
        "icc_exact": original["icc_exact"]
        and all(row["icc_exact"] for row in rebuilt_runs),
        "invalid_profile_rejected_before_output": invalid_rejected,
        "repeat_exact": stable_runs[0] == stable_runs[1],
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "profile_sha256": profile_sha,
        "profile_bytes": len(encoded),
        "source_bindings": profile["source_bindings"],
        "abi_struct_bytes_exact": abi_exact,
        "input_sha256": hashlib.sha256(exposure.tobytes()).hexdigest(),
        "output": stable_runs[0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cr_thomas_export_profile_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(
            canonical_profile_bytes(stable)
        ).hexdigest(),
        **stable,
        "profile": profile,
        "build": build_evidence,
        "original_output": {
            key: value for key, value in original.items() if key != "png"
        },
        "rebuilt_outputs": stable_runs,
    }


__all__ = ["NativeThomasExportProfileError", "evaluate"]
