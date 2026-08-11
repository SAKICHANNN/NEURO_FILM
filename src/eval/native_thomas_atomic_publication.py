"""U6.P8CS profile-bound native PNG atomic-publication evidence."""

from __future__ import annotations

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
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_export_profile import _configure_parallel
from src.eval.native_thomas_rgb16_png_conformance import (
    _decode_rgb16,
    _icc_payload,
    build_msvc,
    load_library,
)
from src.film_physics.atomic_native_output import (
    AtomicNativeOutputError,
    publish_native_thomas_rgb16_png,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
)
from src.film_physics.native_thomas_export_profile import (
    canonical_profile_bytes,
    compile_native_thomas_export_profile,
    reconstruct_native_thomas_export_profile,
    validate_native_thomas_export_profile,
)
from src.preprocess.output_encode import srgb_icc_profile


class NativeThomasAtomicPublicationError(RuntimeError):
    """Raised when the frozen P8CS evidence contract drifts."""


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NativeThomasAtomicPublicationError("P8CS JSON must be an object")
    return payload


def _compile_profile(root: Path, contract: dict[str, Any]) -> tuple[object, ...]:
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    fields = _profiles(
        _json(root / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    parent_bindings = dict(contract["profile_source_bindings"])
    profile = compile_native_thomas_export_profile(
        amplitude,
        fields,
        _gauge_payload(),
        source_bindings=parent_bindings,
    )
    if validate_native_thomas_export_profile(profile) != contract["profile_sha256"]:
        raise NativeThomasAtomicPublicationError("P8CS profile identity drift")
    rebuilt = reconstruct_native_thomas_export_profile(
        json.loads(canonical_profile_bytes(profile))
    )
    return prior, *rebuilt


def evaluate(*, root: Path, contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cs_thomas_atomic_publication_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
    ):
        raise NativeThomasAtomicPublicationError("P8CS contract drift")
    parent = contract["parent"]
    parent_path = root / parent["path"]
    if not parent_path.is_file() or sha256_file(parent_path) != parent["sha256"]:
        raise NativeThomasAtomicPublicationError("P8CS P8CR parent drift")
    parent_payload = _json(parent_path)
    if parent_payload.get("decision") != parent["required_decision"]:
        raise NativeThomasAtomicPublicationError("P8CS parent decision drift")

    prior, amplitude, fields, gauge = _compile_profile(root, contract)
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
    _configure_parallel(library)
    runs = []
    for index in range(int(fixture["runs"])):
        destination = (output_dir / f"run-{index + 1}.png").resolve()
        destination.unlink(missing_ok=True)
        row = publish_native_thomas_rgb16_png(
            library,
            amplitude,
            fields,
            gauge,
            exposure,
            row_partition=int(fixture["row_partition"]),
            destination=destination,
            maximum_output_bytes=int(fixture["maximum_output_bytes"]),
        )
        raw = destination.read_bytes()
        decoded = _decode_rgb16(raw)
        icc = _icc_payload(raw)
        runs.append(
            {
                **{key: value for key, value in row.items() if key != "path"},
                "decoded_rgb16_sha256": hashlib.sha256(
                    decoded.tobytes()
                ).hexdigest(),
                "icc_sha256": hashlib.sha256(icc).hexdigest(),
                "icc_exact": icc == srgb_icc_profile(),
            }
        )

    failure_path = (output_dir / "injected-failure.png").resolve()
    failure_path.unlink(missing_ok=True)
    failure_atomic = False
    try:
        publish_native_thomas_rgb16_png(
            library,
            amplitude,
            fields,
            gauge,
            exposure,
            row_partition=int(fixture["row_partition"]),
            destination=failure_path,
            maximum_output_bytes=16,
        )
    except AtomicNativeOutputError:
        failure_atomic = not failure_path.exists() and not list(
            output_dir.glob(".injected-failure.png.*.stage")
        )

    expected = contract["expected"]
    gates = {
        "png_bytes_exact": all(row["sha256"] == expected["png_sha256"] for row in runs),
        "decoded_rgb16_exact": all(
            row["decoded_rgb16_sha256"] == expected["decoded_rgb16_sha256"]
            for row in runs
        ),
        "icc_exact": all(
            row["icc_exact"] and row["icc_sha256"] == expected["icc_sha256"]
            for row in runs
        ),
        "repeat_exact": runs[0] == runs[1],
        "failure_atomic": failure_atomic,
        "no_stage_residue": not list(output_dir.glob(".*.stage")),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "profile_sha256": contract["profile_sha256"],
        "input_sha256": hashlib.sha256(exposure.tobytes()).hexdigest(),
        "output": runs[0],
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cs_thomas_atomic_publication_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(
            canonical_profile_bytes(stable)
        ).hexdigest(),
        "stable": stable,
        "runs": runs,
        "build": {
            key: value for key, value in build.items() if key != "dll_path"
        },
    }


__all__ = ["NativeThomasAtomicPublicationError", "evaluate"]
