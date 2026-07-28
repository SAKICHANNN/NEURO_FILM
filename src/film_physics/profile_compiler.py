"""Compile the P7F challenger into a strict fixed-reference profile artifact."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_chain import (
    validate_contract as validate_p7f_contract,
)
from src.eval.physical_neutral_gauged_invariance import (
    _bounded_real_source,
    render_challenger,
)
from src.eval.physical_virtual_scan_sampling import (
    compile_virtual_scan_profile,
)
from src.film_physics.contracts import (
    ComponentBinding,
    FilmProfileBundle,
    PhysicalDomain,
)
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_interpretation_bounded_development_adjacency,
)
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator
from src.roll2film.sensitometry_print import SensitometryPrintOperator
from src.roll2film.splines import RationalQuadraticSpline


SCHEMA = "neuro_film.u6_p8a_fixed_reference_profile_compiler_contract.v1"
ARTIFACT_SCHEMA = "neuro_film.u6_p8a_fixed_reference_profile_artifact.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[Any, NeutralAxisGaugeOperator]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["profile"]
        != {
            "claim_level": "generic-physical-inspired",
            "stock_id": "unknown",
            "process_id": "unknown",
            "scanner_profile_id": "unknown",
        }
        or config["preview_policy"].get(
            "independent_low_resolution_equivalence_claim_allowed"
        )
    ):
        raise ValueError("unsupported U6.P8A contract")
    decision = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    parent = _load_exact_json(
        root,
        config["parent_contract"],
        config["parent_contract_sha256"],
    )
    _load_exact_json(
        root,
        config["ao6_runtime_contract"],
        config["ao6_runtime_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8A")
        or not decision["profile_compiler_opened"]
        or decision["production_default_changed"]
        or int(decision["reference_sampling_dpi"])
        != int(config["reference_sampling_dpi"])
        or parent["runtime_parent"] != config["ao6_runtime_contract"]
        or parent["runtime_parent_sha256"]
        != config["ao6_runtime_contract_sha256"]
    ):
        raise ValueError("U6.P8A parent evidence drift")
    return validate_p7f_contract(root, parent)


def _spatial_payload(profile: SpatialResponseProfile) -> dict[str, Any]:
    return {
        "schema": "neuro_film.compiled_spatial_response_profile.v1",
        "pixel_pitch_um": profile.pixel_pitch_um,
        "forward_scatter_sigma_um_rgb": list(
            profile.forward_scatter_sigma_um_rgb
        ),
        "development_adjacency_sigma_um_rgb": list(
            profile.development_adjacency_sigma_um_rgb
        ),
        "development_adjacency_gain_rgb": list(
            profile.development_adjacency_gain_rgb
        ),
        "dye_diffusion_sigma_um_rgb": list(
            profile.dye_diffusion_sigma_um_rgb
        ),
        "scanner_mtf_sigma_um_rgb": list(
            profile.scanner_mtf_sigma_um_rgb
        ),
        "gaussian_truncate": profile.gaussian_truncate,
    }


def _spatial_from_payload(payload: dict[str, Any]) -> SpatialResponseProfile:
    expected = {
        "schema",
        "pixel_pitch_um",
        "forward_scatter_sigma_um_rgb",
        "development_adjacency_sigma_um_rgb",
        "development_adjacency_gain_rgb",
        "dye_diffusion_sigma_um_rgb",
        "scanner_mtf_sigma_um_rgb",
        "gaussian_truncate",
    }
    if (
        set(payload) != expected
        or payload["schema"]
        != "neuro_film.compiled_spatial_response_profile.v1"
    ):
        raise ValueError("compiled spatial payload drift")
    return SpatialResponseProfile(
        pixel_pitch_um=float(payload["pixel_pitch_um"]),
        forward_scatter_sigma_um_rgb=tuple(
            float(value)
            for value in payload["forward_scatter_sigma_um_rgb"]
        ),
        development_adjacency_sigma_um_rgb=tuple(
            float(value)
            for value in payload[
                "development_adjacency_sigma_um_rgb"
            ]
        ),
        development_adjacency_gain_rgb=tuple(
            float(value)
            for value in payload["development_adjacency_gain_rgb"]
        ),
        dye_diffusion_sigma_um_rgb=tuple(
            float(value)
            for value in payload["dye_diffusion_sigma_um_rgb"]
        ),
        scanner_mtf_sigma_um_rgb=tuple(
            float(value)
            for value in payload["scanner_mtf_sigma_um_rgb"]
        ),
        gaussian_truncate=float(payload["gaussian_truncate"]),
    )


def _adjacency_payload(runtime: Any) -> dict[str, Any]:
    keywords = runtime.apply_adjacency.keywords
    return {
        "schema": (
            "neuro_film.interpretation_bounded_adjacency_binding.v1"
        ),
        "maximum_absolute_transmittance_delta": float(
            keywords["maximum_absolute_transmittance_delta"]
        ),
        "maximum_absolute_density_delta": float(
            keywords["maximum_absolute_density_delta"]
        ),
        "black_reference_density": np.asarray(
            keywords["black_reference_density"], dtype=np.float64
        ).tolist(),
        "white_reference_density": np.asarray(
            keywords["white_reference_density"], dtype=np.float64
        ).tolist(),
    }


def _adjacency_from_payload(payload: dict[str, Any]) -> Any:
    expected = {
        "schema",
        "maximum_absolute_transmittance_delta",
        "maximum_absolute_density_delta",
        "black_reference_density",
        "white_reference_density",
    }
    if (
        set(payload) != expected
        or payload["schema"]
        != "neuro_film.interpretation_bounded_adjacency_binding.v1"
    ):
        raise ValueError("compiled adjacency payload drift")
    return partial(
        apply_interpretation_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=float(
            payload["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            payload["maximum_absolute_density_delta"]
        ),
        black_reference_density=np.asarray(
            payload["black_reference_density"], dtype=np.float64
        ),
        white_reference_density=np.asarray(
            payload["white_reference_density"], dtype=np.float64
        ),
    )


def _gauge_payload(
    gauge: NeutralAxisGaugeOperator, *, base_component_sha256: str
) -> dict[str, Any]:
    return {
        "schema": "neuro_film.compiled_neutral_axis_gauge.v1",
        "base_component_sha256": base_component_sha256,
        "inverse_neutral_splines": [
            spline.to_dict() for spline in gauge.inverse_neutral_splines
        ],
    }


def compile_profile_artifact(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime, gauge = validate_contract(root, config)
    compiled_profile = compile_virtual_scan_profile(
        runtime.profile,
        sampling_dpi=int(config["reference_sampling_dpi"]),
    )
    physical = {
        "schema": "neuro_film.compiled_physical_chain_component.v1",
        "reference_sampling_dpi": int(config["reference_sampling_dpi"]),
        "spatial_profile": _spatial_payload(compiled_profile),
        "print_operator": runtime.print_operator.to_dict(),
        "adjacency": _adjacency_payload(runtime),
    }
    physical_sha = _payload_sha256(physical)
    gauge_payload = _gauge_payload(
        gauge, base_component_sha256=physical_sha
    )
    oetf = {
        "schema": "neuro_film.srgb_oetf_component.v1",
        "input": "relative-linear-srgb-d65",
        "output": "iec-61966-2-1-srgb-encoded",
        "quantization": "none",
    }
    ao6 = {
        "schema": "neuro_film.ao6_source_context_display_look_binding.v1",
        "runtime_contract": config["ao6_runtime_contract"],
        "runtime_contract_sha256": config[
            "ao6_runtime_contract_sha256"
        ],
        "source_context_scope": "one-full-frame",
        "production_default_changed": False,
    }
    component_payloads = {
        "physical-chain-4000dpi": physical,
        "neutral-axis-gauge": gauge_payload,
        "srgb-oetf": oetf,
        "ao6-source-context-display-look": ao6,
    }
    component_hashes = {
        name: _payload_sha256(payload)
        for name, payload in component_payloads.items()
    }
    identity = _payload_sha256(
        {
            "components": component_hashes,
            "reference_sampling_dpi": int(
                config["reference_sampling_dpi"]
            ),
            "preview_policy": config["preview_policy"],
        }
    )
    bundle = FilmProfileBundle(
        profile_id=f"generic-p7f-4000-{identity[:16]}",
        claim_level=config["profile"]["claim_level"],
        stock_id=config["profile"]["stock_id"],
        process_id=config["profile"]["process_id"],
        scanner_profile_id=config["profile"]["scanner_profile_id"],
        evidence_manifest_sha256=config["parent_decision_sha256"],
        components=(
            ComponentBinding(
                "physical-chain-4000dpi",
                physical["schema"],
                component_hashes["physical-chain-4000dpi"],
                PhysicalDomain.SCENE_LINEAR,
                PhysicalDomain.SCAN_LINEAR,
            ),
            ComponentBinding(
                "neutral-axis-gauge",
                gauge_payload["schema"],
                component_hashes["neutral-axis-gauge"],
                PhysicalDomain.SCAN_LINEAR,
                PhysicalDomain.DISPLAY_LINEAR,
            ),
            ComponentBinding(
                "srgb-oetf",
                oetf["schema"],
                component_hashes["srgb-oetf"],
                PhysicalDomain.DISPLAY_LINEAR,
                PhysicalDomain.DISPLAY_RGB,
            ),
            ComponentBinding(
                "ao6-source-context-display-look",
                ao6["schema"],
                component_hashes[
                    "ao6-source-context-display-look"
                ],
                PhysicalDomain.DISPLAY_RGB,
                PhysicalDomain.DISPLAY_RGB,
            ),
        ),
    )
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "film_profile_bundle": bundle.to_dict(),
        "bundle_sha256": bundle.bundle_sha256,
        "reference_sampling_dpi": int(
            config["reference_sampling_dpi"]
        ),
        "preview_policy": config["preview_policy"],
        "component_payloads": component_payloads,
        "claim_ceiling": config["claim_ceiling"],
    }
    validate_profile_artifact(artifact)
    return artifact


def validate_profile_artifact(
    artifact: dict[str, Any]
) -> FilmProfileBundle:
    if set(artifact) != {
        "schema",
        "film_profile_bundle",
        "bundle_sha256",
        "reference_sampling_dpi",
        "preview_policy",
        "component_payloads",
        "claim_ceiling",
    } or artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("compiled profile artifact fields drift")
    bundle = FilmProfileBundle.from_dict(
        artifact["film_profile_bundle"]
    )
    if (
        bundle.bundle_sha256 != artifact["bundle_sha256"]
        or bundle.claim_level != "generic-physical-inspired"
        or {
            bundle.stock_id,
            bundle.process_id,
            bundle.scanner_profile_id,
        }
        != {"unknown"}
        or int(artifact["reference_sampling_dpi"]) != 4000
        or artifact["preview_policy"].get(
            "independent_low_resolution_equivalence_claim_allowed"
        )
    ):
        raise ValueError("compiled profile identity or claim drift")
    payloads = artifact["component_payloads"]
    if not isinstance(payloads, dict) or set(payloads) != {
        component.component_id for component in bundle.components
    }:
        raise ValueError("compiled component inventory drift")
    for component in bundle.components:
        if _payload_sha256(payloads[component.component_id]) != component.sha256:
            raise ValueError("compiled component hash drift")
    physical = payloads["physical-chain-4000dpi"]
    if (
        physical["schema"]
        != "neuro_film.compiled_physical_chain_component.v1"
        or int(physical["reference_sampling_dpi"]) != 4000
    ):
        raise ValueError("compiled physical component drift")
    _spatial_from_payload(physical["spatial_profile"])
    SensitometryPrintOperator.from_dict(physical["print_operator"])
    _adjacency_from_payload(physical["adjacency"])
    gauge = payloads["neutral-axis-gauge"]
    if (
        gauge["base_component_sha256"]
        != next(
            component.sha256
            for component in bundle.components
            if component.component_id == "physical-chain-4000dpi"
        )
        or len(gauge["inverse_neutral_splines"]) != 3
    ):
        raise ValueError("compiled gauge base drift")
    tuple(
        RationalQuadraticSpline.from_dict(item)
        for item in gauge["inverse_neutral_splines"]
    )
    return bundle


def reconstruct_runtime(
    artifact: dict[str, Any], runtime: Any
) -> tuple[Any, NeutralAxisGaugeOperator]:
    validate_profile_artifact(artifact)
    payloads = artifact["component_payloads"]
    physical = payloads["physical-chain-4000dpi"]
    print_operator = SensitometryPrintOperator.from_dict(
        physical["print_operator"]
    )
    rebuilt = replace(
        runtime,
        profile=_spatial_from_payload(physical["spatial_profile"]),
        print_operator=print_operator,
        apply_adjacency=_adjacency_from_payload(
            physical["adjacency"]
        ),
    )
    splines = tuple(
        RationalQuadraticSpline.from_dict(item)
        for item in payloads["neutral-axis-gauge"][
            "inverse_neutral_splines"
        ]
    )
    gauge = NeutralAxisGaugeOperator(
        print_operator,
        splines,  # type: ignore[arg-type]
    )
    return rebuilt, gauge


def evaluate_compiled_profile(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime, gauge = validate_contract(root, config)
    first = compile_profile_artifact(root=root, config=config)
    second = compile_profile_artifact(root=root, config=config)
    artifact_exact = _canonical_bytes(first) == _canonical_bytes(second)
    rebuilt_runtime, rebuilt_gauge = reconstruct_runtime(first, runtime)
    replay = config["replay"]
    sources = {
        "synthetic": np.random.default_rng(
            int(replay["synthetic_seed"])
        ).random(
            tuple(int(value) for value in replay["synthetic_shape"])
            + (3,)
        )
    }
    for sample_id in replay["real_ids"]:
        sources[sample_id] = _bounded_real_source(
            root,
            runtime,
            sample_id,
            max_long_edge=int(replay["real_max_long_edge"]),
        )
    rows: list[dict[str, Any]] = []
    replay_exact = True
    for sample_id, source in sources.items():
        expected = render_challenger(
            source,
            runtime,
            gauge,
            sampling_dpi=int(config["reference_sampling_dpi"]),
        )
        actual = render_challenger(
            source,
            rebuilt_runtime,
            rebuilt_gauge,
            sampling_dpi=int(config["reference_sampling_dpi"]),
        )
        exact = np.array_equal(expected, actual)
        replay_exact &= exact
        rows.append(
            {
                "sample_id": sample_id,
                "float_exact": exact,
                "maximum_absolute_error": float(
                    np.max(np.abs(expected - actual))
                ),
                "expected_float64_sha256": hashlib.sha256(
                    np.ascontiguousarray(expected).tobytes()
                ).hexdigest(),
                "actual_float64_sha256": hashlib.sha256(
                    np.ascontiguousarray(actual).tobytes()
                ).hexdigest(),
            }
        )
    core = {
        "schema": "neuro_film.u6_p8a_fixed_reference_profile_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "artifact": first,
        "artifact_canonical_sha256": _payload_sha256(first),
        "artifact_exact": artifact_exact,
        "bundle_roundtrip_exact": (
            FilmProfileBundle.from_dict(
                first["film_profile_bundle"]
            ).to_dict()
            == first["film_profile_bundle"]
        ),
        "replay_rows": rows,
        "replay_exact": replay_exact,
        "decision": (
            "pass"
            if artifact_exact and replay_exact
            else "fail"
        ),
    }
    stable_id = hashlib.sha256(
        _canonical_bytes(core)
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "compile_profile_artifact",
    "evaluate_compiled_profile",
    "reconstruct_runtime",
    "validate_contract",
    "validate_profile_artifact",
    "write_report",
]
