"""U6.P8C canonical artifact and WorkingImage ingress evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.density_witness_frontier import encoded_srgb_to_linear
from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_invariance import _bounded_real_source
from src.film_physics.profile_consumer import (
    _canonical_bytes,
    _payload_sha256,
    compile_standalone_profile_artifact,
    load_standalone_profile_artifact_bytes,
    render_working_image,
    serialize_standalone_profile_artifact,
    validate_contract as validate_p8b_contract,
)
from src.preprocess.types import SourceProfile, WorkingImage


SCHEMA = "neuro_film.u6_p8c_profile_ingress_receipt_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    ingress = config["ingress"]
    execution = config["execution"]
    if (
        config.get("schema") != SCHEMA
        or ingress
        != {
            "working_space": "linear_srgb_d65",
            "transfer_state": "scene_linear",
            "minimum_scene_value": 0.0,
            "maximum_scene_value": 1.0,
            "unmapped_headroom_allowed": False,
            "unknown_color_state_allowed": False,
        }
        or not execution["canonical_artifact_bytes_required"]
        or not execution["repeat_output_and_receipt_exact_required"]
        or execution["final_quantization_allowed"]
        or execution["post_result_retuning_allowed"]
    ):
        raise ValueError("unsupported U6.P8C contract")
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
    if (
        not decision["next_leaf"].startswith("U6.P8C")
        or decision["artifact_canonical_sha256"]
        != config["required_parent_artifact_sha256"]
        or decision["production_default_changed"]
        or decision["native_runtime_opened"]
    ):
        raise ValueError("U6.P8C parent decision drift")
    _, runtime, _ = validate_p8b_contract(root, parent)
    return parent, runtime


def _working(pixels: np.ndarray, sample_id: str) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_srgb_d65",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile(
            "raw_metadata", "U6.P8C bounded replay"
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path(f"{sample_id}.scene-linear"),
    )


def evaluate_profile_ingress(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    parent, reference_runtime = validate_contract(root, config)
    artifact = compile_standalone_profile_artifact(
        root=root, config=parent
    )
    if _payload_sha256(artifact) != config[
        "required_parent_artifact_sha256"
    ]:
        raise ValueError("U6.P8B artifact drift")
    raw = serialize_standalone_profile_artifact(artifact)
    raw_sha = hashlib.sha256(raw).hexdigest()
    loaded = load_standalone_profile_artifact_bytes(
        raw, expected_sha256=raw_sha
    )
    replay = config["replay"]
    sources = {
        "synthetic": np.random.default_rng(
            int(replay["synthetic_seed"])
        ).random(
            tuple(int(value) for value in replay["synthetic_shape"])
            + (3,),
            dtype=np.float32,
        )
    }
    for sample_id in replay["real_ids"]:
        encoded = _bounded_real_source(
            root,
            reference_runtime,
            sample_id,
            max_long_edge=int(replay["real_max_long_edge"]),
        )
        sources[sample_id] = encoded_srgb_to_linear(encoded).astype(
            np.float32
        )
    rows = []
    repeat_exact = True
    for sample_id, pixels in sources.items():
        working = _working(pixels, sample_id)
        first, first_receipt = render_working_image(loaded, working)
        second, second_receipt = render_working_image(loaded, working)
        exact = np.array_equal(first, second)
        receipt_exact = first_receipt == second_receipt
        repeat_exact &= exact and receipt_exact
        rows.append(
            {
                "sample_id": sample_id,
                "output_exact": exact,
                "receipt_exact": receipt_exact,
                "output_float64_sha256": hashlib.sha256(
                    np.ascontiguousarray(first).tobytes()
                ).hexdigest(),
                "receipt_sha256": first_receipt["receipt_sha256"],
                "maximum_scene_value": float(np.max(pixels)),
            }
        )
    core = {
        "schema": "neuro_film.u6_p8c_profile_ingress_receipt_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "artifact_canonical_sha256": _payload_sha256(artifact),
        "artifact_bytes_sha256": raw_sha,
        "artifact_bytes": len(raw),
        "artifact_canonical_roundtrip_exact": (
            _canonical_bytes(loaded) == raw
        ),
        "render_rows": rows,
        "repeat_output_and_receipt_exact": repeat_exact,
        "final_quantization_performed": False,
        "decision": "pass" if repeat_exact else "fail",
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


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
    "evaluate_profile_ingress",
    "validate_contract",
    "write_report",
]
