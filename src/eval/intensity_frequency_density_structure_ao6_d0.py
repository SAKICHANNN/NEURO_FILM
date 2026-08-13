"""Intensity-conditioned frequency-shaped common-density structure D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate as evaluate_base
from src.eval.sigmoid_scanner_ao6_value_d1 import load_contract as load_base_contract

SCHEMA = "neuro-film.u6-p4io-intensity-frequency-density-structure-ao6-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4io-intensity-frequency-density-structure-ao6-d0-result.v1"
_COMMON = float(np.linalg.norm(np.asarray((0.8, -0.55, 0.35))) / np.sqrt(3.0))


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IO contract")
    return value


def _builder(config: Mapping[str, Any]):
    mechanism = config["mechanism"]
    cutoff = float(mechanism["frequency_cutoff_cycles_per_pixel"])
    intervals = int(mechanism["intensity_intervals"])
    seed = int(mechanism["random_seed"])
    if (
        int(mechanism["frequency_cutoff_index"]) != 8
        or mechanism["frequency_cutoff_index_range"] != [2, 14]
        or cutoff != 0.125
        or intervals != 16
        or mechanism.get("hard_clipping_allowed")
        or mechanism.get("post_result_retuning_allowed")
    ):
        raise ValueError("P4IO mechanism drift")

    def build(
        base: np.ndarray,
        source: np.ndarray,
        index: int,
        amplitude: float,
    ) -> np.ndarray:
        height, width = base.shape[:2]
        rng = np.random.default_rng(seed + 1009 * index)
        noise = rng.standard_normal((height, width))
        fy = np.fft.fftfreq(height)[:, None]
        fx = np.fft.rfftfreq(width)[None, :]
        radius = np.sqrt(fx * fx + fy * fy)
        window = np.exp(-0.5 * np.square(radius / cutoff))
        window[0, 0] = 0.0
        field = np.fft.irfft2(np.fft.rfft2(noise) * window, s=(height, width))
        field -= float(np.mean(field))
        luma = source.astype(np.float64) @ np.asarray((0.2126, 0.7152, 0.0722))
        bins = np.minimum(np.floor(np.clip(luma, 0.0, 1.0) * intervals), intervals - 1)
        centers = (bins + 0.5) / intervals
        scaling = 4.0 * centers * (1.0 - centers)
        shaped = field * scaling
        shaped -= float(np.mean(shaped))
        shaped_rms = float(np.sqrt(np.mean(np.square(shaped))))
        y, x = np.mgrid[0:height, 0:width].astype(np.float64)
        reference = np.sin((x + 11 * index) / 23.0) * np.cos((y - 7 * index) / 19.0)
        target_rms = float(amplitude) * _COMMON * float(np.sqrt(np.mean(np.square(reference))))
        residual = shaped * (target_rms / shaped_rms)
        result = base.astype(np.float64) * np.power(10.0, -residual[..., None])
        return np.ascontiguousarray(result, dtype=np.float32)

    return build


def evaluate(
    contract: Mapping[str, Any],
    root: Path,
    *,
    contact_path: Path | None = None,
) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if _sha(parent_path) != contract["parent"]["sha256"]:
        raise ValueError("P4IO parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("P4IO parent decision drift")
    base_path = root / str(contract["base_contract"])
    result = evaluate_base(
        load_base_contract(base_path),
        root,
        contact_path=contact_path,
        structure_builder=_builder(contract),
    )
    passed = bool(result["automatic_pass"])
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "mechanism": contract["mechanism"],
        "rows": result["rows"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "automatic_pass": passed,
        "blind_review_allowed": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    if "contact_sheet_sha256" in result:
        core["contact_sheet_sha256"] = result["contact_sheet_sha256"]
    stable = {key: value for key, value in core.items() if key != "contact_sheet_sha256"}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

