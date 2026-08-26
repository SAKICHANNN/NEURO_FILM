"""Compare existing Portra/Ektar looks with one mature external control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import load_render_profile, render_three_stock_look_rgb

SCHEMA = "neuro-film.rf3-d7-portra-ektar-external-alignment-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d7-portra-ektar-external-alignment-result.v1"


class PortraEktarExternalAlignmentError(ValueError):
    """Raised when the frozen RF3.D7 inputs or contract drift."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise PortraEktarExternalAlignmentError("comparison geometry differs")
    values: list[np.ndarray] = []
    for y0 in range(0, left.shape[0], 128):
        left_lab = rgb2lab(left[y0 : y0 + 128]).astype(np.float32)
        right_lab = rgb2lab(right[y0 : y0 + 128]).astype(np.float32)
        delta = np.sqrt(np.sum(np.square(left_lab - right_lab), axis=2))
        values.append(delta.reshape(-1))
    return float(np.median(np.concatenate(values)))


def _decode_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb8 = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(rgb8.astype(np.float32) / 255.0)


def evaluate(*, root: Path, contract_path: Path, reverse: bool) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA or contract.get("experiment_id") != "RF3.D7":
        raise PortraEktarExternalAlignmentError("unsupported RF3.D7 contract")
    sources = contract["sources"]
    for row in sources.values():
        path = root / row["path"]
        if _sha256_file(path) != row["sha256"]:
            raise PortraEktarExternalAlignmentError(f"source identity drift: {path}")

    external_report = json.loads(
        (root / sources["external_report"]["path"]).read_text(encoding="utf-8")
    )
    records = external_report.get("records")
    if not isinstance(records, list):
        raise PortraEktarExternalAlignmentError("external record inventory is invalid")
    candidate_ids = sources["external_report"]["candidate_ids"]
    sample_ids = list(sources["gold_set"]["sample_ids"])
    stock_ids = list(contract["execution"]["film_stock_ids"])
    if reverse:
        sample_ids.reverse()
        stock_ids.reverse()

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for stock_id, candidate_id in candidate_ids.items():
            if record.get("candidate_id") == candidate_id:
                key = (str(record.get("sample_id")), stock_id)
                if key in by_key:
                    raise PortraEktarExternalAlignmentError(
                        "duplicate external candidate row"
                    )
                by_key[key] = record
    expected = {
        (sample_id, stock_id) for sample_id in sample_ids for stock_id in stock_ids
    }
    if set(by_key) & expected != expected:
        raise PortraEktarExternalAlignmentError(
            "external candidate inventory is incomplete"
        )

    profile = load_render_profile(
        root / sources["render_profile"]["path"], root=root
    )
    statistics = json.loads(
        (root / sources["style_statistics"]["path"]).read_text(encoding="utf-8")
    )["styles"]
    rows: list[dict[str, Any]] = []
    per_sample: dict[str, dict[str, bool]] = {}
    for sample_id in sample_ids:
        representative = by_key[(sample_id, stock_ids[0])]
        source_path = root / representative["source_path"]
        if _sha256_file(source_path) != representative["source_sha256"]:
            raise PortraEktarExternalAlignmentError(
                f"source hash drift: {sample_id}"
            )
        height, width, channels = (int(value) for value in representative["shape"])
        if channels != 3:
            raise PortraEktarExternalAlignmentError("external output is not RGB")
        source = _decode_rgb(source_path, width=width, height=height)
        external: dict[str, np.ndarray] = {}
        for stock_id in stock_ids:
            record = by_key[(sample_id, stock_id)]
            output_path = root / record["output_path"]
            if _sha256_file(output_path) != record["output_sha256"]:
                raise PortraEktarExternalAlignmentError(
                    f"external output hash drift: {sample_id}/{stock_id}"
                )
            external[stock_id] = _decode_rgb(
                output_path, width=width, height=height
            )

        current: dict[str, np.ndarray] = {}
        for stock_id in stock_ids:
            style = "portra_400" if stock_id == "kodak_portra_400" else "ektar_100"
            current[stock_id] = render_three_stock_look_rgb(
                source,
                profile=profile,
                film_stock_id=stock_id,
                look_amount=float(contract["execution"]["look_amount"]),
                style_statistics=statistics[style],
                guardrails=load_guardrail_config(
                    root / sources["guardrails"]["path"], style
                ),
                seed=int(contract["execution"]["seed"]),
                tile_size=int(contract["execution"]["tile_size"]),
            )

        sample_wins: dict[str, bool] = {}
        for stock_id in stock_ids:
            other_id = next(value for value in stock_ids if value != stock_id)
            same = _median_delta_e76(current[stock_id], external[stock_id])
            cross = _median_delta_e76(current[stock_id], external[other_id])
            sample_wins[stock_id] = same < cross
            output = current[stock_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "film_stock_id": stock_id,
                    "same_name_median_delta_e76": same,
                    "cross_name_median_delta_e76": cross,
                    "same_name_wins": same < cross,
                    "current_output_sha256": hashlib.sha256(
                        output.tobytes()
                    ).hexdigest(),
                    "current_boundary_fraction": float(
                        np.mean((output <= 0.0) | (output >= 1.0))
                    ),
                }
            )
        per_sample[sample_id] = sample_wins
        rows.append(
            {
                "sample_id": sample_id,
                "comparison": "pair_separation",
                "current_portra_ektar_median_delta_e76": _median_delta_e76(
                    current["kodak_portra_400"], current["kodak_ektar_100"]
                ),
                "external_portra_ektar_median_delta_e76": _median_delta_e76(
                    external["kodak_portra_400"], external["kodak_ektar_100"]
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["sample_id"]),
            str(row.get("film_stock_id", row.get("comparison"))),
        )
    )
    stock_rows = [row for row in rows if "film_stock_id" in row]
    same_wins = sum(bool(row["same_name_wins"]) for row in stock_rows)
    both_wins = sum(all(values.values()) for values in per_sample.values())
    maximum_boundary = max(
        float(row["current_boundary_fraction"]) for row in stock_rows
    )
    gates_cfg = contract["gates"]
    gates = {
        "all_sources_and_external_outputs_rehash": True,
        "minimum_same_name_arm_wins": same_wins
        >= int(gates_cfg["minimum_same_name_arm_wins_of_18"]),
        "minimum_samples_with_both_same_name_wins": both_wins
        >= int(gates_cfg["minimum_samples_with_both_same_name_wins_of_9"]),
        "maximum_current_output_boundary_fraction": maximum_boundary
        <= float(gates_cfg["maximum_current_output_boundary_fraction"]),
    }
    scientific: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "RF3.D7",
        "contract_sha256": _sha256_file(contract_path),
        "rows": rows,
        "summary": {
            "same_name_arm_wins": same_wins,
            "samples_with_both_same_name_wins": both_wins,
            "maximum_current_output_boundary_fraction": maximum_boundary,
        },
        "gates": gates,
        "decision": (
            contract["decision_if_pass"]
            if all(gates.values())
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_id"] = _stable_id(scientific)
    return scientific


__all__ = [
    "PortraEktarExternalAlignmentError",
    "evaluate",
]
