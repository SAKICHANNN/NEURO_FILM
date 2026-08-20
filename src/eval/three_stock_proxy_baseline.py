"""Three-stock K=1 proxy baseline on one frozen digital source population."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    load_guardrail_config,
    load_profile_values,
    style_transfer_rgb,
)

SCHEMA = "neuro-film.rf3-three-stock-proxy-baseline-contract.v1"
RESULT_SCHEMA = "neuro-film.rf3-three-stock-proxy-baseline-result.v1"


class ThreeStockProxyError(RuntimeError):
    """Raised when the frozen baseline contract or an artifact drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bound_path(root: Path, binding: dict[str, Any]) -> Path:
    path = root / str(binding["path"])
    if not path.is_file():
        raise ThreeStockProxyError(f"missing bound file: {binding['path']}")
    actual = _sha256(path)
    if actual != binding["sha256"]:
        raise ThreeStockProxyError(f"hash drift for {binding['path']}: {actual}")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreeStockProxyError(f"invalid JSON: {path}") from exc


def load_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    _validate_contract_payload(contract)
    return contract


def _validate_contract_payload(contract: Any) -> None:
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ThreeStockProxyError("unexpected three-stock contract schema")
    matrix = contract.get("stock_evidence_matrix")
    expected_stocks = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    if not isinstance(matrix, list) or [row.get("stock_id") for row in matrix] != expected_stocks:
        raise ThreeStockProxyError("stock evidence matrix must retain the frozen three-stock order")
    if any(row.get("target_closeness_evaluable") is not False for row in matrix):
        raise ThreeStockProxyError("target closeness must remain unavailable for every stock")
    comparison = contract.get("comparison", {})
    if comparison.get("target_closeness") != "not_evaluable_without_controlled_stock_targets":
        raise ThreeStockProxyError("target closeness claim was inflated")
    if comparison.get("stock_distinguishability") != "not_evaluable_without_controlled_stock_targets":
        raise ThreeStockProxyError("stock distinguishability claim was inflated")
    ao6 = contract.get("ao6_velvia_baseline", {})
    if ao6.get("allowed_stock_id") != "fujifilm_velvia_50":
        raise ThreeStockProxyError("AO6 is not bound exclusively to Velvia 50")
    if set(ao6.get("forbidden_stock_ids", [])) != {"kodak_portra_400", "kodak_ektar_100"}:
        raise ThreeStockProxyError("AO6 forbidden-stock boundary drifted")
    required_gates = {
        "minimum_pairwise_population_median_delta_e76",
        "minimum_ao6_vs_legacy_velvia_population_median_delta_e76",
        "maximum_new_output_boundary_fraction",
        "maximum_output_boundary_fraction",
    }
    if not required_gates.issubset(comparison):
        raise ThreeStockProxyError("required proxy or boundary gate is missing")
    if any(not np.isfinite(float(comparison[key])) for key in required_gates):
        raise ThreeStockProxyError("proxy and boundary gates must be finite")


def _validate_contract_bindings(contract: dict[str, Any], root: Path) -> None:
    for binding in contract["evidence_bindings"].values():
        path = _bound_path(root, binding)
        required_decision = binding.get("required_decision")
        if required_decision is not None:
            payload = _read_json(path)
            if payload.get("decision") != required_decision:
                raise ThreeStockProxyError(f"decision drift for {binding['path']}")


def _load_sources(contract: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = contract["source"]
    manifest_path = root / source["manifest"]
    if _sha256(manifest_path) != source["manifest_sha256"]:
        raise ThreeStockProxyError("source manifest hash drift")
    payload = _read_json(manifest_path)
    if not isinstance(payload, list):
        raise ThreeStockProxyError("source manifest must be a list")
    by_id = {row.get("id"): row for row in payload if isinstance(row, dict)}
    included = source["included_ids"]
    if len(included) != len(set(included)) or len(included) != source["expected_rows"]:
        raise ThreeStockProxyError("included source IDs are not an exact unique row set")
    rows: list[dict[str, Any]] = []
    for source_id in included:
        row = by_id.get(source_id)
        if row is None:
            raise ThreeStockProxyError(f"missing source row: {source_id}")
        for key, expected_key in (
            ("allowed_use", "required_allowed_use"),
            ("rights_scope", "required_rights_scope"),
            ("decoded_color_state", "required_color_state"),
        ):
            if row.get(key) != source[expected_key]:
                raise ThreeStockProxyError(f"source policy drift for {source_id}: {key}")
        decoded = root / row["decoded_path"]
        if _sha256(decoded) != row["decoded_sha256"]:
            raise ThreeStockProxyError(f"decoded source hash drift: {source_id}")
        rows.append(row)
    return rows


def _load_ao6_records(
    contract: dict[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], Path]:
    binding = contract["ao6_velvia_baseline"]
    manifest_path = root / binding["run_manifest"]
    if _sha256(manifest_path) != binding["run_manifest_sha256"]:
        raise ThreeStockProxyError("AO6 manifest hash drift")
    manifest = _read_json(manifest_path)
    records = [
        row
        for row in manifest.get("records", [])
        if row.get("candidate_id") == binding["candidate_id"]
    ]
    by_id = {row.get("sample_id"): row for row in records}
    if len(by_id) != len(records):
        raise ThreeStockProxyError("duplicate AO6 sample IDs")
    return by_id, manifest_path.parent


def _save_png_create_only(path: Path, array: np.ndarray) -> str:
    if path.exists():
        raise ThreeStockProxyError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array, mode="RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=6)
    try:
        with path.open("xb") as handle:
            handle.write(encoded.getbuffer())
    except FileExistsError as exc:
        raise ThreeStockProxyError(f"refusing to overwrite output: {path}") from exc
    return _sha256(path)


def _load_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ThreeStockProxyError(f"invalid RGB image: {path}")
    return rgb


def _delta_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = rgb2lab(left.astype(np.float32) / 255.0) - rgb2lab(right.astype(np.float32) / 255.0)
    distance = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32))
    return {
        "median_delta_e76": float(np.median(distance)),
        "p95_delta_e76": float(np.quantile(distance, 0.95)),
    }


def _boundary_metrics(source: np.ndarray, output: np.ndarray) -> dict[str, float]:
    output_boundary = (output == 0) | (output == 255)
    source_boundary = (source == 0) | (source == 255)
    return {
        "output_boundary_fraction": float(np.mean(output_boundary)),
        "new_output_boundary_fraction": float(np.mean(output_boundary & ~source_boundary)),
    }


def _style_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "strength": float(profile.get("strength", 0.55)),
        "luma_strength": float(profile.get("luma_strength", 0.35)),
        "grain": 0.0,
        "gamut_safe": bool(profile.get("gamut_safe", False)),
        "gamut_mode": profile.get("gamut_mode"),
        "tone_rolloff": float(profile.get("tone_rolloff", 0.0)),
        "shadow_floor_l": float(profile.get("shadow_floor_l", 1.0)),
        "highlight_ceiling_l": float(profile.get("highlight_ceiling_l", 99.0)),
        "preserve_luma_detail_strength": float(profile.get("preserve_luma_detail", 0.0)),
        "chroma_curve_strength": float(profile.get("chroma_curve_strength", 0.0)),
        "output_margin": int(profile.get("output_margin", 0)),
        "dither": 0.0,
    }


def evaluate(contract_path: Path, root: Path, output_dir: Path, order: str) -> dict[str, Any]:
    if order not in {"canonical", "reverse"}:
        raise ThreeStockProxyError("order must be canonical or reverse")
    contract = load_contract(contract_path)
    config_sha256 = _sha256(contract_path)
    _validate_contract_bindings(contract, root)
    sources = _load_sources(contract, root)
    ao6_records, ao6_root = _load_ao6_records(contract, root)

    legacy = contract["legacy_k1_operator"]
    stats_path = _bound_path(root, legacy["stats"])
    profile_path = _bound_path(root, legacy["profile"])
    guardrail_path = _bound_path(root, legacy["guardrails"])
    stats_doc = _read_json(stats_path)
    styles = stats_doc.get("styles", {})
    arms = legacy["arms"]
    if [arm["style"] for arm in arms] != ["velvia_50", "portra_400", "ektar_100"]:
        raise ThreeStockProxyError("legacy K=1 arm order drifted")

    processing_rows = list(reversed(sources)) if order == "reverse" else list(sources)
    report_rows: list[dict[str, Any]] = []
    for source_row in processing_rows:
        source_id = source_row["id"]
        source_rgb = _load_rgb8(root / source_row["decoded_path"])
        outputs: dict[str, np.ndarray] = {}
        inventory: dict[str, dict[str, Any]] = {}
        for arm in arms:
            style = arm["style"]
            profile = load_profile_values(profile_path, legacy["profile"]["preset"], style)
            guardrails = load_guardrail_config(guardrail_path, style)
            rendered = style_transfer_rgb(
                source_rgb.astype(np.float32) / 255.0,
                styles[style],
                style,
                seed=1729,
                guardrails=guardrails,
                **_style_kwargs(profile),
            )
            if not np.isfinite(rendered).all() or np.any((rendered < 0.0) | (rendered > 1.0)):
                raise ThreeStockProxyError(f"non-finite or unbounded render: {source_id}/{style}")
            rgb8 = np.rint(rendered * 255.0).astype(np.uint8)
            relative = Path("renders") / arm["arm_id"] / f"{source_id}.png"
            png_sha = _save_png_create_only(output_dir / relative, rgb8)
            reopened = _load_rgb8(output_dir / relative)
            if not np.array_equal(rgb8, reopened):
                raise ThreeStockProxyError(f"PNG readback drift: {source_id}/{style}")
            outputs[arm["arm_id"]] = rgb8
            inventory[arm["arm_id"]] = {
                "relative_path": relative.as_posix(),
                "png_sha256": png_sha,
                "pixel_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
                **_boundary_metrics(source_rgb, rgb8),
            }

        ao6_record = ao6_records.get(source_id)
        if ao6_record is None or ao6_record.get("decoded_source_sha256") != source_row["decoded_sha256"]:
            raise ThreeStockProxyError(f"AO6 source identity drift: {source_id}")
        ao6_source = ao6_root / ao6_record["output"]
        if _sha256(ao6_source) != ao6_record["output_sha256"]:
            raise ThreeStockProxyError(f"AO6 output hash drift: {source_id}")
        ao6_rgb = _load_rgb8(ao6_source)
        if ao6_rgb.shape != source_rgb.shape:
            raise ThreeStockProxyError(f"AO6 geometry drift: {source_id}")
        ao6_relative = Path("renders") / contract["ao6_velvia_baseline"]["arm_id"] / f"{source_id}.png"
        ao6_sha = _save_png_create_only(output_dir / ao6_relative, ao6_rgb)
        inventory[contract["ao6_velvia_baseline"]["arm_id"]] = {
            "relative_path": ao6_relative.as_posix(),
            "png_sha256": ao6_sha,
            "pixel_sha256": hashlib.sha256(ao6_rgb.tobytes()).hexdigest(),
            **_boundary_metrics(source_rgb, ao6_rgb),
        }

        velvia_id, portra_id, ektar_id = [arm["arm_id"] for arm in arms]
        comparisons = {
            "velvia_vs_portra": _delta_metrics(outputs[velvia_id], outputs[portra_id]),
            "velvia_vs_ektar": _delta_metrics(outputs[velvia_id], outputs[ektar_id]),
            "portra_vs_ektar": _delta_metrics(outputs[portra_id], outputs[ektar_id]),
            "ao6_vs_legacy_velvia": _delta_metrics(ao6_rgb, outputs[velvia_id]),
        }
        report_rows.append(
            {
                "source_id": source_id,
                "make": source_row["make"],
                "raw_sha256": source_row["raw_sha256"],
                "decoded_sha256": source_row["decoded_sha256"],
                "width": int(source_row["width"]),
                "height": int(source_row["height"]),
                "outputs": inventory,
                "comparisons": comparisons,
            }
        )

    report_rows.sort(key=lambda row: contract["source"]["included_ids"].index(row["source_id"]))
    comparison_names = ["velvia_vs_portra", "velvia_vs_ektar", "portra_vs_ektar"]
    population = {
        name: float(np.median([row["comparisons"][name]["median_delta_e76"] for row in report_rows]))
        for name in comparison_names
    }
    ao6_population = float(
        np.median([row["comparisons"]["ao6_vs_legacy_velvia"]["median_delta_e76"] for row in report_rows])
    )
    all_inventory = [entry for row in report_rows for entry in row["outputs"].values()]
    gates = contract["comparison"]
    failed: list[str] = []
    for name, value in population.items():
        if value < gates["minimum_pairwise_population_median_delta_e76"]:
            failed.append(f"minimum separation: {name}")
    if ao6_population < gates["minimum_ao6_vs_legacy_velvia_population_median_delta_e76"]:
        failed.append("minimum separation: ao6_vs_legacy_velvia")
    if max(entry["new_output_boundary_fraction"] for entry in all_inventory) > gates["maximum_new_output_boundary_fraction"]:
        failed.append("new output boundary")
    if max(entry["output_boundary_fraction"] for entry in all_inventory) > gates["maximum_output_boundary_fraction"]:
        failed.append("output boundary")

    automatic_pass = not failed
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": config_sha256,
        "order": order,
        "source_manifest_sha256": contract["source"]["manifest_sha256"],
        "source_count": len(report_rows),
        "stock_evidence_matrix": contract["stock_evidence_matrix"],
        "target_closeness": gates["target_closeness"],
        "stock_distinguishability": gates["stock_distinguishability"],
        "rows": report_rows,
        "aggregate": {
            "pairwise_median_of_source_medians_delta_e76": population,
            "ao6_vs_legacy_velvia_median_of_source_medians_delta_e76": ao6_population,
            "maximum_new_output_boundary_fraction": max(
                entry["new_output_boundary_fraction"] for entry in all_inventory
            ),
            "maximum_output_boundary_fraction": max(entry["output_boundary_fraction"] for entry in all_inventory),
            "failed_gates": failed,
            "automatic_pass": automatic_pass,
        },
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    science = dict(report)
    science.pop("order")
    report["scientific_identity"] = _canonical_sha256(science)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise ThreeStockProxyError(f"refusing to overwrite output: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    args = parser.parse_args()
    report = evaluate(args.contract, args.root.resolve(), args.output_dir, args.order)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "scientific_identity": report["scientific_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
