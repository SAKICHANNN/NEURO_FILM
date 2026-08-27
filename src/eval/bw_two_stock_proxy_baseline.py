"""Frozen HP5/Tri-X K=1 Look Approximation baseline."""

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

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values
from src.inference.style_safe_engine import render_resolved_safe_lab_rgb

SCHEMA = "neuro-film.bw2-hp5-trix-k1-baseline-contract.v1"
RESULT_SCHEMA = "neuro-film.bw2-hp5-trix-k1-baseline-result.v1"


class BwBaselineError(RuntimeError):
    """Raised when the frozen B&W baseline contract or an input drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BwBaselineError(f"invalid JSON: {path}") from exc


def load_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise BwBaselineError("unexpected B&W baseline schema")
    if [row.get("style") for row in contract.get("candidates", [])] != ["hp5", "tri_x_400"]:
        raise BwBaselineError("candidate order or identity drift")
    gates = contract.get("gates", {})
    required = {
        "maximum_new_output_boundary_fraction",
        "minimum_population_median_delta_e76",
        "minimum_per_source_median_delta_e76",
        "minimum_sources_passing_separation",
    }
    if not required.issubset(gates):
        raise BwBaselineError("required gate is missing")
    if contract.get("direction", {}).get("this_experiment_is_stock_completion") is not False:
        raise BwBaselineError("stock-completion claim boundary drift")
    return contract


def _bound(root: Path, binding: dict[str, Any]) -> Path:
    path = root / binding["path"]
    if not path.is_file() or _sha256(path) != binding["sha256"]:
        raise BwBaselineError(f"bound artifact drift: {binding['path']}")
    return path


def _load_sources(contract: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = contract["source"]
    manifest = root / source["manifest"]
    if _sha256(manifest) != source["manifest_sha256"]:
        raise BwBaselineError("source manifest drift")
    rows = _read_json(manifest)
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    included = source["included_ids"]
    if len(included) != source["expected_rows"] or len(set(included)) != len(included):
        raise BwBaselineError("source population drift")
    selected: list[dict[str, Any]] = []
    for source_id in included:
        row = by_id.get(source_id)
        if row is None:
            raise BwBaselineError(f"missing source: {source_id}")
        checks = (
            ("allowed_use", "required_allowed_use"),
            ("rights_scope", "required_rights_scope"),
            ("decoded_color_state", "required_color_state"),
        )
        if any(row.get(key) != source[expected] for key, expected in checks):
            raise BwBaselineError(f"source policy drift: {source_id}")
        decoded = root / row["decoded_path"]
        if _sha256(decoded) != row["decoded_sha256"]:
            raise BwBaselineError(f"decoded source drift: {source_id}")
        selected.append(row)
    return selected


def _load_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise BwBaselineError(f"invalid RGB image: {path}")
    return value


def _write_png_create_only(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(buffer, format="PNG", compress_level=6)
    try:
        with path.open("xb") as handle:
            handle.write(buffer.getbuffer())
    except FileExistsError as exc:
        raise BwBaselineError(f"refusing to overwrite: {path}") from exc
    return _sha256(path)


def _delta_e76(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    delta = rgb2lab(left.astype(np.float32) / 255.0) - rgb2lab(right.astype(np.float32) / 255.0)
    distance = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32))
    return float(np.median(distance)), float(np.quantile(distance, 0.95))


def evaluate(contract_path: Path, root: Path, output_dir: Path, order: str) -> dict[str, Any]:
    if order not in {"canonical", "reverse"}:
        raise BwBaselineError("order must be canonical or reverse")
    contract = load_contract(contract_path)
    config_sha = _sha256(contract_path)
    bindings = contract["bindings"]
    for binding in bindings.values():
        _bound(root, binding)
    stats = _read_json(root / bindings["stats"]["path"])["styles"]
    profile_path = root / bindings["profile"]["path"]
    guardrail_path = root / bindings["guardrails"]["path"]
    sources = _load_sources(contract, root)
    processing = list(reversed(sources)) if order == "reverse" else sources
    rows: list[dict[str, Any]] = []
    for source in processing:
        source_rgb = _load_rgb8(root / source["decoded_path"])
        outputs: dict[str, np.ndarray] = {}
        inventory: dict[str, Any] = {}
        for arm in contract["candidates"]:
            style = arm["style"]
            parameters = load_profile_values(profile_path, bindings["profile"]["preset"], style)
            parameters.update({"grain": 0.0, "dither": 0.0})
            rendered = render_resolved_safe_lab_rgb(
                source_rgb.astype(np.float32) / 255.0,
                style=style,
                style_statistics=stats[style],
                style_parameters=parameters,
                guardrails=load_guardrail_config(guardrail_path, style),
                seed=contract["render"]["seed"],
            )
            if not np.isfinite(rendered).all() or np.any((rendered < 0.0) | (rendered > 1.0)):
                raise BwBaselineError(f"unbounded render: {source['id']}/{style}")
            rgb8 = np.rint(rendered * 255.0).astype(np.uint8)
            neutral = bool(np.array_equal(rgb8[..., 0], rgb8[..., 1]) and np.array_equal(rgb8[..., 1], rgb8[..., 2]))
            source_boundary = (source_rgb == 0) | (source_rgb == 255)
            output_boundary = (rgb8 == 0) | (rgb8 == 255)
            relative = Path("renders") / arm["arm_id"] / f"{source['id']}.png"
            png_sha = _write_png_create_only(output_dir / relative, rgb8)
            if not np.array_equal(rgb8, _load_rgb8(output_dir / relative)):
                raise BwBaselineError(f"PNG readback drift: {source['id']}/{style}")
            outputs[arm["arm_id"]] = rgb8
            inventory[arm["arm_id"]] = {
                "relative_path": relative.as_posix(),
                "png_sha256": png_sha,
                "pixel_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
                "exact_rgb8_neutral_axis": neutral,
                "new_output_boundary_fraction": float(np.mean(output_boundary & ~source_boundary)),
            }
        left, right = (outputs[arm["arm_id"]] for arm in contract["candidates"])
        median, p95 = _delta_e76(left, right)
        rows.append(
            {
                "source_id": source["id"],
                "make": source["make"],
                "decoded_sha256": source["decoded_sha256"],
                "width": int(source["width"]),
                "height": int(source["height"]),
                "outputs": inventory,
                "hp5_vs_tri_x_400": {"median_delta_e76": median, "p95_delta_e76": p95},
            }
        )
    rows.sort(key=lambda row: contract["source"]["included_ids"].index(row["source_id"]))
    medians = [row["hp5_vs_tri_x_400"]["median_delta_e76"] for row in rows]
    inventory = [entry for row in rows for entry in row["outputs"].values()]
    gates = contract["gates"]
    aggregate = {
        "population_median_of_source_medians_delta_e76": float(np.median(medians)),
        "sources_passing_separation": sum(value >= gates["minimum_per_source_median_delta_e76"] for value in medians),
        "minimum_source_median_delta_e76": float(min(medians)),
        "maximum_new_output_boundary_fraction": max(entry["new_output_boundary_fraction"] for entry in inventory),
        "all_outputs_exact_rgb8_neutral_axis": all(entry["exact_rgb8_neutral_axis"] for entry in inventory),
    }
    failed: list[str] = []
    if aggregate["population_median_of_source_medians_delta_e76"] < gates["minimum_population_median_delta_e76"]:
        failed.append("population separation")
    if aggregate["sources_passing_separation"] < gates["minimum_sources_passing_separation"]:
        failed.append("per-source separation coverage")
    if aggregate["maximum_new_output_boundary_fraction"] > gates["maximum_new_output_boundary_fraction"]:
        failed.append("new output boundary")
    if not aggregate["all_outputs_exact_rgb8_neutral_axis"]:
        failed.append("neutral axis")
    aggregate["failed_gates"] = failed
    aggregate["automatic_pass"] = not failed
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": config_sha,
        "order": order,
        "source_manifest_sha256": contract["source"]["manifest_sha256"],
        "source_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "decision": contract["decision_if_pass"] if not failed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    science = dict(report)
    science.pop("order")
    report["scientific_identity"] = _canonical_sha256(science)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    try:
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise BwBaselineError(f"refusing to overwrite: {report_path}") from exc
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    args = parser.parse_args()
    report = evaluate(args.contract, args.root.resolve(), args.output_dir, args.order)
    print(json.dumps({"decision": report["decision"], "scientific_identity": report["scientific_identity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
