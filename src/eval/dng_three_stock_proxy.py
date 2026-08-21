"""RF3.D0R exact DNG-to-display three-stock proxy evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    load_guardrail_config,
    load_profile_values,
    style_transfer_rgb,
)
from src.eval.three_stock_proxy_baseline import _style_kwargs
from src.film_physics.display_look import build_source_context_display_look_stages
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.preprocess.dng_forward_raster import load_dng_forward_working_image
from src.preprocess.ocio_aces2_output import apply_working_image_aces2_output
from src.preprocess.output_encode import save_srgb16_png

SCHEMA = "neuro-film.rf3-d0r-dng-three-stock-proxy-contract.v1"
RESULT_SCHEMA = "neuro-film.rf3-d0r-dng-three-stock-proxy-result.v1"


class DngThreeStockProxyError(RuntimeError):
    """Raised when a frozen RF3.D0R input or result drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DngThreeStockProxyError(f"expected JSON object: {path}")
    return value


def _bound(root: Path, binding: dict[str, Any]) -> Path:
    path = root / str(binding["path"])
    if not path.is_file() or _sha256(path) != binding["sha256"]:
        raise DngThreeStockProxyError(f"bound file drift: {binding['path']}")
    required = binding.get("required_decision")
    if required is not None and _load_json(path).get("decision") != required:
        raise DngThreeStockProxyError(f"bound decision drift: {binding['path']}")
    return path


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != SCHEMA:
        raise DngThreeStockProxyError("unsupported RF3.D0R contract")
    arms = value.get("arms")
    expected = [
        "legacy_unpaired_safe_lab_k1_velvia50",
        "legacy_unpaired_safe_lab_k1_portra400",
        "legacy_unpaired_safe_lab_k1_ektar100",
        "ao6_velvia50_display_proxy_t15_c35",
    ]
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != expected:
        raise DngThreeStockProxyError("RF3.D0R arm order drift")
    if (
        arms[-1].get("stock_id") != "fujifilm_velvia_50"
        or arms[-1].get("role") != "velvia_display_proxy_baseline_only"
    ):
        raise DngThreeStockProxyError("AO6 role drift")
    gates = value.get("gates", {})
    numeric = (
        "minimum_pairwise_population_median_delta_e76",
        "minimum_ao6_vs_legacy_velvia_population_median_delta_e76",
        "maximum_output_code_boundary_fraction",
        "maximum_new_output_code_boundary_fraction",
    )
    if any(key not in gates or not np.isfinite(float(gates[key])) for key in numeric):
        raise DngThreeStockProxyError("RF3.D0R gate missing or non-finite")
    for binding in value["bindings"].values():
        _bound(root, binding)
    return value


def _read_rgb16(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        value is None
        or value.dtype != np.uint16
        or value.ndim != 3
        or value.shape[2] != 3
    ):
        raise DngThreeStockProxyError(f"invalid RGB16 PNG: {path}")
    return np.ascontiguousarray(value[..., ::-1])


def _write_rgb16(path: Path, values: np.ndarray) -> tuple[np.ndarray, str]:
    if path.exists():
        raise DngThreeStockProxyError(f"refusing to overwrite {path}")
    save_srgb16_png(values, path)
    expected = np.rint(values * 65535.0).astype(np.uint16)
    reopened = _read_rgb16(path)
    if not np.array_equal(expected, reopened):
        raise DngThreeStockProxyError(f"RGB16 sample drift: {path}")
    return reopened, _sha256(path)


def _delta_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise DngThreeStockProxyError("comparison geometry drift")
    distances: list[np.ndarray] = []
    for start in range(0, left.shape[0], 128):
        a = rgb2lab(left[start : start + 128].astype(np.float32) / 65535.0)
        b = rgb2lab(right[start : start + 128].astype(np.float32) / 65535.0)
        distances.append(
            np.sqrt(np.sum((a - b) ** 2, axis=2, dtype=np.float32)).reshape(-1)
        )
    distance = np.concatenate(distances)
    return {
        "median_delta_e76": float(np.median(distance)),
        "p95_delta_e76": float(np.quantile(distance, 0.95)),
    }


def _boundary_metrics(source: np.ndarray, output: np.ndarray) -> dict[str, float]:
    source_boundary = (source == 0) | (source == 65535)
    output_boundary = (output == 0) | (output == 65535)
    return {
        "output_code_boundary_fraction": float(np.mean(output_boundary)),
        "new_output_code_boundary_fraction": float(
            np.mean(output_boundary & ~source_boundary)
        ),
    }


def evaluate(
    contract_path: Path, root: Path, output_dir: Path, order: str
) -> dict[str, Any]:
    if order not in {"canonical", "reverse"}:
        raise DngThreeStockProxyError("order must be canonical or reverse")
    contract = load_contract(contract_path, root)
    p98 = _load_json(_bound(root, contract["bindings"]["p98_contract"]))
    rows = list(p98["rows"])
    if (
        len(rows) != contract["input"]["expected_rows"]
        or len({row["camera_make"] for row in rows})
        != contract["input"]["expected_camera_makes"]
    ):
        raise DngThreeStockProxyError("P98 row or camera-make count drift")
    processing = list(reversed(rows)) if order == "reverse" else rows

    stats = _load_json(_bound(root, contract["bindings"]["legacy_stats"]))["styles"]
    profile_path = _bound(root, contract["bindings"]["legacy_profiles"])
    guardrail_path = _bound(root, contract["bindings"]["legacy_guardrails"])
    compiler = _load_json(_bound(root, contract["bindings"]["ao6_profile_compiler"]))
    artifact = compile_standalone_profile_artifact(root=root, config=compiler)
    ao6_payload = artifact["component_payloads"]["ao6-source-context-display-look"]
    arm_by_style = {row["style"]: row for row in contract["arms"] if "style" in row}
    report_rows: list[dict[str, Any]] = []
    input_failures: list[dict[str, Any]] = []

    for row in processing:
        working = load_dng_forward_working_image(
            root / row["logical_path"],
            expected_source_bytes=int(row["source_bytes"]),
            expected_source_sha256=row["source_sha256"],
        )
        if (
            working.working_space != contract["input"]["working_space"]
            or working.transfer_state != contract["input"]["transfer_state"]
        ):
            raise DngThreeStockProxyError("P98 WorkingImage boundary drift")
        display = np.ascontiguousarray(
            apply_working_image_aces2_output(
                working, contract["input"]["display_target"]
            ),
            dtype=np.float32,
        )
        finite = np.isfinite(display)
        below = finite & (display < 0.0)
        above = finite & (display > 1.0)
        if not finite.all() or np.any(below | above):
            finite_values = display[finite]
            input_failures.append(
                {
                    "source_id": row["source_id"],
                    "camera_make": row["camera_make"],
                    "source_sha256": row["source_sha256"],
                    "shape": list(display.shape),
                    "finite_values": int(np.count_nonzero(finite)),
                    "nonfinite_values": int(display.size - np.count_nonzero(finite)),
                    "below_zero_values": int(np.count_nonzero(below)),
                    "above_one_values": int(np.count_nonzero(above)),
                    "finite_minimum": float(np.min(finite_values))
                    if finite_values.size
                    else None,
                    "finite_maximum": float(np.max(finite_values))
                    if finite_values.size
                    else None,
                    "failure": "aces2_sdr_nonfinite_or_unbounded",
                }
            )
            del working, display
            continue
        source_code = np.rint(display * 65535.0).astype(np.uint16)
        rendered: dict[str, np.ndarray] = {}
        inventory: dict[str, Any] = {}
        for style in ("velvia_50", "portra_400", "ektar_100"):
            arm = arm_by_style[style]
            profile = load_profile_values(profile_path, "safe-rich", style)
            guardrails = load_guardrail_config(guardrail_path, style)
            values = np.ascontiguousarray(
                style_transfer_rgb(
                    display,
                    stats[style],
                    style,
                    seed=1729,
                    guardrails=guardrails,
                    **_style_kwargs(profile),
                ),
                dtype=np.float32,
            )
            if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                raise DngThreeStockProxyError(
                    f"legacy proxy is non-finite or unbounded: {style}"
                )
            relative = Path("renders") / arm["arm_id"] / f"{row['source_id']}.png"
            code, png_sha = _write_rgb16(output_dir / relative, values)
            rendered[arm["arm_id"]] = code
            inventory[arm["arm_id"]] = {
                "relative_path": relative.as_posix(),
                "png_sha256": png_sha,
                "pixel_sha256": hashlib.sha256(code.tobytes()).hexdigest(),
                **_boundary_metrics(source_code, code),
            }

        apply_base, apply_residual = build_source_context_display_look_stages(
            ao6_payload, display
        )
        ao6 = np.ascontiguousarray(
            apply_residual(apply_base(display)), dtype=np.float32
        )
        if not np.isfinite(ao6).all() or np.any((ao6 < 0.0) | (ao6 > 1.0)):
            raise DngThreeStockProxyError("AO6 is non-finite or unbounded")
        ao6_id = contract["arms"][-1]["arm_id"]
        relative = Path("renders") / ao6_id / f"{row['source_id']}.png"
        ao6_code, png_sha = _write_rgb16(output_dir / relative, ao6)
        rendered[ao6_id] = ao6_code
        inventory[ao6_id] = {
            "relative_path": relative.as_posix(),
            "png_sha256": png_sha,
            "pixel_sha256": hashlib.sha256(ao6_code.tobytes()).hexdigest(),
            **_boundary_metrics(source_code, ao6_code),
        }

        velvia, portra, ektar, _ = [item["arm_id"] for item in contract["arms"]]
        report_rows.append(
            {
                "source_id": row["source_id"],
                "camera_make": row["camera_make"],
                "source_sha256": row["source_sha256"],
                "shape": list(display.shape),
                "aces2_sdr_sha256": hashlib.sha256(display.tobytes()).hexdigest(),
                "outputs": inventory,
                "comparisons": {
                    "velvia_vs_portra": _delta_metrics(
                        rendered[velvia], rendered[portra]
                    ),
                    "velvia_vs_ektar": _delta_metrics(
                        rendered[velvia], rendered[ektar]
                    ),
                    "portra_vs_ektar": _delta_metrics(
                        rendered[portra], rendered[ektar]
                    ),
                    "ao6_vs_legacy_velvia": _delta_metrics(
                        rendered[ao6_id], rendered[velvia]
                    ),
                },
            }
        )
        del working, display, rendered

    canonical_ids = [row["source_id"] for row in rows]
    report_rows.sort(key=lambda item: canonical_ids.index(item["source_id"]))
    input_failures.sort(key=lambda item: canonical_ids.index(item["source_id"]))
    if not report_rows:
        raise DngThreeStockProxyError("no RF3.D0R input reached the comparison arms")
    pair_names = ("velvia_vs_portra", "velvia_vs_ektar", "portra_vs_ektar")
    pairwise = {
        name: float(
            np.median(
                [row["comparisons"][name]["median_delta_e76"] for row in report_rows]
            )
        )
        for name in pair_names
    }
    ao6_sep = float(
        np.median(
            [
                row["comparisons"]["ao6_vs_legacy_velvia"]["median_delta_e76"]
                for row in report_rows
            ]
        )
    )
    outputs = [value for row in report_rows for value in row["outputs"].values()]
    maximum_boundary = max(row["output_code_boundary_fraction"] for row in outputs)
    maximum_new_boundary = max(
        row["new_output_code_boundary_fraction"] for row in outputs
    )
    gates = contract["gates"]
    failed = [f"aces2_sdr_finite_bounded:{row['source_id']}" for row in input_failures]
    failed.extend(
        name
        for name, value in pairwise.items()
        if value < gates["minimum_pairwise_population_median_delta_e76"]
    )
    if ao6_sep < gates["minimum_ao6_vs_legacy_velvia_population_median_delta_e76"]:
        failed.append("ao6_vs_legacy_velvia")
    if maximum_boundary > gates["maximum_output_code_boundary_fraction"]:
        failed.append("output_code_boundary")
    if maximum_new_boundary > gates["maximum_new_output_code_boundary_fraction"]:
        failed.append("new_output_code_boundary")
    automatic_pass = not failed
    scientific = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": _sha256(contract_path),
        "expected_input_rows": int(contract["input"]["expected_rows"]),
        "rendered_input_rows": len(report_rows),
        "input_failures": input_failures,
        "rows": report_rows,
        "aggregate": {
            "pairwise_median_of_source_medians_delta_e76": pairwise,
            "ao6_vs_legacy_velvia_median_of_source_medians_delta_e76": ao6_sep,
            "maximum_output_code_boundary_fraction": maximum_boundary,
            "maximum_new_output_code_boundary_fraction": maximum_new_boundary,
            "failed_gates": failed,
            "automatic_pass": automatic_pass,
        },
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report = {
        **scientific,
        "execution_order": order,
        "stable_evidence_id": _canonical_sha(scientific),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise DngThreeStockProxyError(f"refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    args = parser.parse_args()
    report = evaluate(
        args.contract.resolve(),
        args.root.resolve(),
        args.output_dir.resolve(),
        args.order,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
