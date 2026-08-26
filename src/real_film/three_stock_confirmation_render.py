"""SF3.A4 exact confirmation renders from a passing three-stock K=1 report."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.preprocess.output_encode import save_srgb16_png
from src.real_film.gold_transform_consistency import operator_from_dict
from src.real_film.three_stock_k1_baseline import REPORT_SCHEMA as K1_REPORT_SCHEMA
from src.real_film.three_stock_k1_baseline import (
    SINGLE_STOCK_REPORT_SCHEMA as SINGLE_STOCK_K1_REPORT_SCHEMA,
)
from src.real_film.three_stock_k1_file_runner import REPORT_SCHEMA as FILE_REPORT_SCHEMA
from src.real_film.three_stock_k1_file_runner import (
    SINGLE_STOCK_REPORT_SCHEMA as SINGLE_STOCK_FILE_REPORT_SCHEMA,
)
from src.real_film.three_stock_scan_integrity import decode_integer_rgb

CONTRACT_SCHEMA = "neuro-film.sf3-a4-three-stock-confirmation-render-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a4-three-stock-confirmation-render-report.v1"
SINGLE_STOCK_REPORT_SCHEMA = (
    "neuro-film.sf3-a4-single-stock-confirmation-render-report.v1"
)
SINGLE_STOCK_A2_DECISION = (
    "RETAIN_SINGLE_STOCK_K1_CANDIDATE_PENDING_THREE_STOCK_CONTROLS"
)


class ThreeStockConfirmationRenderError(ValueError):
    """Raised when SF3.A4 cannot produce an exact, target-blind render set."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockConfirmationRenderError(f"expected JSON object: {path}")
    return raw, value


def _repo_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockConfirmationRenderError(f"invalid {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockConfirmationRenderError(f"{field} must be repository-relative")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ThreeStockConfirmationRenderError(f"missing {field}: {value}")
    return path


def load_contract(path: Path, *, root: Path) -> tuple[bytes, dict[str, Any]]:
    raw, contract = _read_object(path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ThreeStockConfirmationRenderError("unsupported SF3.A4 contract")
    parent = contract.get("parent", {})
    binding = parent.get("k1_contract", {})
    k1_path = _repo_file(root, binding.get("path"), field="K1 contract")
    if _sha256_file(k1_path) != binding.get("sha256"):
        raise ThreeStockConfirmationRenderError("SF3.A2 contract hash drift")
    if parent.get("required_file_runner_schema") != FILE_REPORT_SCHEMA:
        raise ThreeStockConfirmationRenderError("file-runner schema drift")
    if parent.get("required_k1_report_schema") != K1_REPORT_SCHEMA:
        raise ThreeStockConfirmationRenderError("K1 report schema drift")
    if contract.get("required_stocks") != [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]:
        raise ThreeStockConfirmationRenderError("stock order drift")
    if contract.get("source", {}).get("film_target_file_reads_allowed") != 0:
        raise ThreeStockConfirmationRenderError("film target read gate drift")
    if contract.get("source", {}).get("operator_refits_allowed") != 0:
        raise ThreeStockConfirmationRenderError("operator refit gate drift")
    if int(contract.get("render", {}).get("row_tile_height", 0)) <= 0:
        raise ThreeStockConfirmationRenderError("invalid row tile height")
    if (
        float(contract.get("gates", {}).get("maximum_raw_output_clip_fraction", -1))
        != 0.0
    ):
        raise ThreeStockConfirmationRenderError("unclipped render gate drift")
    return raw, contract


def _validate_parent_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    ledger_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    parent = contract["parent"]
    result = report.get("k1_result")
    if report.get("schema") != parent["required_file_runner_schema"]:
        raise ThreeStockConfirmationRenderError("unsupported file-runner report")
    if (
        report.get("automatic_pass") is not True
        or report.get("decision") != parent["required_decision"]
    ):
        raise ThreeStockConfirmationRenderError("SF3.A2 did not open rendering")
    if (
        report.get("integrity_automatic_pass") is not True
        or report.get("paired_sampling_executed") is not True
        or not isinstance(report.get("operator_fits"), int)
        or report["operator_fits"] <= 0
    ):
        raise ThreeStockConfirmationRenderError("SF3.A2 execution chain is incomplete")
    if (
        report.get("ledger_sha256") != ledger_sha256
        or report.get("manifest_sha256") != manifest_sha256
    ):
        raise ThreeStockConfirmationRenderError("SF3.A2 input identity drift")
    if report.get("k1_contract_sha256") != parent["k1_contract"]["sha256"]:
        raise ThreeStockConfirmationRenderError("SF3.A2 contract identity drift")
    if (
        not isinstance(result, dict)
        or result.get("schema") != parent["required_k1_report_schema"]
    ):
        raise ThreeStockConfirmationRenderError("missing K1 scientific report")
    if (
        result.get("automatic_pass") is not True
        or result.get("decision") != parent["required_decision"]
    ):
        raise ThreeStockConfirmationRenderError("K1 scientific result did not pass")
    if (
        result.get("contract_sha256") != parent["k1_contract"]["sha256"]
        or result.get("selection_used_confirmation_targets") is not False
    ):
        raise ThreeStockConfirmationRenderError("K1 selection provenance drift")
    if set(result.get("stocks", {})) != set(contract["required_stocks"]):
        raise ThreeStockConfirmationRenderError("K1 stock inventory drift")
    if any(
        not isinstance(value, dict) or value.get("automatic_pass") is not True
        for value in result["stocks"].values()
    ):
        raise ThreeStockConfirmationRenderError(
            "a required stock did not pass K1 gates"
        )
    return result


def _validate_single_stock_parent_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    ledger_sha256: str,
    manifest_sha256: str,
    stock: str,
) -> Mapping[str, Any]:
    parent = contract["parent"]
    result = report.get("k1_result")
    if stock not in contract["required_stocks"]:
        raise ThreeStockConfirmationRenderError("unsupported single stock")
    if report.get("schema") != SINGLE_STOCK_FILE_REPORT_SCHEMA:
        raise ThreeStockConfirmationRenderError("unsupported single-stock file report")
    if (
        report.get("stock") != stock
        or report.get("automatic_pass") is not True
        or report.get("decision") != SINGLE_STOCK_A2_DECISION
        or report.get("cross_stock_controls_evaluated") is not False
    ):
        raise ThreeStockConfirmationRenderError("single-stock SF3.A2 did not open rendering")
    if (
        report.get("integrity_automatic_pass") is not True
        or report.get("paired_sampling_executed") is not True
        or not isinstance(report.get("operator_fits"), int)
        or report["operator_fits"] <= 0
    ):
        raise ThreeStockConfirmationRenderError(
            "single-stock SF3.A2 execution chain is incomplete"
        )
    if (
        report.get("ledger_sha256") != ledger_sha256
        or report.get("manifest_sha256") != manifest_sha256
        or report.get("k1_contract_sha256") != parent["k1_contract"]["sha256"]
    ):
        raise ThreeStockConfirmationRenderError("single-stock SF3.A2 identity drift")
    if (
        not isinstance(result, dict)
        or result.get("schema") != SINGLE_STOCK_K1_REPORT_SCHEMA
        or result.get("stock") != stock
        or result.get("automatic_pass") is not True
        or result.get("decision") != SINGLE_STOCK_A2_DECISION
        or result.get("selection_used_confirmation_targets") is not False
        or result.get("wrong_stock_control_evaluated") is not False
        or result.get("cross_stock_distinguishability_evaluated") is not False
        or result.get("contract_sha256") != parent["k1_contract"]["sha256"]
        or not isinstance(result.get("operator"), dict)
    ):
        raise ThreeStockConfirmationRenderError(
            "single-stock K1 scientific result did not pass"
        )
    return result


def _confirmation_sources(
    *,
    root: Path,
    ledger: Mapping[str, Any],
    manifest: Mapping[str, Any],
    stocks: list[str],
) -> list[dict[str, Any]]:
    ledger_rows = ledger.get("rows")
    manifest_rows = manifest.get("rows")
    if (
        not isinstance(ledger_rows, list)
        or not isinstance(manifest_rows, list)
        or len(ledger_rows) != len(manifest_rows)
    ):
        raise ThreeStockConfirmationRenderError("ledger/manifest inventory drift")
    by_scene: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for source, row in zip(ledger_rows, manifest_rows, strict=True):
        if not isinstance(source, dict) or not isinstance(row, dict):
            raise ThreeStockConfirmationRenderError("ledger/manifest row type drift")
        if row.get("role") == "confirmation":
            by_scene.setdefault(str(row.get("scene_id", "")), []).append((source, row))
    if not by_scene or "" in by_scene:
        raise ThreeStockConfirmationRenderError("confirmation scenes are missing")
    result: list[dict[str, Any]] = []
    for scene_id in sorted(by_scene):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scene_id):
            raise ThreeStockConfirmationRenderError("unsafe confirmation scene id")
        rows = by_scene[scene_id]
        observed_stocks = {str(row["stock_id"]) for _, row in rows}
        if observed_stocks != set(stocks):
            raise ThreeStockConfirmationRenderError("confirmation stock coverage drift")
        paths = {str(source.get("digital_reference_path", "")) for source, _ in rows}
        hashes = {str(row.get("digital_reference_sha256", "")) for _, row in rows}
        if len(paths) != 1 or len(hashes) != 1:
            raise ThreeStockConfirmationRenderError(
                "confirmation source differs across stocks"
            )
        path_value = paths.pop()
        digest = hashes.pop()
        relative = Path(path_value)
        if not relative.parts or relative.parts[0].casefold() != "data":
            raise ThreeStockConfirmationRenderError(
                "digital reference must use logical data root"
            )
        path = _repo_file(root, path_value, field="digital reference")
        if len(digest) != 64 or _sha256_file(path) != digest:
            raise ThreeStockConfirmationRenderError("digital reference hash drift")
        result.append({"scene_id": scene_id, "path": path, "sha256": digest})
    return result


def _render_operator(
    source: np.ndarray, operator_payload: Mapping[str, Any], *, tile_rows: int
) -> tuple[np.ndarray, dict[str, Any]]:
    maximum = float(np.iinfo(source.dtype).max)
    normalized = source.astype(np.float64) / maximum
    output = np.empty(source.shape, dtype=np.float32)
    operator = operator_from_dict(operator_payload)
    minimum = float("inf")
    maximum_value = float("-inf")
    outside = 0
    total = int(source.size)
    for top in range(0, source.shape[0], tile_rows):
        bottom = min(source.shape[0], top + tile_rows)
        raw = np.asarray(
            operator.apply(normalized[top:bottom].reshape(-1, 3)), dtype=np.float64
        ).reshape(bottom - top, source.shape[1], 3)
        if not np.isfinite(raw).all():
            raise ThreeStockConfirmationRenderError(
                "operator produced non-finite output"
            )
        minimum = min(minimum, float(np.min(raw)))
        maximum_value = max(maximum_value, float(np.max(raw)))
        outside += int(np.count_nonzero((raw < 0.0) | (raw > 1.0)))
        output[top:bottom] = raw.astype(np.float32)
    clip_fraction = outside / total
    if clip_fraction != 0.0:
        raise ThreeStockConfirmationRenderError("unclipped SF3.A4 output left [0,1]")
    return output, {
        "raw_minimum": minimum,
        "raw_maximum": maximum_value,
        "raw_output_clip_fraction": clip_fraction,
    }


def _save_verified_png(output: np.ndarray, path: Path) -> dict[str, Any]:
    expected = np.rint(np.clip(output, 0.0, 1.0) * 65535.0).astype(np.uint16)
    save_srgb16_png(output, path)
    decoded_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded_bgr is None
        or decoded_bgr.dtype != np.uint16
        or decoded_bgr.shape != expected.shape
    ):
        raise ThreeStockConfirmationRenderError(
            "RGB16 PNG readback shape/type mismatch"
        )
    decoded = np.ascontiguousarray(decoded_bgr[..., ::-1])
    if not np.array_equal(decoded, expected):
        raise ThreeStockConfirmationRenderError("RGB16 PNG readback samples differ")
    with Image.open(path) as image:
        if not image.info.get("icc_profile"):
            raise ThreeStockConfirmationRenderError("RGB16 PNG lacks embedded ICC")
    return {
        "png_sha256": _sha256_file(path),
        "uint16_sample_sha256": _sha256(expected.tobytes()),
        "width": int(expected.shape[1]),
        "height": int(expected.shape[0]),
    }


def evaluate_and_materialize(
    contract_path: Path,
    *,
    root: Path,
    a2_report_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    output_dir: Path,
    stock: str | None = None,
) -> dict[str, Any]:
    """Create exact confirmation renders without reading any film target file.

    ``stock`` selects the lower-claim single-stock path.  The default preserves
    the original complete three-stock behavior.
    """

    contract_raw, contract = load_contract(contract_path, root=root)
    report_raw, a2_report = _read_object(a2_report_path)
    ledger_raw, ledger = _read_object(ledger_path)
    manifest_raw, manifest = _read_object(manifest_path)
    if stock is None:
        k1_result = _validate_parent_report(
            a2_report,
            contract=contract,
            ledger_sha256=_sha256(ledger_raw),
            manifest_sha256=_sha256(manifest_raw),
        )
        stocks = list(contract["required_stocks"])
    else:
        k1_result = _validate_single_stock_parent_report(
            a2_report,
            contract=contract,
            ledger_sha256=_sha256(ledger_raw),
            manifest_sha256=_sha256(manifest_raw),
            stock=stock,
        )
        stocks = [stock]
    sources = _confirmation_sources(
        root=root, ledger=ledger, manifest=manifest, stocks=stocks
    )
    if stock is None:
        if sorted(
            str(value) for value in k1_result.get("common_confirmation_scenes", [])
        ) != [row["scene_id"] for row in sources]:
            raise ThreeStockConfirmationRenderError(
                "confirmation scene identity drift"
            )
    elif (
        not isinstance(
            k1_result.get("metrics", {}).get("confirmation_frames"), int
        )
        or k1_result["metrics"]["confirmation_frames"] < len(sources)
    ):
        raise ThreeStockConfirmationRenderError(
            "single-stock confirmation frames do not cover rendered scenes"
        )
    logical_output_root = (root / "outputs").resolve()
    output_dir = output_dir.resolve()
    if output_dir == logical_output_root or not output_dir.is_relative_to(
        logical_output_root
    ):
        raise ThreeStockConfirmationRenderError(
            "output directory must be below logical outputs root"
        )
    if output_dir.exists():
        raise ThreeStockConfirmationRenderError("output directory already exists")
    stage = output_dir.with_name(f".{output_dir.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    k1_contract = json.loads(
        _repo_file(
            root, contract["parent"]["k1_contract"]["path"], field="K1 contract"
        ).read_text(encoding="utf-8")
    )
    integrity_contract = json.loads(
        _repo_file(
            root,
            k1_contract["parent"]["integrity_contract"]["path"],
            field="integrity contract",
        ).read_text(encoding="utf-8")
    )
    decode_contract = integrity_contract["decode"]
    output_rows: list[dict[str, Any]] = []
    source_decodes = 0
    try:
        for source_row in sources:
            source = decode_integer_rgb(source_row["path"], decode_contract)
            source_decodes += 1
            scene_hashes: set[str] = set()
            for stock_id in stocks:
                operator_payload = (
                    k1_result["stocks"][stock_id]["operator"]
                    if stock is None
                    else k1_result["operator"]
                )
                rendered, diagnostics = _render_operator(
                    source,
                    operator_payload,
                    tile_rows=int(contract["render"]["row_tile_height"]),
                )
                relative = Path(stock_id) / f"{source_row['scene_id']}.png"
                facts = _save_verified_png(rendered, stage / relative)
                if facts["png_sha256"] in scene_hashes:
                    raise ThreeStockConfirmationRenderError(
                        "stock output bytes are identical within scene"
                    )
                scene_hashes.add(facts["png_sha256"])
                output_rows.append(
                    {
                        "scene_id": source_row["scene_id"],
                        "stock_id": stock_id,
                        "digital_reference_sha256": source_row["sha256"],
                        "operator_kind": operator_payload["kind"],
                        "relative_path": relative.as_posix(),
                        **diagnostics,
                        **facts,
                    }
                )
                del rendered
        core = {
            "schema": REPORT_SCHEMA if stock is None else SINGLE_STOCK_REPORT_SCHEMA,
            "experiment_id": (
                contract["experiment_id"]
                if stock is None
                else f"{contract['experiment_id']}.{stock}"
            ),
            "contract_sha256": _sha256(contract_raw),
            "a2_report_sha256": _sha256(report_raw),
            "ledger_sha256": _sha256(ledger_raw),
            "manifest_sha256": _sha256(manifest_raw),
            "confirmation_scenes": [row["scene_id"] for row in sources],
            "stocks": stocks,
            "digital_source_decodes": source_decodes,
            "film_target_file_reads": 0,
            "operator_refits": 0,
            "outputs": output_rows,
            "automatic_pass": True,
            "decision": (
                contract["decision_if_pass"]
                if stock is None
                else "OPEN_SINGLE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_ONLY_PENDING_THREE_STOCK_CONTROLS"
            ),
            "claim_ceiling": (
                contract["claim_ceiling"]
                if stock is None
                else (
                    "Exact full-resolution confirmation-source render materialization "
                    "for one independently fitted controlled K=1 stock-labelled "
                    "operator. Passing opens severe-artifact review for that stock "
                    "only; it is not wrong-stock rejection, stock distinguishability, "
                    "calibrated stock response, preference, product promotion or "
                    "multi-stock completion."
                )
            ),
        }
        if stock is not None:
            core["stock"] = stock
            core["cross_stock_controls_evaluated"] = False
        report = {**core, "stable_evidence_id": _sha256(_canonical(core))}
        (stage / "report.json").write_bytes(_canonical(report))
        os.rename(stage, output_dir)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def evaluate_single_stock_and_materialize(
    contract_path: Path,
    *,
    root: Path,
    a2_report_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    output_dir: Path,
    stock: str,
) -> dict[str, Any]:
    """Render one passing stock lane for severe-artifact review only."""

    return evaluate_and_materialize(
        contract_path,
        root=root,
        a2_report_path=a2_report_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stock=stock,
    )


__all__ = [
    "CONTRACT_SCHEMA",
    "REPORT_SCHEMA",
    "SINGLE_STOCK_REPORT_SCHEMA",
    "ThreeStockConfirmationRenderError",
    "evaluate_and_materialize",
    "evaluate_single_stock_and_materialize",
    "load_contract",
]
