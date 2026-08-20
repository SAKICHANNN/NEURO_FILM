"""Strict FGAesQ scoring and consumed-preference concordance for U5.R2FGAESQ0.

The scorer phase is deliberately blind to private mappings and direct decisions.
The aggregation phase consumes only the frozen score lock plus those private facts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def _resolve_within(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"asset escapes root: {relative}")
    return resolved


def _decode_png(path: Path) -> tuple[Image.Image, str]:
    if path.suffix.lower() != ".png":
        raise ValueError(f"only PNG is permitted: {path}")
    with Image.open(path) as opened:
        if opened.format != "PNG":
            raise ValueError(f"extension/format mismatch: {path} ({opened.format})")
        opened.load()
        image = opened.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"empty image: {path}")
    return image, _sha256(path)


def _encoded_to_linear(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.float64)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def image_statistic_controls(image: Image.Image) -> dict[str, float]:
    encoded = np.asarray(image, dtype=np.float64) / 255.0
    linear = _encoded_to_linear(encoded)
    lms = linear @ _SRGB_TO_LMS.T
    oklab = np.cbrt(np.maximum(lms, 0.0)) @ _LMS_TO_OKLAB.T
    chroma = np.hypot(oklab[..., 1], oklab[..., 2])
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(luminance - np.mean(luminance)))))
    values = {
        "mean_oklab_chroma": float(np.mean(chroma)),
        "linear_srgb_rms_luminance_contrast": rms,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite statistic control")
    return values


def load_public_round_one(
    *,
    dataset_id: str,
    root: Path,
    expected_labels: Sequence[str],
    expected_sources: int,
) -> list[dict[str, Any]]:
    manifest_path = root / "public_manifest.json"
    manifest = _load_json(manifest_path)
    rows = [row for row in manifest.get("rows", []) if row.get("round") == 1]
    if len(rows) != expected_sources:
        raise ValueError(f"{dataset_id}: expected {expected_sources} round-one rows")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        presentation_id = str(row["presentation_id"])
        if presentation_id in seen:
            raise ValueError(f"duplicate presentation: {dataset_id}/{presentation_id}")
        seen.add(presentation_id)
        labels = [str(label) for label in row["labels"]]
        if labels != list(expected_labels):
            raise ValueError(f"unexpected labels: {dataset_id}/{presentation_id}")
        assets = row["assets"]
        decoded: dict[str, Image.Image] = {}
        hashes: dict[str, str] = {}
        paths: dict[str, str] = {}
        statistics: dict[str, dict[str, float]] = {}
        for label in labels:
            relative = str(assets[label])
            path = _resolve_within(root, relative)
            image, asset_sha = _decode_png(path)
            decoded[label] = image
            hashes[label] = asset_sha
            paths[label] = relative.replace("\\", "/")
            statistics[label] = image_statistic_controls(image)
        output.append(
            {
                "dataset_id": dataset_id,
                "presentation_id": presentation_id,
                "role": str(row["role"]),
                "labels": labels,
                "paths": paths,
                "hashes": hashes,
                "statistics": statistics,
                "images": decoded,
            }
        )
    return output


def official_confidence_adjustment(
    pair_scores: Mapping[str, Sequence[float]], *, item_count: int
) -> dict[str, float]:
    all_scores = [float(score) for scores in pair_scores.values() for score in scores]
    if not all_scores:
        raise ValueError("no pair scores")
    series_mean = float(np.mean(np.asarray(all_scores, dtype=np.float64)))
    result: dict[str, float] = {}
    for label, scores in pair_scores.items():
        if not scores:
            raise ValueError(f"no scores for label {label}")
        raw = float(np.mean(np.asarray(scores, dtype=np.float64)))
        count = len(scores)
        if item_count >= 4:
            if count == 1:
                adjusted = 0.3 * raw + 0.7 * series_mean
            elif count == 2:
                adjusted = 0.7 * raw + 0.3 * series_mean
            else:
                adjusted = 0.9 * raw + 0.1 * series_mean
        else:
            adjusted = raw
        result[label] = float(adjusted)
    return result


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> None:
    import torch

    _seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _import_external_modules(external_root: Path) -> tuple[Any, type[Any]]:
    inference_root = external_root / "FGAesQ_Inference"
    if not (inference_root / "utils" / "FGAesQ.py").is_file():
        raise FileNotFoundError(f"invalid FGAesQ source root: {external_root}")
    path = str(inference_root.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    fgaesq_module = importlib.import_module("utils.FGAesQ")
    diff_module = importlib.import_module("utils.DiffToken")
    return fgaesq_module, diff_module.DiffToken


def load_strict_model(
    *, external_root: Path, model_path: Path, clip_path: Path, device: str
) -> tuple[Any, type[Any], dict[str, Any]]:
    import clip
    import torch

    if (
        _sha256(clip_path)
        != "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
    ):
        raise ValueError("CLIP asset SHA-256 mismatch")
    fgaesq_module, diff_token_class = _import_external_modules(external_root)
    original_clip_load = clip.load

    def local_clip_load(name: str, device: str = "cpu", **kwargs: Any) -> Any:
        if name != "ViT-B/16":
            raise ValueError(f"unexpected CLIP model request: {name}")
        return original_clip_load(str(clip_path), device=device, jit=False)

    clip.load = local_clip_load
    fgaesq_module.clip.load = local_clip_load
    try:
        model = fgaesq_module.FGAesQ(pretrained_path=None)
    finally:
        clip.load = original_clip_load
        fgaesq_module.clip.load = original_clip_load
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("FGAesQ checkpoint is not a state mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("FGAesQ state_dict has an unexpected structure")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "FGAesQ strict-load incompatibility: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    facts = {
        "checkpoint_sha256": _sha256(model_path),
        "checkpoint_bytes": model_path.stat().st_size,
        "clip_sha256": _sha256(clip_path),
        "clip_bytes": clip_path.stat().st_size,
        "state_tensor_count": len(state),
        "state_scalar_count": int(sum(value.numel() for value in state.values())),
        "missing_keys": [],
        "unexpected_keys": [],
    }
    return model, diff_token_class, facts


def strict_similarity_pair(
    preprocessor: Any,
    image1: Image.Image,
    image2: Image.Image,
    *,
    pair_id: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    preprocessor._setup_processing_state(pair_id)
    clip1, clip2, original1, original2 = preprocessor.prepare_dual_tensors(
        image1, image2
    )
    importance = preprocessor.calculate_pair_importance(original1, original2)
    scores1, types1 = preprocessor.apply_pair_importance_to_image(
        importance, is_img1=True
    )
    preprocessor._current_importance = scores1
    preprocessor._current_patch_types = types1
    patches1, pos1, mask1 = preprocessor.prepare_patches(clip1)
    scores2, types2 = preprocessor.apply_pair_importance_to_image(
        importance, is_img1=False
    )
    preprocessor._current_importance = scores2
    preprocessor._current_patch_types = types2
    patches2, pos2, mask2 = preprocessor.prepare_patches(clip2)
    return patches1, pos1, mask1, patches2, pos2, mask2


def _model_score(
    model: Any, patch: Any, position: Any, mask: Any, device: str
) -> float:
    import torch

    with torch.inference_mode():
        score = model(
            patch.unsqueeze(0).to(device),
            position.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )
    value = float(score.reshape(-1)[0].detach().cpu())
    if not math.isfinite(value):
        raise ValueError("non-finite FGAesQ score")
    return value


def score_series(
    *,
    model: Any,
    diff_token_class: type[Any],
    row: Mapping[str, Any],
    label_order: Sequence[str],
    device: str,
) -> dict[str, float]:
    preprocessor = diff_token_class(
        clip_model=model.clip_model, patch_selection="similarity"
    )
    accum: dict[str, list[float]] = defaultdict(list)
    for left, right in itertools.combinations(label_order, 2):
        canonical_pair = "|".join(sorted((left, right)))
        pair_id = f"{row['dataset_id']}|{row['presentation_id']}|{canonical_pair}"
        tensors = strict_similarity_pair(
            preprocessor,
            row["images"][left],
            row["images"][right],
            pair_id=pair_id,
        )
        left_score = _model_score(model, tensors[0], tensors[1], tensors[2], device)
        right_score = _model_score(model, tensors[3], tensors[4], tensors[5], device)
        accum[left].append(left_score)
        accum[right].append(right_score)
    return official_confidence_adjustment(accum, item_count=len(label_order))


def score_single_controls(
    *,
    model: Any,
    diff_token_class: type[Any],
    row: Mapping[str, Any],
    device: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for label in row["labels"]:
        scores: list[float] = []
        asset_sha = row["hashes"][label]
        for repeat in range(3):
            seed_bytes = hashlib.sha256(
                f"u5-r2fgaesq0-single|{asset_sha}|{repeat}".encode("ascii")
            ).digest()[:4]
            seed = int.from_bytes(seed_bytes, "big")
            _seed_all(seed)
            preprocessor = diff_token_class(
                clip_model=model.clip_model, patch_selection="random"
            )
            patch, position, mask = preprocessor.process_image(row["images"][label])
            scores.append(_model_score(model, patch, position, mask, device))
        output[label] = float(np.mean(np.asarray(scores, dtype=np.float64)))
    return output


def build_score_lock(
    *,
    config_path: Path,
    external_root: Path,
    model_path: Path,
    clip_path: Path,
    p401_root: Path,
    p402_root: Path,
    device: str,
) -> dict[str, Any]:
    config = _load_json(config_path)
    expected_commit = config["external_asset"]["repository_commit"]
    head_path = external_root / ".git"
    if not head_path.exists():
        raise ValueError("external source is not a Git checkout")
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(external_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    non_cache_entries = [
        entry
        for entry in porcelain
        if "__pycache__/" not in entry.replace("\\", "/")
        and not entry.replace("\\", "/").endswith(".pyc")
    ]
    if head != expected_commit or non_cache_entries:
        raise ValueError("external FGAesQ checkout identity mismatch or dirty state")
    external_asset = config["external_asset"]
    if (
        model_path.stat().st_size != external_asset["model_size_bytes"]
        or _sha256(model_path) != external_asset["model_sha256"]
        or clip_path.stat().st_size != external_asset["clip_size_bytes"]
        or _sha256(clip_path) != external_asset["clip_sha256"]
    ):
        raise ValueError("frozen FGAesQ or CLIP asset identity mismatch")
    configure_determinism(int(config["strict_mechanics"]["python_numpy_torch_seed"]))
    model, diff_token_class, asset_facts = load_strict_model(
        external_root=external_root,
        model_path=model_path,
        clip_path=clip_path,
        device=device,
    )
    datasets = [
        load_public_round_one(
            dataset_id="p401",
            root=p401_root,
            expected_labels=("A", "B"),
            expected_sources=12,
        ),
        load_public_round_one(
            dataset_id="p402",
            root=p402_root,
            expected_labels=("A", "B", "C", "D"),
            expected_sources=12,
        ),
    ]
    score_rows: list[dict[str, Any]] = []
    for rows in datasets:
        for row in rows:
            labels = list(row["labels"])
            canonical = score_series(
                model=model,
                diff_token_class=diff_token_class,
                row=row,
                label_order=labels,
                device=device,
            )
            reversed_scores = score_series(
                model=model,
                diff_token_class=diff_token_class,
                row=row,
                label_order=list(reversed(labels)),
                device=device,
            )
            single = score_single_controls(
                model=model,
                diff_token_class=diff_token_class,
                row=row,
                device=device,
            )
            score_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "presentation_id": row["presentation_id"],
                    "role": row["role"],
                    "asset_paths": row["paths"],
                    "asset_sha256": row["hashes"],
                    "series_scores": canonical,
                    "reversed_series_scores": reversed_scores,
                    "single_scores": single,
                    "statistic_controls": row["statistics"],
                }
            )
    result = {
        "schema": "neuro-film.u5-r2fgaesq0-score-lock.v1",
        "experiment_id": "U5.R2FGAESQ0",
        "config_sha256": _sha256(config_path),
        "external_repository_commit": head,
        "external_generated_cache_entries": porcelain,
        "asset_facts": asset_facts,
        "public_manifest_sha256": {
            "p401": _sha256(p401_root / "public_manifest.json"),
            "p402": _sha256(p402_root / "public_manifest.json"),
        },
        "private_mapping_reads": 0,
        "direct_result_reads": 0,
        "rows": score_rows,
    }
    return result


def _sign(score_left: float, score_right: float, epsilon: float = 1e-6) -> int:
    difference = float(score_left) - float(score_right)
    if abs(difference) <= epsilon:
        return 0
    return 1 if difference > 0 else -1


def _truth_winner(row: Mapping[str, Any], left: str, right: str) -> str:
    left_wins = right in row["pairwise_winners"][left]
    right_wins = left in row["pairwise_winners"][right]
    if left_wins == right_wins:
        raise ValueError(f"non-binary frozen P402 truth: {left}/{right}")
    return left if left_wins else right


def _control_score(row: Mapping[str, Any], control: str, label: str) -> float:
    if control == "series":
        return float(row["series_scores"][label])
    if control == "single":
        return float(row["single_scores"][label])
    if control in {"mean_oklab_chroma", "linear_srgb_rms_luminance_contrast"}:
        return float(row["statistic_controls"][label][control])
    raise KeyError(control)


def aggregate_score_lock(
    *,
    config_path: Path,
    score_lock_paths: Sequence[Path],
    p401_mapping_path: Path,
    p401_result_path: Path,
    p402_mapping_path: Path,
    p402_result_path: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    locks = [_load_json(path) for path in score_lock_paths]
    if len(locks) != 2:
        raise ValueError("exactly two fresh-process score locks are required")
    for lock in locks:
        if lock["config_sha256"] != _sha256(config_path):
            raise ValueError("score lock/config mismatch")
        if lock["private_mapping_reads"] != 0 or lock["direct_result_reads"] != 0:
            raise ValueError("score lock violated blind execution order")
    rows_by_lock = [
        {(row["dataset_id"], row["presentation_id"]): row for row in lock["rows"]}
        for lock in locks
    ]
    if rows_by_lock[0].keys() != rows_by_lock[1].keys():
        raise ValueError("fresh-process row inventory mismatch")
    replay_max = 0.0
    replay_ranks_exact = True
    for key in sorted(rows_by_lock[0]):
        first = rows_by_lock[0][key]
        second = rows_by_lock[1][key]
        labels = sorted(first["series_scores"])
        for field in ("series_scores", "reversed_series_scores", "single_scores"):
            for label in labels:
                replay_max = max(
                    replay_max,
                    abs(float(first[field][label]) - float(second[field][label])),
                )
            first_rank = sorted(labels, key=lambda label: (-first[field][label], label))
            second_rank = sorted(
                labels, key=lambda label: (-second[field][label], label)
            )
            replay_ranks_exact = replay_ranks_exact and first_rank == second_rank
    p401_mapping = _load_json(p401_mapping_path)
    p401_result = _load_json(p401_result_path)
    p402_mapping = _load_json(p402_mapping_path)
    p402_result = _load_json(p402_result_path)
    expected = config["consumed_inputs"]
    bindings = {
        "p401_mapping": _sha256(p401_mapping_path),
        "p401_result": _sha256(p401_result_path),
        "p402_mapping": _sha256(p402_mapping_path),
        "p402_result": _sha256(p402_result_path),
    }
    if bindings != {
        "p401_mapping": expected["p401"]["private_mapping_sha256"],
        "p401_result": expected["p401"]["formal_report_sha256"],
        "p402_mapping": expected["p402"]["private_mapping_sha256"],
        "p402_result": expected["p402"]["formal_result_sha256"],
    }:
        raise ValueError("private/direct result binding mismatch")
    primary_rows = rows_by_lock[0]
    p401_map = {
        row["source_id"]: row for row in p401_mapping["rows"] if row["round"] == 1
    }
    p401_agreement = 0
    p401_decisive = 0
    p401_tie_margins: list[float] = []
    for truth in p401_result["source_results"]:
        mapping = p401_map[truth["source_id"]]
        score_row = primary_rows[("p401", mapping["presentation_id"])]
        variant_to_label = {
            variant: label for label, variant in mapping["label_to_variant"].items()
        }
        candidate = float(score_row["series_scores"][variant_to_label["candidate"]])
        identity = float(score_row["series_scores"][variant_to_label["identity"]])
        decision = truth["preference_decision"]
        if decision == "tie":
            p401_tie_margins.append(abs(candidate - identity))
            continue
        p401_decisive += 1
        pair_sign = _sign(candidate, identity)
        predicted = (
            "candidate" if pair_sign > 0 else "identity" if pair_sign < 0 else None
        )
        p401_agreement += int(predicted == decision)
    p402_map = {
        row["source_id"]: row for row in p402_mapping["rows"] if row["round"] == 1
    }
    controls = (
        "series",
        "single",
        "mean_oklab_chroma",
        "linear_srgb_rms_luminance_contrast",
    )
    concordance = {control: 0 for control in controls}
    transform_identity = {control: 0 for control in controls}
    role_concordance: dict[str, int] = defaultdict(int)
    total_pairs = 0
    unique_top1 = 0
    unique_total = 0
    direct_rows = {row["source_id"]: row for row in p402_result["rows"]}
    truth_pairs: list[tuple[str, str, str, str]] = []
    for source_id, mapping in sorted(p402_map.items()):
        result_row = direct_rows[source_id]
        score_row = primary_rows[("p402", mapping["presentation_id"])]
        variant_to_label = {
            variant: label for label, variant in mapping["label_to_candidate"].items()
        }
        variants = sorted(variant_to_label)
        for left, right in itertools.combinations(variants, 2):
            winner = _truth_winner(result_row, left, right)
            truth_pairs.append((source_id, left, right, winner))
            total_pairs += 1
            for control in controls:
                left_score = _control_score(score_row, control, variant_to_label[left])
                right_score = _control_score(
                    score_row, control, variant_to_label[right]
                )
                pair_sign = _sign(left_score, right_score)
                predicted = left if pair_sign > 0 else right if pair_sign < 0 else None
                concordance[control] += int(predicted == winner)
            if "identity" in (left, right):
                transform = right if left == "identity" else left
                for control in controls:
                    transform_score = _control_score(
                        score_row, control, variant_to_label[transform]
                    )
                    identity_score = _control_score(
                        score_row, control, variant_to_label["identity"]
                    )
                    pair_sign = _sign(transform_score, identity_score)
                    predicted = (
                        transform
                        if pair_sign > 0
                        else "identity"
                        if pair_sign < 0
                        else None
                    )
                    transform_identity[control] += int(predicted == winner)
            series_left = _control_score(score_row, "series", variant_to_label[left])
            series_right = _control_score(score_row, "series", variant_to_label[right])
            pair_sign = _sign(series_left, series_right)
            series_predicted = (
                left if pair_sign > 0 else right if pair_sign < 0 else None
            )
            role_concordance[result_row["role"]] += int(series_predicted == winner)
        if result_row["unique_winner"] is not None:
            unique_total += 1
            ranked = sorted(
                variants,
                key=lambda variant: (
                    -_control_score(score_row, "series", variant_to_label[variant]),
                    variant,
                ),
            )
            top_margin = _control_score(
                score_row, "series", variant_to_label[ranked[0]]
            ) - _control_score(score_row, "series", variant_to_label[ranked[1]])
            predicted_top = ranked[0] if top_margin > 1e-6 else None
            unique_top1 += int(predicted_top == result_row["unique_winner"])
    if total_pairs != 72 or p401_decisive != 11 or unique_total != 11:
        raise ValueError("unexpected direct-truth inventory")
    derangement_counts: list[int] = []
    for shift in (1, 2, 3):
        count = 0
        for source_id, left, right, winner in truth_pairs:
            mapping = p402_map[source_id]
            score_row = primary_rows[("p402", mapping["presentation_id"])]
            variant_to_label = {
                variant: label
                for label, variant in mapping["label_to_candidate"].items()
            }
            variants = sorted(variant_to_label)
            shifted = {
                variant: variants[(index + shift) % len(variants)]
                for index, variant in enumerate(variants)
            }
            predicted_left = _control_score(
                score_row, "series", variant_to_label[shifted[left]]
            )
            predicted_right = _control_score(
                score_row, "series", variant_to_label[shifted[right]]
            )
            pair_sign = _sign(predicted_left, predicted_right)
            predicted = left if pair_sign > 0 else right if pair_sign < 0 else None
            count += int(predicted == winner)
        derangement_counts.append(count)
    order_sign_exact = True
    for row in primary_rows.values():
        labels = sorted(row["series_scores"])
        for left, right in itertools.combinations(labels, 2):
            order_sign_exact = order_sign_exact and (
                _sign(row["series_scores"][left], row["series_scores"][right])
                == _sign(
                    row["reversed_series_scores"][left],
                    row["reversed_series_scores"][right],
                )
            )
    gates_cfg = config["gates"]
    strongest_control = max(
        concordance["single"],
        concordance["mean_oklab_chroma"],
        concordance["linear_srgb_rms_luminance_contrast"],
    )
    gate_results = {
        "p402_all_pair_concordance": concordance["series"]
        >= gates_cfg["p402_all_pair_concordant_min"],
        "p402_transform_vs_identity": transform_identity["series"]
        >= gates_cfg["p402_transform_vs_identity_concordant_min"],
        "p402_unique_winner_top1": unique_top1
        >= gates_cfg["p402_unique_winner_top1_agreement_min"],
        "p402_each_content_role": all(
            value >= gates_cfg["p402_each_content_role_pairwise_concordant_min"]
            for value in role_concordance.values()
        )
        and len(role_concordance) == 3,
        "p401_decisive_agreement": p401_agreement
        >= gates_cfg["p401_decisive_agreement_min"],
        "gain_over_legitimate_controls": concordance["series"] - strongest_control
        >= gates_cfg["p402_gain_over_strongest_legitimate_control_min"],
        "gain_over_derangements": concordance["series"] - max(derangement_counts)
        >= gates_cfg["p402_gain_over_best_derangement_min"],
        "canonical_reverse_signs_exact": order_sign_exact,
        "fresh_process_scores": replay_max
        <= gates_cfg["fresh_process_score_max_abs_error_max"],
        "fresh_process_rankings": replay_ranks_exact,
    }
    passed = all(gate_results.values())
    return {
        "schema": "neuro-film.u5-r2fgaesq0-consumed-preference-concordance-result.v1",
        "experiment_id": "U5.R2FGAESQ0",
        "status": (
            "PASS_PRIVATE_CONSUMED_RETROSPECTIVE_CONCORDANCE"
            if passed
            else "FAIL_CLOSED_EXACT_FGAESQ_SERIES_SCORER"
        ),
        "config_sha256": _sha256(config_path),
        "score_lock_sha256": [_sha256(path) for path in score_lock_paths],
        "private_fact_bindings": bindings,
        "metrics": {
            "p402_concordance": concordance,
            "p402_transform_vs_identity": transform_identity,
            "p402_content_role_concordance": dict(sorted(role_concordance.items())),
            "p402_unique_winner_top1_agreement": unique_top1,
            "p402_unique_winner_total": unique_total,
            "p402_derangement_concordance": derangement_counts,
            "p401_decisive_agreement": p401_agreement,
            "p401_decisive_total": p401_decisive,
            "p401_tie_margins": p401_tie_margins,
            "canonical_reverse_pair_signs_exact": order_sign_exact,
            "fresh_process_score_max_abs_error": replay_max,
            "fresh_process_rankings_exact": replay_ranks_exact,
        },
        "gate_results": gate_results,
        "failed_gates": sorted(key for key, value in gate_results.items() if not value),
        "claim_ceiling": config["claim_ceiling"],
        "product_capability_opened": False,
    }


def audit_score_lock_mechanics(
    *, config_path: Path, score_lock_paths: Sequence[Path]
) -> dict[str, Any]:
    config = _load_json(config_path)
    locks = [_load_json(path) for path in score_lock_paths]
    if len(locks) != 2:
        raise ValueError("exactly two fresh-process score locks are required")
    config_sha = _sha256(config_path)
    if any(lock.get("config_sha256") != config_sha for lock in locks):
        raise ValueError("score lock/config mismatch")
    if any(
        lock.get("private_mapping_reads") != 0 or lock.get("direct_result_reads") != 0
        for lock in locks
    ):
        raise ValueError("score lock violated blind execution order")
    rows = [
        {(row["dataset_id"], row["presentation_id"]): row for row in lock["rows"]}
        for lock in locks
    ]
    if rows[0].keys() != rows[1].keys() or len(rows[0]) != 24:
        raise ValueError("fresh-process row inventory mismatch")
    finite = True
    replay_max = 0.0
    replay_rank_exact = True
    order_match = 0
    order_total = 0
    order_exact_rows = 0
    affected_rows: list[str] = []
    max_order_score_delta = 0.0
    for key in sorted(rows[0]):
        first = rows[0][key]
        second = rows[1][key]
        labels = sorted(first["series_scores"])
        row_order_exact = True
        for field in ("series_scores", "reversed_series_scores", "single_scores"):
            first_values = [float(first[field][label]) for label in labels]
            second_values = [float(second[field][label]) for label in labels]
            finite = finite and all(math.isfinite(value) for value in first_values)
            finite = finite and all(math.isfinite(value) for value in second_values)
            replay_max = max(
                replay_max,
                max(
                    abs(a - b) for a, b in zip(first_values, second_values, strict=True)
                ),
            )
            replay_rank_exact = replay_rank_exact and sorted(
                labels, key=lambda label: (-first[field][label], label)
            ) == sorted(labels, key=lambda label: (-second[field][label], label))
        for label in labels:
            max_order_score_delta = max(
                max_order_score_delta,
                abs(
                    float(first["series_scores"][label])
                    - float(first["reversed_series_scores"][label])
                ),
            )
        for left, right in itertools.combinations(labels, 2):
            canonical_sign = _sign(
                first["series_scores"][left], first["series_scores"][right]
            )
            reversed_sign = _sign(
                first["reversed_series_scores"][left],
                first["reversed_series_scores"][right],
            )
            matched = canonical_sign == reversed_sign
            order_total += 1
            order_match += int(matched)
            row_order_exact = row_order_exact and matched
        if row_order_exact:
            order_exact_rows += 1
        else:
            affected_rows.append(f"{key[0]}/{key[1]}")
    gate_results = {
        "complete_inventory": len(rows[0]) == 24 and order_total == 84,
        "all_scores_finite": finite,
        "canonical_reverse_pair_signs_exact": order_match == order_total,
        "fresh_process_score_max_abs_error": replay_max
        <= config["gates"]["fresh_process_score_max_abs_error_max"],
        "fresh_process_rankings_exact": replay_rank_exact,
        "zero_private_or_direct_reads": True,
    }
    mechanics_pass = all(gate_results.values())
    return {
        "schema": "neuro-film.u5-r2fgaesq0-mechanics-audit.v1",
        "experiment_id": "U5.R2FGAESQ0",
        "status": (
            "PASS_MECHANICS_READY_FOR_PRIVATE_AGGREGATION"
            if mechanics_pass
            else "INVALID_MECHANICS_ORDER_DEPENDENT_FGAESQ_SERIES"
        ),
        "config_sha256": config_sha,
        "score_lock_sha256": [_sha256(path) for path in score_lock_paths],
        "private_mapping_reads": 0,
        "direct_result_reads": 0,
        "metrics": {
            "row_count": len(rows[0]),
            "pair_sign_match_count": order_match,
            "pair_sign_total": order_total,
            "pair_sign_match_rate": order_match / order_total,
            "all_pair_signs_exact_row_count": order_exact_rows,
            "affected_rows": affected_rows,
            "maximum_canonical_reverse_score_delta": max_order_score_delta,
            "fresh_process_score_max_abs_error": replay_max,
            "fresh_process_rankings_exact": replay_rank_exact,
        },
        "gate_results": gate_results,
        "failed_gates": sorted(key for key, value in gate_results.items() if not value),
        "scientific_preference_metrics_computed": False,
        "claim_ceiling": (
            "Private consumed-asset mechanics audit of exact FGAesQ series scoring only; "
            "no preference concordance, candidate promotion, product, package, schema, "
            "capability or commercial claim."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--external-root", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--clip", type=Path, required=True)
    score.add_argument("--p401-root", type=Path, required=True)
    score.add_argument("--p402-root", type=Path, required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--score-lock", type=Path, action="append", required=True)
    aggregate.add_argument("--p401-mapping", type=Path, required=True)
    aggregate.add_argument("--p401-result", type=Path, required=True)
    aggregate.add_argument("--p402-mapping", type=Path, required=True)
    aggregate.add_argument("--p402-result", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit-mechanics")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--score-lock", type=Path, action="append", required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score":
        result = build_score_lock(
            config_path=args.config,
            external_root=args.external_root,
            model_path=args.model,
            clip_path=args.clip,
            p401_root=args.p401_root,
            p402_root=args.p402_root,
            device=args.device,
        )
    elif args.command == "aggregate":
        result = aggregate_score_lock(
            config_path=args.config,
            score_lock_paths=args.score_lock,
            p401_mapping_path=args.p401_mapping,
            p401_result_path=args.p401_result,
            p402_mapping_path=args.p402_mapping,
            p402_result_path=args.p402_result,
        )
    else:
        result = audit_score_lock_mechanics(
            config_path=args.config,
            score_lock_paths=args.score_lock,
        )
    _write_canonical_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
