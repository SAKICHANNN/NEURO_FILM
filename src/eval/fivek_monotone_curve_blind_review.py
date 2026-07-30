"""Build deterministic blinded AO6 product-value comparison packs."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_residual,
)
from src.eval.fivek_monotone_channel_curve_development import (
    apply_curve_operator,
    validate_contract as validate_development_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_unseen_content_confirmation import load_srgb16


class FiveKMonotoneCurveBlindError(ValueError):
    """Raised when blind-review evidence or selection drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKMonotoneCurveBlindError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKMonotoneCurveBlindError("contract is not frozen")
    if (
        config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or config["comparison"].get("strength_or_parameter_change_allowed")
        or config["selection"].get("semantic_or_manual_selection_allowed")
        or not config["comparison"].get("candidate_labels_hidden")
    ):
        raise FiveKMonotoneCurveBlindError("blind boundary drift")
    parent = config["parent"]
    parent_config = _load_hashed_json(
        root, parent["config"], parent["config_sha256"]
    )
    decision = _load_hashed_json(
        root, parent["decision"], parent["decision_sha256"]
    )
    report = _load_hashed_json(
        root, parent["report"], parent["report_sha256"]
    )
    development = parent_config["development"]
    development_config = _load_hashed_json(
        root, development["config"], development["config_sha256"]
    )
    validated_development = validate_development_contract(
        root, development_config
    )
    confirmation = parent_config["confirmation"]
    manifest = _load_hashed_json(
        root, confirmation["manifest"], confirmation["manifest_sha256"]
    )
    if (
        decision.get("status") != parent["required_status"]
        or report.get("automatic_pass") is not True
        or len(report.get("population", {}).get("rows", [])) != 64
        or len(manifest.get("rows", [])) != 64
        or config["selection"]["rows"] != 16
        or len(config["selection"]["round_seeds"]) != 3
    ):
        raise FiveKMonotoneCurveBlindError("blind eligibility drift")
    return {
        "parent_config": parent_config,
        "report": report,
        "development_config": development_config,
        "validated_development": validated_development,
        "manifest": manifest,
    }


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select four fixed ranks from each signed-benefit quartile."""

    if len(rows) != 64:
        raise FiveKMonotoneCurveBlindError("expected 64 confirmation rows")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["ridge"]["neutral_rmse"])
            - float(row["global"]["neutral_rmse"]),
            row["pair_id"],
        ),
    )
    selected = []
    for start in range(0, 64, 16):
        stratum = ordered[start : start + 16]
        selected.extend(stratum[index] for index in (1, 5, 9, 13))
    return selected


def _tile(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return ImageOps.contain(Image.fromarray(pixels, mode="RGB"), size)


def build_packs(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    maximum_side: int = 1024,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    rows = select_rows(validated["report"]["population"]["rows"])
    evidence = {
        row["pair_id"]: row for row in validated["manifest"]["rows"]
    }
    development_config = validated["development_config"]
    knots = np.asarray(
        development_config["operator"]["knot_inputs"], dtype=np.float64
    )
    epsilon = float(
        development_config["parents"]["ay3"]["boundary_epsilon"]
    )
    renderer = build_fixed_ao6_renderer(
        validated["validated_development"]["safe_validated"][
            "fixed_config"
        ],
        validated["validated_development"]["safe_validated"][
            "fixed_validated"
        ],
    )
    rendered = {}
    for row in rows:
        pair_id = row["pair_id"]
        item = evidence[pair_id]
        source = load_srgb16(Path(item["source_path"]), maximum_side)
        target = load_srgb16(Path(item["target_path"]), maximum_side)
        candidates = {}
        for method in ("global", "ridge"):
            raw = apply_curve_operator(
                source,
                np.asarray(row[method]["parameters"]),
                knots=knots,
            )
            neutral, _ = apply_boundary_safe_residual(
                source, raw, boundary_epsilon=epsilon
            )
            candidates[method], _ = renderer(neutral)
        reference, _ = renderer(target)
        rendered[pair_id] = {
            "reference": reference,
            **candidates,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = (320, 210)
    header = 28
    private_rounds = []
    public_rounds = []
    for round_index, seed in enumerate(
        config["selection"]["round_seeds"], start=1
    ):
        rng = random.Random(int(seed))
        round_rows = list(rows)
        rng.shuffle(round_rows)
        sheet = Image.new(
            "RGB",
            (
                tile_size[0] * 3,
                (tile_size[1] + header) * len(round_rows),
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        private_items = []
        public_items = []
        for index, row in enumerate(round_rows, start=1):
            pair_id = row["pair_id"]
            ridge_slot = "A" if rng.random() < 0.5 else "B"
            global_slot = "B" if ridge_slot == "A" else "A"
            slots = {
                "reference": rendered[pair_id]["reference"],
                "A": rendered[pair_id][
                    "ridge" if ridge_slot == "A" else "global"
                ],
                "B": rendered[pair_id][
                    "ridge" if ridge_slot == "B" else "global"
                ],
            }
            blind_id = f"R{round_index}-{index:02d}"
            top = (index - 1) * (tile_size[1] + header)
            for column, label in enumerate(("reference", "A", "B")):
                tile = _tile(slots[label], tile_size)
                left = column * tile_size[0]
                sheet.paste(
                    tile,
                    (
                        left + (tile_size[0] - tile.width) // 2,
                        top + header + (tile_size[1] - tile.height) // 2,
                    ),
                )
                draw.text(
                    (left + 4, top + 4),
                    f"{blind_id} | {label}",
                    fill="black",
                )
            public_items.append({"blind_id": blind_id})
            private_items.append(
                {
                    "blind_id": blind_id,
                    "pair_id": pair_id,
                    "ridge_slot": ridge_slot,
                    "global_slot": global_slot,
                }
            )
        image_path = output_dir / f"round_{round_index}.png"
        sheet.save(image_path, format="PNG", compress_level=6)
        public_path = output_dir / f"round_{round_index}_blind.json"
        public_path.write_bytes(
            _canonical_bytes(
                {
                    "round": round_index,
                    "seed": seed,
                    "items": public_items,
                    "criterion": config["comparison"]["criterion"],
                }
            )
        )
        public_rounds.append(
            {
                "round": round_index,
                "image": image_path.as_posix(),
                "image_sha256": _sha256(image_path),
                "blind_manifest": public_path.as_posix(),
                "blind_manifest_sha256": _sha256(public_path),
            }
        )
        private_rounds.append(
            {
                "round": round_index,
                "seed": seed,
                "items": private_items,
            }
        )
    key_path = output_dir / "private_key.json"
    key_path.write_bytes(
        _canonical_bytes(
            {
                "experiment_id": config["experiment_id"],
                "rounds": private_rounds,
            }
        )
    )
    pack = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "selected_pair_count": len(rows),
        "rounds": public_rounds,
        "private_key": key_path.as_posix(),
        "private_key_sha256": _sha256(key_path),
        "claim_ceiling": config["claim_ceiling"],
    }
    pack_path = output_dir / "pack.json"
    pack_path.write_bytes(_canonical_bytes(pack))
    return {
        "pack": pack,
        "pack_path": pack_path,
        "pack_sha256": _sha256(pack_path),
    }


__all__ = [
    "FiveKMonotoneCurveBlindError",
    "build_packs",
    "select_rows",
    "validate_contract",
]
