#!/usr/bin/env python3
"""Stream a known-look matrix through the fail-closed promotion gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (  # noqa: E402
    adjudicate_promotion,
    aggregate_known_operator_samples,
    canonical_sha256,
    evaluate_known_operator_batch,
    evaluate_recipe_batch_context_invariance,
    evaluate_recipe_batch_photographic_safety,
    fit_reference_look,
    render_reference_look,
)
from src.inference import sha256_file  # noqa: E402
from src.preprocess import load_working_image  # noqa: E402


REPORT_SCHEMA_ID = "neuro-film.reference-match-promotion-report.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one recipe per reference, evaluate every cross-content "
            "reference/source pair against known same-content targets, and "
            "run the photographic tail gate."
        )
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        required=True,
        help="Unique filename prefixes shared by all three directories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_prefix(directory: Path, sample_id: str) -> Path:
    if not directory.is_dir():
        raise ValueError(f"input directory does not exist: {directory}")
    matches = tuple(
        path for path in directory.glob(f"{sample_id}_*") if path.is_file()
    )
    if len(matches) != 1:
        raise ValueError(
            f"{directory} must contain exactly one {sample_id}_* file; "
            f"found {len(matches)}"
        )
    return matches[0]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    sample_ids = tuple(args.sample_ids)
    if (
        len(sample_ids) < 2
        or len(set(sample_ids)) != len(sample_ids)
        or any(not sample_id.strip() for sample_id in sample_ids)
    ):
        raise ValueError(
            "sample IDs must contain at least two unique non-empty values"
        )

    reference_paths = {
        sample_id: _resolve_prefix(args.reference_dir, sample_id)
        for sample_id in sample_ids
    }
    source_paths = {
        sample_id: _resolve_prefix(args.source_dir, sample_id)
        for sample_id in sample_ids
    }
    target_paths = {
        sample_id: _resolve_prefix(args.target_dir, sample_id)
        for sample_id in sample_ids
    }
    recipes = {
        sample_id: fit_reference_look(load_working_image(path))
        for sample_id, path in reference_paths.items()
    }

    rows = []
    for reference_id in sample_ids:
        recipe = recipes[reference_id]
        for source_id in sample_ids:
            if source_id == reference_id:
                continue
            source = load_working_image(source_paths[source_id])
            target = load_working_image(target_paths[source_id])
            candidate = render_reference_look(recipe, source).image
            row = evaluate_known_operator_batch(
                [source],
                [target],
                [candidate],
                sample_ids=[f"{reference_id}->{source_id}"],
            ).samples[0]
            rows.append(row)

    known_operator = aggregate_known_operator_samples(rows)
    photographic_safety = evaluate_recipe_batch_photographic_safety(
        recipes.values()
    )
    context_invariance = evaluate_recipe_batch_context_invariance(
        recipes.values()
    )
    decision = adjudicate_promotion(
        known_operator,
        photographic_safety,
        context_invariance,
    )
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "algorithm_ids": sorted(
            {recipe.algorithm_id for recipe in recipes.values()}
        ),
        "sample_ids": list(sample_ids),
        "reference_paths": {
            key: str(value.resolve()) for key, value in reference_paths.items()
        },
        "source_paths": {
            key: str(value.resolve()) for key, value in source_paths.items()
        },
        "target_paths": {
            key: str(value.resolve()) for key, value in target_paths.items()
        },
        "reference_file_sha256": {
            key: sha256_file(value) for key, value in reference_paths.items()
        },
        "source_file_sha256": {
            key: sha256_file(value) for key, value in source_paths.items()
        },
        "target_file_sha256": {
            key: sha256_file(value) for key, value in target_paths.items()
        },
        "recipe_ids": {
            key: recipe.recipe_id for key, recipe in recipes.items()
        },
        "known_operator": asdict(known_operator),
        "photographic_safety": asdict(photographic_safety),
        "context_invariance": asdict(context_invariance),
        "visual_review": None,
        "promotion_decision": asdict(decision),
    }
    path_only_keys = {
        "report_id",
        "reference_paths",
        "source_paths",
        "target_paths",
    }
    payload["report_id"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in path_only_keys
        }
    )
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "report_id": payload["report_id"],
                "sample_count": len(known_operator.samples),
                "promotion_status": decision.status,
                "reasons": list(decision.reasons),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
