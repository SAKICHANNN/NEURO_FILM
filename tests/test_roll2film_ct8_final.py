import hashlib
import json
from pathlib import Path

import pytest

from src.roll2film.ct8_final import load_final_manifest, select_review_cases
from src.roll2film.manifests import FILMSET_MANIFEST_SCHEMA, FilmSetManifestError


def _write_manifest(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(content: str, domain: str) -> dict:
    return {
        "schema_version": FILMSET_MANIFEST_SCHEMA,
        "content_id": content,
        "duplicate_cluster_id": content,
        "domain": domain,
        "path": f"{content}/{domain}.jpg",
        "sha256": "0" * 64,
        "research_pool": "final_628_lockbox",
        "payload_access": "final_evaluator_only",
    }


def test_final_manifest_requires_complete_exact_roles(tmp_path: Path) -> None:
    path = tmp_path / "final.jsonl"
    rows = [_row("a", domain) for domain in ("input", "cinema", "classneg", "velvia")]
    digest = _write_manifest(path, rows)
    loaded = load_final_manifest(
        path,
        expected_sha256=digest,
        expected_identities=1,
        expected_rows=4,
        domains=("cinema", "classneg", "velvia"),
    )
    assert set(loaded["a"]) == {"input", "cinema", "classneg", "velvia"}
    rows[-1]["payload_access"] = "training_target_only"
    digest = _write_manifest(path, rows)
    with pytest.raises(FilmSetManifestError, match="unexpected access"):
        load_final_manifest(
            path,
            expected_sha256=digest,
            expected_identities=1,
            expected_rows=4,
            domains=("cinema", "classneg", "velvia"),
        )


def test_review_selection_includes_every_axis_maximum() -> None:
    axes = (
        "raw_excursion_max",
        "new_display_clip_pixel_fraction_vs_target",
        "mean_delta_e00_to_target",
        "red_cyan_boundary_occupancy",
        "speckle_candidate_percent",
    )
    rows = []
    for index, axis in enumerate(axes):
        flat = {name: 0.0 for name in axes}
        flat[axis] = 100.0
        rows.append(
            {
                "content_id": f"axis-{index}",
                "metrics": {
                    **{key: value for key, value in flat.items() if key != "speckle_candidate_percent"},
                    "chroma_speckle": {
                        "speckle_candidate_percent": flat["speckle_candidate_percent"]
                    },
                },
            }
        )
    selected = select_review_cases(
        rows, axes=axes, composite_count=1, per_axis_count=1
    )
    assert set(selected["union"]) == {f"axis-{index}" for index in range(len(axes))}
