from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.render_contract import sha256_file
from src.inference.resumable_three_stock_input_batch import (
    render_resumable_three_stock_input_batch_to_directory,
)
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    ThreeStockInputBatchError,
    _job_id_is_safe_path_component,
    _load_jobs,
    render_three_stock_input_batch_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"

VALID_IDS = ("a", "a-b_c.1", "scene.001", "canon-eos-5d-mark-iv-sraw")
INVALID_IDS = (
    "a.",
    "con",
    "con.txt",
    "prn",
    "aux.log",
    "nul",
    "com1",
    "com9.raw",
    "lpt1",
    "lpt9.txt",
)


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:4, :6]
    rgb = np.stack((xx * 11 + yy, xx + yy * 13, xx * 7 + yy * 5), axis=-1)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _manifest(path: Path, job_ids: tuple[str, ...], source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "jobs": [
                    {
                        "job_id": job_id,
                        "input_path": str(source),
                        "input_sha256": sha256_file(source),
                    }
                    for job_id in job_ids
                ],
            }
        ),
        encoding="utf-8",
    )


def _render_input(manifest: Path, destination: Path) -> dict:
    return render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def _render_resumable(manifest: Path, workspace: Path, destination: Path) -> dict:
    return render_resumable_three_stock_input_batch_to_directory(
        manifest,
        workspace,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


@pytest.mark.parametrize("job_id", VALID_IDS)
def test_valid_job_ids_preserve_exact_component(job_id: str, tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, (job_id,), source)
    jobs = _load_jobs(manifest)
    assert jobs[0]["job_id"] == job_id
    assert _job_id_is_safe_path_component(job_id) is True


@pytest.mark.parametrize("job_id", INVALID_IDS)
@pytest.mark.parametrize("entrypoint", ("input", "resumable"))
def test_unsafe_job_ids_reject_before_input_or_output_activity(
    job_id: str,
    entrypoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, (job_id,), source)
    workspace = tmp_path / "workspace"
    destination = tmp_path / "published"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid job id reached input preflight or render")

    monkeypatch.setattr(
        "src.inference.three_stock_input_batch._preflight_jobs", forbidden
    )
    monkeypatch.setattr(
        "src.inference.three_stock_input_batch.render_three_stock_batch_to_directory",
        forbidden,
    )
    monkeypatch.setattr(
        "src.inference.resumable_three_stock_input_batch._preflight_jobs", forbidden
    )
    monkeypatch.setattr(
        "src.inference.resumable_three_stock_input_batch.render_three_stock_batch_to_directory",
        forbidden,
    )

    operation = lambda: _render_input(manifest, destination)
    if entrypoint == "resumable":
        operation = lambda: _render_resumable(manifest, workspace, destination)
    with pytest.raises(ThreeStockInputBatchError, match="id is invalid"):
        operation()
    assert not workspace.exists()
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.stage"))
    assert not list(tmp_path.glob(".*.lease"))


def test_forward_reverse_valid_ids_parse_to_one_canonical_order(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, ("scene.001", "a-b_c.1"), source)
    forward = _load_jobs(manifest)
    _manifest(manifest, ("a-b_c.1", "scene.001"), source)
    reverse = _load_jobs(manifest)
    assert forward == reverse
    assert [row["job_id"] for row in forward] == ["a-b_c.1", "scene.001"]


@pytest.mark.skipif(os.name != "nt", reason="Windows alias witness")
def test_windows_trailing_dot_alias_witness(tmp_path: Path) -> None:
    canonical = tmp_path / "a"
    alias = tmp_path / "a."
    canonical.mkdir()
    with pytest.raises(FileExistsError):
        alias.mkdir()
    assert canonical.is_dir()
    assert alias.is_dir()
    assert [item.name for item in tmp_path.iterdir()] == ["a"]
