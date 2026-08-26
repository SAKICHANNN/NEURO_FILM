from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.recipe_export_request import (
    RecipeExportRequestError,
    build_recipe_export_request_set,
    export_recipe_request,
    load_recipe_export_request,
    materialize_recipe_export_request_set,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
EVIDENCE = ROOT / "docs/evidence/U7_3F_OFFLINE_RECIPE_EXPORT_REQUEST_RESULT.json"


@pytest.fixture()
def history(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source.png"
    rendered = tmp_path / "rendered.png"
    history_root = tmp_path / "history"
    history_root.mkdir()
    pixels = np.arange(31 * 43 * 3, dtype=np.uint32).reshape(31, 43, 3)
    Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB").save(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--style",
            "portra_400",
            "--use-render-profile",
            "--output-bit-depth",
            "16",
            "--write-recipe",
            "--output",
            str(rendered),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe_path = history_root / "portra_400.recipe.json"
    recipe_path.write_bytes(rendered.with_suffix(".recipe.json").read_bytes())
    return {
        "root": history_root,
        "recipe": recipe_path,
        "rendered": rendered,
        "expected": rendered.read_bytes(),
    }


def test_request_set_is_exact_portable_and_contains_no_machine_paths(
    history: dict[str, object], tmp_path: Path
) -> None:
    first = build_recipe_export_request_set(history["root"])  # type: ignore[arg-type]
    second = build_recipe_export_request_set(history["root"])  # type: ignore[arg-type]
    assert first == second
    assert first["receipt"]["request_count"] == 1
    row = first["receipt"]["requests"][0]
    payload = first["files"][row["request_file"]]
    decoded = json.loads(payload)
    assert decoded["recipe_path"] == "portra_400.recipe.json"
    assert decoded["style"] == "portra_400"
    forbidden = (str(tmp_path), "C:\\", "P:\\", "D:\\")
    text = b"\n".join(first["files"].values()).decode("utf-8")
    assert not any(value in text for value in forbidden)
    assert b"Save export request" in first["files"]["index.html"]


def test_materialized_request_executes_existing_exact_replay(
    history: dict[str, object], tmp_path: Path
) -> None:
    request_set = tmp_path / "requests"
    receipt = materialize_recipe_export_request_set(
        history["root"], request_set  # type: ignore[arg-type]
    )
    request_path = request_set / receipt["requests"][0]["request_file"]
    destination = tmp_path / "export.png"
    export = export_recipe_request(
        history["root"],  # type: ignore[arg-type]
        request_path,
        profile_path=PROFILE,
        output_path=destination,
        root=ROOT,
    )
    assert destination.read_bytes() == history["expected"]
    assert export["style"] == "portra_400"
    assert export["request_sha256"] == receipt["requests"][0]["request_sha256"]


def test_request_drift_and_strict_json_fail_before_render(
    history: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_set = build_recipe_export_request_set(history["root"])  # type: ignore[arg-type]
    name = request_set["receipt"]["requests"][0]["request_file"]
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request_set["files"][name])

    calls = 0

    def forbidden_export(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("render must not start")

    monkeypatch.setattr(
        "src.inference.recipe_export_request.export_recipe_history_entry",
        forbidden_export,
    )
    value = json.loads(request_path.read_text(encoding="utf-8"))
    value["recipe_sha256"] = "0" * 64
    request_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RecipeExportRequestError, match="recipe_sha256 drift"):
        export_recipe_request(
            history["root"],  # type: ignore[arg-type]
            request_path,
            profile_path=PROFILE,
            output_path=tmp_path / "forbidden.png",
            root=ROOT,
        )
    assert calls == 0

    request_path.write_text(
        '{"schema":"kmcfm.recipe-export-request.v1","schema":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(RecipeExportRequestError, match="duplicate key"):
        load_recipe_export_request(request_path)


def test_cli_builds_and_executes_request(history: dict[str, object], tmp_path: Path) -> None:
    request_set = tmp_path / "cli-requests"
    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_recipe_request.py"),
            "--history-root",
            str(history["root"]),
            "--build-request-set",
            str(request_set),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    receipt = json.loads(build.stdout)
    request_path = request_set / receipt["requests"][0]["request_file"]
    destination = tmp_path / "cli-export.png"
    execute = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_recipe_request.py"),
            "--history-root",
            str(history["root"]),
            "--request",
            str(request_path),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert execute.returncode == 0, execute.stderr
    assert destination.read_bytes() == history["expected"]


def test_formal_evidence_binds_all_three_look_approximation_outputs() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_OFFLINE_EXPORT_REQUEST_BRIDGE"
    assert evidence["formal_reports"]["both_status"] == "PASS"
    assert evidence["portable_request_set"]["request_count"] == 3
    assert evidence["portable_request_set"]["machine_absolute_paths_present"] is False
    assert {row["style"] for row in evidence["rows"]} == {
        "ektar_100",
        "portra_400",
        "velvia_50",
    }
    assert all(row["forward_reverse_output_exact"] for row in evidence["rows"])
    assert all(evidence["gates"].values())
    assert evidence["scientific_boundary"]["stock_evidence_added"] is False
    assert evidence["scientific_boundary"]["stock_distinguishability_changed"] is False
