from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from src.inference.recipe_desktop_workflow import (
    IntegratedRecipeDesktopSession,
    RecipeDesktopWorkflowError,
)
from src.inference.recipe_export_request import build_recipe_export_request_set

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STYLES = ("velvia_50", "portra_400", "ektar_100")


@pytest.fixture(scope="module")
def three_recipe_history(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("u7-3j-history")
    source = root / "source.png"
    history = root / "history"
    history.mkdir()
    yy, xx = np.mgrid[:61, :79]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11) % 256,
            (xx * 13 + yy * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(source)
    expected: dict[str, bytes] = {}
    for style in STYLES:
        output = root / f"{style}.png"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_film.py"),
                str(source),
                "--style",
                style,
                "--use-render-profile",
                "--output-bit-depth",
                "16",
                "--write-recipe",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        (history / f"{style}.recipe.json").write_bytes(
            output.with_suffix(".recipe.json").read_bytes()
        )
        expected[style] = output.read_bytes()
    return {"root": history, "expected": expected}


def _post(url: str, fields: dict[str, str]) -> bytes:
    request = Request(
        url,
        data=urlencode(fields).encode("ascii"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def test_integrated_page_contains_three_verified_looks_without_paths(
    three_recipe_history: dict[str, object], tmp_path: Path
) -> None:
    history = three_recipe_history["root"]
    with IntegratedRecipeDesktopSession(
        history,  # type: ignore[arg-type]
        tmp_path / "exports",
        profile_path=PROFILE,
        root=ROOT,
    ) as session:
        with urlopen(session.url, timeout=10) as response:
            page = response.read().decode("utf-8")
        assert page.count('class="look-card"') == 3
        assert page.count("data:image/png;base64,") == 3
        assert page.count("Export this look") == 3
        assert page.count("Look Approximation") >= 4
        assert "Fujifilm Velvia 50" in page
        assert "Kodak Portra 400" in page
        assert "Kodak Ektar 100" in page
        assert 'aria-live="polite"' in page
        assert str(tmp_path) not in page
        assert str(history) not in page


@pytest.mark.parametrize("style", STYLES)
def test_each_stock_selection_replays_exact_recipe(
    style: str, three_recipe_history: dict[str, object], tmp_path: Path
) -> None:
    history = three_recipe_history["root"]
    request_set = build_recipe_export_request_set(history)  # type: ignore[arg-type]
    row = next(item for item in request_set["receipt"]["requests"] if item["style"] == style)
    with IntegratedRecipeDesktopSession(
        history,  # type: ignore[arg-type]
        tmp_path / "exports",
        profile_path=PROFILE,
        root=ROOT,
    ) as session:
        token = urlsplit(session.url).query.removeprefix("token=")
        result_page = _post(
            f"http://127.0.0.1:{session.port}/export",
            {"token": token, "request_file": row["request_file"]},
        )
        assert b"Export complete" in result_page
        result = session.wait(1)
        assert result is not None
        assert result.style == style
        expected = three_recipe_history["expected"]
        assert result.output_path.read_bytes() == expected[style]  # type: ignore[index]
        assert [path.name for path in result.output_path.parent.iterdir()] == [
            result.output_path.name
        ]


def test_integrated_session_rejects_incomplete_catalog(
    three_recipe_history: dict[str, object], tmp_path: Path
) -> None:
    history = three_recipe_history["root"]
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "velvia_50.recipe.json").write_bytes(
        (history / "velvia_50.recipe.json").read_bytes()  # type: ignore[operator]
    )
    with pytest.raises(RecipeDesktopWorkflowError, match="exact Velvia"):
        IntegratedRecipeDesktopSession(
            incomplete,
            tmp_path / "exports",
            profile_path=PROFILE,
            root=ROOT,
        )
