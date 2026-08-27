from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from src.inference.recipe_browser_export import (
    LOOPBACK_HOST,
    RecipeBrowserExportSession,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


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
            "ektar_100",
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
    recipe = history_root / "ektar_100.recipe.json"
    recipe.write_bytes(rendered.with_suffix(".recipe.json").read_bytes())
    return {"root": history_root, "expected": rendered.read_bytes()}


def _post(url: str, fields: dict[str, str]) -> bytes:
    request = Request(
        url,
        data=urlencode(fields).encode("ascii"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def test_browser_page_posts_one_exact_export(
    history: dict[str, object], tmp_path: Path
) -> None:
    output_root = tmp_path / "exports"
    with RecipeBrowserExportSession(
        history["root"],  # type: ignore[arg-type]
        output_root,
        profile_path=PROFILE,
        root=ROOT,
    ) as session:
        parsed = urlsplit(session.url)
        assert parsed.hostname == LOOPBACK_HOST
        with urlopen(session.url, timeout=10) as response:
            page = response.read().decode("utf-8")
        assert "Export this look" in page
        assert "name=\"request_file\"" in page
        assert str(tmp_path) not in page
        token = parsed.query.removeprefix("token=")
        request_file = session.request_files[0]
        result_page = _post(
            f"http://{LOOPBACK_HOST}:{session.port}/export",
            {"token": token, "request_file": request_file},
        )
        assert b"Export complete" in result_page
        result = session.wait(1)
        assert result is not None
        assert result.style == "ektar_100"
        assert result.output_path.read_bytes() == history["expected"]
        with pytest.raises(HTTPError) as second:
            _post(
                f"http://{LOOPBACK_HOST}:{session.port}/export",
                {"token": token, "request_file": request_file},
            )
        assert second.value.code == 400
        assert result.output_path.read_bytes() == history["expected"]


def test_invalid_token_and_request_publish_nothing(
    history: dict[str, object], tmp_path: Path
) -> None:
    output_root = tmp_path / "exports"
    with RecipeBrowserExportSession(
        history["root"],  # type: ignore[arg-type]
        output_root,
        profile_path=PROFILE,
        root=ROOT,
    ) as session:
        endpoint = f"http://{LOOPBACK_HOST}:{session.port}/export"
        with pytest.raises(HTTPError) as invalid_token:
            _post(endpoint, {"token": "invalid", "request_file": "missing.json"})
        assert invalid_token.value.code == 400
        assert list(output_root.iterdir()) == []


def test_server_rejects_non_loopback_get(history: dict[str, object], tmp_path: Path) -> None:
    with RecipeBrowserExportSession(
        history["root"],  # type: ignore[arg-type]
        tmp_path / "exports",
        profile_path=PROFILE,
        root=ROOT,
    ) as session:
        with pytest.raises(HTTPError) as missing_token:
            urlopen(
                f"http://{LOOPBACK_HOST}:{session.port}/", timeout=10
            )
        assert missing_token.value.code == 404
