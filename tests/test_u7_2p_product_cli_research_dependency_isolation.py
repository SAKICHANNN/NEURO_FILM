from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
BLOCKED_ROOTS = (
    "src.eval",
    "pypdf",
    "cryptography",
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "peft",
    "wandb",
    "pytorch_lightning",
)


def _source(path: Path) -> None:
    y, x = np.mgrid[:29, :37]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _blocker_environment(root: Path) -> dict[str, str]:
    sitecustomize = root / "sitecustomize.py"
    roots = repr(BLOCKED_ROOTS)
    sitecustomize.write_text(
        "import importlib.abc\n"
        "import sys\n"
        f"BLOCKED = {roots}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if any(fullname == root or fullname.startswith(root + '.') "
        "for root in BLOCKED):\n"
        "            raise ImportError(f'U7.2P blocked import: {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not current else f"{root}{os.pathsep}{current}"
    return env


def _run(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _normalized_recipe(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<COMMIT>"
    payload["input"]["path"] = "<INPUT>"
    payload["output"]["path"] = "<OUTPUT>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_product_discovery_and_renders_do_not_import_research_graph(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    env = _blocker_environment(blocker)

    reference_discovery = _run("--list-product-looks")
    blocked_discovery = _run("--list-product-looks", env=env)
    assert reference_discovery.returncode == blocked_discovery.returncode == 0
    assert reference_discovery.stdout == blocked_discovery.stdout
    assert "U7.2P blocked import" not in blocked_discovery.stderr

    source = tmp_path / "source.png"
    _source(source)
    for look in ("velvia_50", "portra_400", "ektar_100"):
        reference = tmp_path / f"reference-{look}.png"
        candidate = tmp_path / f"candidate-{look}.png"
        common = (
            str(source),
            "--product-look",
            look,
            "--look-amount",
            "0.5",
            "--write-recipe",
        )
        first = _run(*common, "--output", str(reference))
        second = _run(*common, "--output", str(candidate), env=env)
        assert first.returncode == second.returncode == 0
        assert reference.read_bytes() == candidate.read_bytes()
        assert _normalized_recipe(reference.with_suffix(".recipe.json")) == (
            _normalized_recipe(candidate.with_suffix(".recipe.json"))
        )


def test_analytic_branch_attempts_research_import_only_when_selected(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    env = _blocker_environment(blocker)
    output = tmp_path / "must-not-exist.png"
    completed = _run(
        str(tmp_path / "must-not-decode.png"),
        "--color-engine",
        "analytic-y-chromaticity",
        "--output",
        str(output),
        env=env,
    )
    assert completed.returncode != 0
    assert "U7.2P blocked import: src.eval" in completed.stderr
    assert not output.exists()
