# ARM Mac Migration Runbook

This repo is intended to move through GitHub without bundled datasets. The source Windows workspace has hundreds of GB under `data/`; those files are deliberately ignored and must be recreated on the Mac from public sources or separately licensed sources.

## 1. What Belongs In Git

Commit these:

- Source code: `src/`, `scripts/`, `tests/`
- Configs: `configs/`
- Project docs: root `*.md`, `docs/`, `guidelines.pdf`
- Dependency manifests: `requirements.txt`, `requirements-macos-arm.txt`
- Empty data directory markers: `data/**/.gitkeep`

Do not commit these:

- `.venv/`, `.pytest_cache/`, `.claude/`
- `checkpoints/`, `runs/`, `logs/`, `wandb/`, `output/`
- Any downloaded or generated data under `data/`
- Model weights: `*.pt`, `*.pth`, `*.safetensors`
- TIFF outputs: `*.tif`, `*.tiff`

Before the first GitHub push on the Windows machine:

```powershell
git status --short --ignored
git add .gitignore AGENTS.md DATA_LICENSE_BOUNDARIES.md FILMGRAINSTYLE740K_REQUEST_EMAIL.md GAP_ANALYSIS.md IMPL_PLAN.md README.md TASK_BOARD.md guidelines.pdf requirements.txt requirements-macos-arm.txt configs scripts src tests docs data/raw/.gitkeep data/processed/.gitkeep data/synthetic/.gitkeep data/calibration/.gitkeep data/physics/.gitkeep
git status --short
```

If any real file under `data/` appears in `git status --short`, stop and fix `.gitignore` before committing.

## 2. ARM Mac Environment

Use the Mac-specific manifest. The root `requirements.txt` is the CUDA 12.8 / RTX 5070 Ti environment and is not valid for Apple Silicon.

```zsh
xcode-select --install

git clone <github-url> neuro_film
cd neuro_film

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-macos-arm.txt
```

If the Mac does not already have Python 3.12:

```zsh
brew install pyenv
pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv
```

Verify the environment:

```zsh
python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps_available", torch.backends.mps.is_available())
print("mps_built", torch.backends.mps.is_built())

import diffusers, timm, kornia, rawpy, kan
print("core imports OK")
PY
```

Expected ARM Mac differences:

- Use `mps` when available, otherwise CPU.
- `causal-conv1d`, `mamba-ssm`, `bitsandbytes`, and `xformers` are intentionally omitted.
- CUDA-specific training assumptions in `AGENTS.md` still describe the Windows GPU baseline; Mac is for development, data preparation, tests, and small smoke runs unless later code adds MPS-specific training configs.

## 3. Recreate Data Directories

```zsh
mkdir -p data/raw data/processed data/synthetic data/physics data/calibration/cie data/calibration/camera_spectral
```

Approximate disk target for the same local resources as the Windows machine: 350 GB before future processed manifests/checkpoints.

## 4. Public Downloads

FiveK RAW archive:

```zsh
python scripts/download_data.py fivek-dng
```

FiveK Expert C TIFF16 files:

```zsh
python scripts/download_data.py fivek-expert --expert c --workers 8
```

FilmSet archive from Kaggle:

```zsh
python scripts/download_data.py filmset-zip
python -m zipfile -e data/raw/filmset/filmset.zip data/raw/filmset
```

If Kaggle blocks the direct URL, install Kaggle credentials in `~/.kaggle/kaggle.json` and download dataset `xuhangc/filmset` with the Kaggle CLI, then place the archive at `data/raw/filmset/filmset.zip` and extract to `data/raw/filmset/FilmSet`.

DPED auxiliary archives:

```zsh
python scripts/download_data.py dped --dped-part all
```

CIE colorimetry CSVs and metadata:

```zsh
python scripts/download_data.py cie
```

Seed film physics PDFs currently supported by the script:

```zsh
python scripts/download_data.py physics
```

The Windows workspace had additional manufacturer PDFs under `data/physics/` and RIT/Tokyo camera spectral files under `data/calibration/camera_spectral/`. Recreate those from the source links in `AGENTS.md` section 6 before implementing physics calibration extraction. Keep the same destination layout:

```text
data/physics/<stock>/technical_data.pdf
data/calibration/camera_spectral/rit_camspec/
data/calibration/camera_spectral/tokyo_open_vision/
```

## 5. Restricted Or Unavailable Data

Do not try to put these in Git:

- FilmGrainStyle740k: requires an email request. Use `FILMGRAINSTYLE740K_REQUEST_EMAIL.md`.
- Cinestill800T/sillystill: dataset was not published at the time this project state was recorded.
- L5 real scan validation set: user-owned future capture/scan data; keep local only.

## 6. Post-Download Checks

```zsh
python scripts/download_data.py --help

python - <<'PY'
from pathlib import Path
checks = [
    "data/raw/fivek/fivek_dataset.tar",
    "data/raw/fivek/expert_tiff/c",
    "data/raw/filmset/filmset.zip",
    "data/raw/dped/dped_patches.gz",
    "data/calibration/cie/CIE_xyz_1931_2deg.csv",
]
for item in checks:
    path = Path(item)
    print(item, "OK" if path.exists() else "MISSING")
PY
```

Run tests after environment setup:

```zsh
pytest
```

The project is still before task `1.2` data pipeline implementation, so dataset manifest generation is not available yet. The next agent should read `AGENTS.md`, then `TASK_BOARD.md`, then start `1.2` by building `data/processed/manifest.jsonl` from the recreated data tree.
