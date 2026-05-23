#!/usr/bin/env python3
"""Download community film LoRAs from Civitai for initial Phase 1 testing.

Civitai API: GET https://civitai.com/api/v1/models/{id}
File download: from modelVersions[].files[].downloadUrl
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"
LORAS_DIR.mkdir(exist_ok=True)

# SDXL-compatible film LoRAs (verified as of 2026-05)
# Model IDs from Civitai
SDXL_FILM_LORAS = {
    "portra_400": {
        "model_id": "723250",
        "name": "Kodak Portra 400",
        "trigger": "Portra 400",
        "file_pattern": "portra_400",
    },
    "vision3_500t": {
        "model_id": "725625",
        "name": "Kodak Vision3 500T",
        "trigger": "Kodak Vision3 500T",
        "file_pattern": "vision3_500t",
        "variant": "XL",  # Pick XL variant
    },
    "vision3_250d": {
        "model_id": "725625",
        "name": "Kodak Vision3 250D", 
        "trigger": "Kodak VISION3 250D",
        "file_pattern": "vision3_250d",
    },
    "portra_800": {
        "model_id": "725625",
        "name": "Kodak Portra 800",
        "trigger": "Kodak Portra 800",
        "file_pattern": "portra_800",
    },
    "ektar_100": {
        "model_id": "779013",
        "name": "Kodak Ektar 100",
        "trigger": "Kodak Ektar",
        "file_pattern": "ektar_100",
        "variant": "XL",
    },
    "tri_x_400": {
        "model_id": "521049",
        "name": "Kodak Tri-X 400",
        "trigger": "Tri-X 400",
        "file_pattern": "tri_x_400",
    },
}


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "K-MCFM/3.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  skipping (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "K-MCFM/3.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {downloaded//1024//1024}MB / {total//1024//1024}MB ({pct}%)", end="", flush=True)
        print()
    return True


def download_sdxl_lora(key: str, info: dict):
    model_id = info["model_id"]
    print(f"\n[{key}] {info['name']} (model {model_id})")

    # Get model metadata
    try:
        data = api_get(f"https://civitai.com/api/v1/models/{model_id}")
    except Exception as e:
        print(f"  API error: {e}")
        return False

    versions = data.get("modelVersions", [])
    if not versions:
        print(f"  No versions found")
        return False

    # Pick the right version
    selected = None
    for v in versions:
        base = v.get("baseModel", "")
        if "SDXL" in base or "XL" in base:
            if info.get("variant") == "XL" and "XL" in v.get("name", ""):
                selected = v
                break
            elif info.get("variant") != "F1D" and "F1D" not in v.get("name", ""):
                selected = v
                break
            selected = selected or v

    if selected is None:
        selected = versions[0]

    version_name = selected.get("name", "unknown")
    print(f"  Version: {version_name} (base: {selected.get('baseModel', '?')})")

    # Download the safetensors file
    files = selected.get("files", [])
    for f in files:
        fname = f.get("name", "")
        url = f.get("downloadUrl", "")
        if not url:
            continue
        if fname.endswith(".safetensors"):
            dest_name = f"{info['file_pattern']}.safetensors"
            dest = LORAS_DIR / dest_name
            print(f"  Downloading: {fname} → {dest_name}")
            download_file(url, dest)
            return True

    print(f"  No safetensors file found")
    return False


def main():
    print("=== Downloading SDXL Film LoRAs from Civitai ===\n")

    results = {}
    for key, info in SDXL_FILM_LORAS.items():
        ok = download_sdxl_lora(key, info)
        results[key] = ok
        time.sleep(1)

    print("\n=== Results ===")
    for key, ok in results.items():
        fname = LORAS_DIR / f"{SDXL_FILM_LORAS[key]['file_pattern']}.safetensors"
        exists = fname.exists()
        size = fname.stat().st_size // 1024 // 1024 if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {key}: {fname.name} ({size}MB)" if exists else f"  {status} {key}")

    print(f"\nLoRAs stored in: {LORAS_DIR}")
    print("\nNext: run scripts/pipeline.py to test inference")


if __name__ == "__main__":
    main()
