#!/usr/bin/env python3
"""Download community film LoRAs from Civitai for initial Phase 1 testing.

Civitai API: GET https://civitai.com/api/v1/models/{id}
File download: from modelVersions[].files[].downloadUrl
"""

import json
import os
import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"
LORAS_DIR.mkdir(exist_ok=True)

# Exact-stock SDXL-compatible film LoRAs verified against the Civitai API
# on 2026-05-25. Keep this list strict: do not point a stock at a generic
# film LoRA or an unrelated search hit.
SDXL_FILM_LORAS = {
    "portra_400": {
        "model_id": "723250",
        "name": "Kodak Portra 400",
        "trigger": "Portra 400",
        "file_pattern": "portra_400",
        "version_id": "808680",
        "size_kb": 223106.71484375,
    },
    "vision3_500t": {
        "model_id": "725625",
        "name": "Kodak Vision3 500T",
        "trigger": "Kodak Vision3 500T",
        "file_pattern": "vision3_500t",
        "variant": "XL",  # SDXL version id 820808
        "version_id": "820808",
        "size_kb": 445786.15234375,
    },
    "vision3_250d": {
        "model_id": "725620",
        "name": "Kodak Vision3 250D", 
        "trigger": "Kodak VISION3 250D",
        "file_pattern": "vision3_250d",
        "variant": "XL",  # SDXL version id 820761
        "version_id": "820761",
        "size_kb": 445786.15234375,
    },
    "ektar_100": {
        "model_id": "779013",
        "name": "Kodak Ektar 100",
        "trigger": "Kodak Ektar",
        "file_pattern": "ektar_100",
        "variant": "XL",  # SDXL version id 1167852
        "version_id": "1167852",
        "size_kb": 445789.86328125,
    },
}

# No exact SDXL Civitai LoRA was verified for these stocks on 2026-05-25.
# Use self-trained LoRAs or a non-SDXL fallback explicitly instead.
NEEDS_SELF_TRAINED_SDXL = [
    "portra_800",
    "tri_x_400",
    "velvia_50",
    "hp5",
]


def load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def auth_headers() -> dict[str, str]:
    headers = {"User-Agent": "K-MCFM/3.0"}
    token = os.environ.get("CIVITAI_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def with_token(url: str) -> str:
    token = os.environ.get("CIVITAI_API_TOKEN", "").strip()
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    if "/api/download/models/" in parsed.path:
        query.setdefault("type", "Model")
        query.setdefault("format", "SafeTensor")
    if not token:
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))
    query.setdefault("token", token)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def download_file(url: str, dest: Path, expected_size_kb: float | None = None) -> bool:
    expected_bytes = int(expected_size_kb * 1024) if expected_size_kb else None
    if dest.exists() and (not expected_bytes or dest.stat().st_size >= expected_bytes * 0.98):
        print(f"  skipping (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  existing file looks incomplete, replacing: {dest.name}")
        dest.replace(tmp)
    req = urllib.request.Request(with_token(url), headers=auth_headers())
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (32 * 1024 * 1024) < len(chunk):
                    pct = downloaded * 100 // total
                    print(f"  {dest.name}: {downloaded//1024//1024}MB / {total//1024//1024}MB ({pct}%)", flush=True)
        tmp.replace(dest)
        print(f"  saved: {dest.name} ({dest.stat().st_size//1024//1024}MB)")
    return True


def download_sdxl_lora(key: str, info: dict, dry_run: bool = False):
    model_id = info["model_id"]
    print(f"\n[{key}] {info['name']} (model {model_id})")

    # Get model metadata
    try:
        data = api_get(f"https://civitai.com/api/v1/models/{model_id}")
    except Exception as e:
        print(f"  API error: {e}")
        version_id = info.get("version_id")
        if not version_id:
            return False
        dest_name = f"{info['file_pattern']}.safetensors"
        dest = LORAS_DIR / dest_name
        url = f"https://civitai.com/api/download/models/{version_id}"
        if dry_run:
            print(f"  Would download via verified version id: {version_id} -> {dest_name}")
            return True
        print(f"  Falling back to verified version id: {version_id}")
        return download_file(url, dest, info.get("size_kb"))

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
            if dry_run:
                size_mb = int(f.get("sizeKB", 0) // 1024)
                print(f"  Would download: {fname} -> {dest_name} ({size_mb}MB)")
                return True
            print(f"  Downloading: {fname} -> {dest_name}")
            download_file(url, dest, f.get("sizeKB"))
            return True

    print(f"  No safetensors file found")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download verified SDXL film LoRAs from Civitai.")
    parser.add_argument("--list", action="store_true", help="List verified and self-trained stocks without network access.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata but do not download weights.")
    parser.add_argument("--style", choices=sorted(SDXL_FILM_LORAS), help="Download only one verified style.")
    args = parser.parse_args()

    load_local_env()
    if args.list:
        print("Verified SDXL Civitai LoRAs:")
        for key, info in SDXL_FILM_LORAS.items():
            print(f"  {key}: model {info['model_id']}")
        print("\nNeeds self-trained SDXL:")
        for key in NEEDS_SELF_TRAINED_SDXL:
            print(f"  {key}")
        return

    print("=== Downloading SDXL Film LoRAs from Civitai ===\n")
    print(f"Civitai token: {'set' if os.environ.get('CIVITAI_API_TOKEN') else 'not set'}\n")

    results = {}
    selected_loras = {args.style: SDXL_FILM_LORAS[args.style]} if args.style else SDXL_FILM_LORAS
    for key, info in selected_loras.items():
        ok = download_sdxl_lora(key, info, dry_run=args.dry_run)
        results[key] = ok
        time.sleep(1)

    print("\n=== Results ===")
    for key, ok in results.items():
        fname = LORAS_DIR / f"{SDXL_FILM_LORAS[key]['file_pattern']}.safetensors"
        exists = fname.exists() and not args.dry_run
        size = fname.stat().st_size // 1024 // 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {status} {key}: {fname.name} ({size}MB)" if exists else f"  {status} {key}")

    if NEEDS_SELF_TRAINED_SDXL:
        print("\nNo exact SDXL Civitai LoRA verified for:")
        for key in NEEDS_SELF_TRAINED_SDXL:
            print(f"  - {key}")
        print("Use self-trained weights or add a newly verified model id before downloading.")

    print(f"\nLoRAs stored in: {LORAS_DIR}")
    if not args.dry_run:
        print("\nNext: run scripts/pipeline.py to test inference")


if __name__ == "__main__":
    main()
