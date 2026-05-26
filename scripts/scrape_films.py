#!/usr/bin/env python3
"""Download real film scans from Flickr for LoRA training.

Requires Flickr API key (free at flickr.com/services/apps/create).
Store in .env: FLICKR_API_KEY=xxx, FLICKR_API_SECRET=xxx
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILM_DOMAIN = ROOT / "data" / "film_domain"

FILM_STOCKS = {
    "portra_400": {
        "name": "Kodak Portra 400",
        "tags": ["kodak portra 400", "portra400", "portra 400 film"],
        "dir": "portra_400",
    },
    "vision3_500t": {
        "name": "Kodak Vision3 500T",
        "tags": ["kodak vision3 500t", "vision3 500t", "5219 film"],
        "dir": "vision3_500t",
    },
    "vision3_250d": {
        "name": "Kodak Vision3 250D",
        "tags": ["kodak vision3 250d", "vision3 250d", "5207 film"],
        "dir": "vision3_250d",
    },
    "portra_800": {
        "name": "Kodak Portra 800",
        "tags": ["kodak portra 800", "portra800", "portra 800 film"],
        "dir": "portra_800",
    },
    "ektar_100": {
        "name": "Kodak Ektar 100",
        "tags": ["kodak ektar 100", "ektar100", "ektar 100 film"],
        "dir": "ektar_100",
    },
    "velvia_50": {
        "name": "Fujifilm Velvia 50",
        "tags": ["fujifilm velvia 50", "velvia 50", "velvia50", "fuji velvia film"],
        "dir": "velvia_50",
    },
    "hp5": {
        "name": "Ilford HP5 Plus",
        "tags": ["ilford hp5", "hp5+", "ilford hp5 plus", "hp5 plus film"],
        "dir": "hp5",
    },
    "tri_x_400": {
        "name": "Kodak Tri-X 400",
        "tags": ["kodak tri-x 400", "tri-x 400", "tri-x400", "tri x 400 film"],
        "dir": "tri_x_400",
    },
}


def load_api_creds():
    key = os.environ.get("FLICKR_API_KEY", "").strip()
    secret = os.environ.get("FLICKR_API_SECRET", "").strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FLICKR_API_KEY="):
                key = key or line.split("=", 1)[1].strip()
            elif line.startswith("FLICKR_API_SECRET="):
                secret = secret or line.split("=", 1)[1].strip()
    if not key or not secret:
        print("ERROR: FLICKR_API_KEY and FLICKR_API_SECRET required in .env")
        exit(1)
    return key, secret


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png"))


def scrape_stock(stock_key: str, target_total: int, flickr, dest_dir: Path):
    stock = FILM_STOCKS[stock_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    seen_urls = set()

    for tag in stock["tags"]:
        if count_images(dest_dir) >= target_total:
            break
        page = 1
        while count_images(dest_dir) < target_total and page <= 100:
            try:
                r = flickr.photos.search(
                    tags=tag,
                    tag_mode="all",
                    sort="relevance",
                    content_type=1,
                    media="photos",
                    per_page=50,
                    page=page,
                    extras="url_o,url_l,url_c",
                )
                photos = r.get("photos", {}).get("photo", [])
                if not photos:
                    break

                for photo in photos:
                    if count_images(dest_dir) >= target_total:
                        break
                    url = photo.get("url_o") or photo.get("url_l") or photo.get("url_c")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    h = hashlib.sha256(url.encode()).hexdigest()[:16]
                    ext = os.path.splitext(urllib.request.urlparse(url).path)[1] or ".jpg"
                    if ext.lower() not in (".jpg", ".jpeg", ".png"):
                        ext = ".jpg"
                    dest = dest_dir / f"fl_{h}{ext}"
                    if dest.exists():
                        continue

                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "K-MCFM/3.0 research"})
                        with urllib.request.urlopen(req, timeout=20) as resp:
                            dest.write_bytes(resp.read())
                        downloaded += 1
                    except Exception:
                        pass

                page += 1
                time.sleep(1.0)

            except Exception as e:
                err = str(e)[:80]
                if "100" in err or "limit" in err.lower():
                    print(f"  Rate limited, waiting 30s...")
                    time.sleep(30)
                else:
                    print(f"  API error: {err}")
                    time.sleep(5)

    return downloaded


def dedup(directory: Path, min_kb: int = 30):
    seen = {}
    removed = 0
    kept = 0
    for img in sorted(directory.rglob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        sz = img.stat().st_size
        if sz < min_kb * 1024:
            img.unlink(); removed += 1; continue
        h = hashlib.md5(img.read_bytes()).hexdigest()
        if h in seen:
            img.unlink(); removed += 1
        else:
            seen[h] = img; kept += 1
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Scrape real film scans from Flickr")
    parser.add_argument("--stock", choices=list(FILM_STOCKS), help="Specific stock to scrape")
    parser.add_argument("--count", type=int, default=500, help="Target images per stock")
    parser.add_argument("--all", action="store_true", help="Scrape all 8 stocks")
    parser.add_argument("--dedup", action="store_true", help="Dedup after download")
    args = parser.parse_args()

    api_key, api_secret = load_api_creds()
    import flickrapi
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format="parsed-json")

    stocks_to_scrape = list(FILM_STOCKS) if args.all else [args.stock]

    for sk in stocks_to_scrape:
        dest = FILM_DOMAIN / FILM_STOCKS[sk]["dir"]
        existing = count_images(dest)
        needed = max(0, args.count - existing)
        print(f"\n[{sk}] exists={existing}, need={needed}")

        if needed > 0:
            n = scrape_stock(sk, args.count, flickr, dest)
            print(f"  downloaded {n} new images")

        if args.dedup:
            k, r = dedup(dest)
            print(f"  dedup: {k} kept, {r} removed")

    print("\nDone. Status:")
    for sk in sorted(FILM_STOCKS):
        dest = FILM_DOMAIN / FILM_STOCKS[sk]["dir"]
        cnt = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"  {sk}: {cnt} images")


if __name__ == "__main__":
    main()
