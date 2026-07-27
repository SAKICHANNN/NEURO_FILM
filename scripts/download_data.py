from __future__ import annotations

import argparse
import concurrent.futures
import html.parser
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIVEK_BASE = "https://data.csail.mit.edu/graphics/fivek/"
FIVEK_ARCHIVE = urllib.parse.urljoin(FIVEK_BASE, "fivek_dataset.tar")
FILMSET_LIST = "https://www.kaggle.com/api/v1/datasets/list/xuhangc/filmset"
FILMSET_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/xuhangc/filmset"
DPED_URLS = {
    "sample": "https://download.ai-benchmark.com/s/9RqJ5JZPDggMoyo/download/sample_images.gz",
    "patches": "https://download.ai-benchmark.com/s/dWZz5RD9kHpedix/download/dped.gz",
    "original": "https://download.ai-benchmark.com/s/rC6PwBK8exRomy8/download/original_images.gz",
}
CIE_FILES = [
    "CIE_xyz_1931_2deg.csv",
    "CIE_xyz_1931_2deg.csv_metadata.json",
    "CIE_std_illum_D65.csv",
    "CIE_std_illum_D65.csv_metadata.json",
    "CIE_std_illum_D50.csv",
    "CIE_std_illum_D50.csv_metadata.json",
    "CIE_illum_D55.csv",
    "CIE_illum_D55.csv_metadata.json",
    "CIE_std_illum_A_1nm.csv",
    "CIE_std_illum_A_1nm.csv_metadata.json",
]
PHYSICS_JOBS = [
    (
        "https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-50D-5203-7203-technical-information.pdf",
        "data/physics/kodak_vision3_50d/technical_data.pdf",
    ),
    (
        "https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-250D-5207-7207-technical-information.pdf",
        "data/physics/kodak_vision3_250d/technical_data.pdf",
    ),
    (
        "https://www.kodak.com/content/products-brochures/motion-picture/KODAK-VISION-Color-Print-Film-2383-3383-technical-information.pdf",
        "data/physics/kodak_vision_print_2383/technical_data.pdf",
    ),
    (
        "https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-VISION-Color-Print-Film-H-61b.pdf",
        "data/physics/kodak_vision_print_2383/lad_print_h61b.pdf",
    ),
    (
        "https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf",
        "data/physics/kodak_vision_print_2383/lad_negative_h61a.pdf",
    ),
    (
        "https://www.kodak.com/content/products-brochures/Film/Additive-and-Subtractive-Printing-on-Motion-Picture-Film-tib5207.pdf",
        "data/physics/kodak_vision_print_2383/additive_subtractive_printing_tib5207.pdf",
    ),
    (
        "https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-500T-5219-7219-technical-information.pdf",
        "data/physics/kodak_vision3_500t/technical_data.pdf",
    ),
    (
        "https://imaging.kodakalaris.com/sites/default/files/files/resources/e4051_Portra_160.pdf",
        "data/physics/kodak_portra_160/technical_data.pdf",
    ),
    (
        "https://imaging.kodakalaris.com/sites/default/files/files/resources/e4050_portra_400.pdf",
        "data/physics/kodak_portra_400/technical_data.pdf",
    ),
    (
        "https://imaging.kodakalaris.com/sites/default/files/files/products/e4040_portra_800.pdf",
        "data/physics/kodak_portra_800/technical_data.pdf",
    ),
    (
        "https://www.kodakprofessional.com/sites/default/files/wysiwyg/pro/resources/e4046_ektar_100.pdf",
        "data/physics/kodak_ektar_100/technical_data.pdf",
    ),
    (
        "https://kodakprofessional.com/sites/default/files/wysiwyg/film/f4017_trix_320400.pdf",
        "data/physics/kodak_trix/technical_data.pdf",
    ),
    (
        "https://www.ilfordphoto.com/amfile/file/download/file/1903/product/692/",
        "data/physics/ilford_hp5/technical_data.pdf",
    ),
    (
        "https://asset.fujifilm.com/www/us/files/2020-03/85d928f44b0df3b2a95913e46608881d/ProfessionalFilmDataGuide.pdf",
        "data/physics/fujifilm_professional_guide/professional_film_data_guide.pdf",
    ),
    (
        "https://asset.fujifilm.com/master/emea/files/2020-10/a71dda63e2662f012b3b74110794918a/films_velvia-50_datasheet_01.pdf",
        "data/physics/fujifilm_velvia_50/product_information_bulletin.pdf",
    ),
    (
        "https://asset.fujifilm.com/www/us/files/2020-03/64873257f4644939d7bd75d95600a561/AF3-960E.pdf",
        "data/physics/fujifilm_velvia_rvp/technical_data.pdf",
    ),
    (
        "https://asset.fujifilm.com/www/us/files/2020-03/226db49cde8b549d243c6498a80479cc/Velvia100_Brochure_Final.pdf",
        "data/physics/fujifilm_velvia_100/brochure.pdf",
    ),
]
CAMERA_SPECTRAL_JOBS = [
    (
        "https://www.gujinwei.org/research/camspec/camspec_database.txt",
        "data/calibration/camera_spectral/rit_camspec/camspec_database.txt",
    ),
    (
        "https://www.gujinwei.org/research/camspec/camlist%26equipment.txt",
        "data/calibration/camera_spectral/rit_camspec/camlist_equipment.txt",
    ),
    (
        "https://www.gujinwei.org/research/camspec/css_code.zip",
        "data/calibration/camera_spectral/rit_camspec/css_code.zip",
    ),
    (
        "https://www.gujinwei.org/research/camspec/camspec.pdf",
        "data/calibration/camera_spectral/rit_camspec/paper.pdf",
    ),
    (
        "https://www.gujinwei.org/research/camspec/supp.pdf",
        "data/calibration/camera_spectral/rit_camspec/supplement.pdf",
    ),
]
TOKYO_SPECTRAL_BASE = "https://open-vision.sc.e.titech.ac.jp/~reikawa/research/cs/zhao/"
DORF_ARCHIVE = "https://www.cs.columbia.edu/CAVE/software/dorf/response/dorfCurves.zip"


class FiveKParser(html.parser.HTMLParser):
    def __init__(self, expert: str):
        super().__init__()
        self.expert = expert
        self.paths: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        marker = f"img/tiff16_{self.expert}/"
        if href.startswith(marker) and href.endswith(".tif"):
            self.paths.append(href)


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def download_one(url: str, out: Path) -> tuple[str, bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return str(out), False, "exists"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        data = fetch_bytes(url)
        tmp.write_bytes(data)
        tmp.replace(out)
        return str(out), True, "ok"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return str(out), False, f"{type(exc).__name__}: {exc}"


def download_stream(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.stat().st_size if out.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(req, timeout=120) as response:
        if existing and getattr(response, "status", None) != 206:
            existing = 0
            mode = "wb"
        handle = out.open(mode)
        total_header = response.headers.get("Content-Length")
        remaining = int(total_header) if total_header else None
        total = existing + remaining if remaining else None
        downloaded = existing
        started = time.time()
        last_report = started
        try:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 30:
                    last_report = now
                    if total:
                        print(f"{downloaded / 1e9:.2f}/{total / 1e9:.2f} GB {out}", flush=True)
                    else:
                        print(f"{downloaded / 1e9:.2f} GB {out}", flush=True)
        finally:
            handle.close()
        if total:
            print(f"{downloaded / 1e9:.2f}/{total / 1e9:.2f} GB {out}", flush=True)
        else:
            print(f"{downloaded / 1e9:.2f} GB {out}", flush=True)


def run_download(jobs: list[tuple[str, Path]], workers: int) -> None:
    total = len(jobs)
    done = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, url, out) for url, out in jobs]
        for future in concurrent.futures.as_completed(futures):
            path, changed, status = future.result()
            done += 1
            if done % 25 == 0 or status not in {"ok", "exists"}:
                elapsed = max(time.time() - started, 1.0)
                rate = done / elapsed
                print(f"{done}/{total} {rate:.2f}/s {status} {path}", flush=True)


def download_physics() -> None:
    jobs = [(url, ROOT / rel) for url, rel in PHYSICS_JOBS]
    run_download(jobs, workers=4)


def list_filmset_files(limit: int | None = None) -> list[str]:
    files: list[str] = []
    token = None
    while True:
        url = FILMSET_LIST
        if token:
            url += "?" + urllib.parse.urlencode({"pageToken": token})
        payload = json.loads(fetch_bytes(url))
        for item in payload.get("datasetFiles", []):
            name = item.get("name")
            if name:
                files.append(name)
                if limit and len(files) >= limit:
                    return files
        token = payload.get("nextPageToken")
        if not token:
            return files


def download_filmset(workers: int, limit: int | None) -> None:
    files = list_filmset_files(limit)
    manifest = ROOT / "data/raw/filmset/files.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(files, indent=2), encoding="utf-8")
    jobs = []
    for name in files:
        url = FILMSET_DOWNLOAD + "?" + urllib.parse.urlencode({"file_name": name})
        rel = Path(*Path(name).parts[1:])
        jobs.append((url, ROOT / "data/raw/filmset" / rel))
    run_download(jobs, workers)


def download_filmset_zip() -> None:
    out = ROOT / "data/raw/filmset/filmset.zip"
    download_stream(FILMSET_DOWNLOAD, out)


def download_fivek_dng() -> None:
    out = ROOT / "data/raw/fivek/fivek_dataset.tar"
    download_stream(FIVEK_ARCHIVE, out)


def download_dped(part: str) -> None:
    selected = DPED_URLS.keys() if part == "all" else [part]
    names = {
        "sample": "sample_images.gz",
        "patches": "dped_patches.gz",
        "original": "original_images.gz",
    }
    for key in selected:
        download_stream(DPED_URLS[key], ROOT / "data/raw/dped" / names[key])


def download_cie() -> None:
    jobs = [
        (f"https://files.cie.co.at/{name}", ROOT / "data/calibration/cie" / name)
        for name in CIE_FILES
    ]
    run_download(jobs, workers=4)


def download_camera_spectral() -> None:
    jobs = [(url, ROOT / rel) for url, rel in CAMERA_SPECTRAL_JOBS]
    tokyo_paths = [f"files/camera_{idx}.spectra" for idx in range(12)]
    for idx in range(12):
        name = f"camera_{idx}.spectra"
        jobs.append(
            (
                urllib.parse.urljoin(TOKYO_SPECTRAL_BASE, f"files/{name}"),
                ROOT / "data/calibration/camera_spectral/tokyo_open_vision" / name,
            )
        )
    jobs.append(
        (
            urllib.parse.urljoin(TOKYO_SPECTRAL_BASE, "database.html"),
            ROOT / "data/calibration/camera_spectral/tokyo_open_vision/index.html",
        )
    )
    run_download(jobs, workers=6)
    paths_file = ROOT / "data/calibration/camera_spectral/tokyo_open_vision/spectra_paths.txt"
    paths_file.write_text("\n".join(tokyo_paths) + "\n", encoding="utf-8")


def download_dorf() -> None:
    out = ROOT / "data/calibration/film_response/cave_dorf/dorfCurves.zip"
    path, _changed, status = download_one(DORF_ARCHIVE, out)
    print(f"{status} {path}", flush=True)


def list_fivek_expert(expert: str, limit: int | None = None) -> list[str]:
    index = ROOT / "data/raw/fivek/fivek_index.html"
    if index.exists():
        text = index.read_text(encoding="utf-8")
    else:
        text = fetch_bytes(FIVEK_BASE).decode("utf-8", errors="replace")
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(text, encoding="utf-8")
    parser = FiveKParser(expert)
    parser.feed(text)
    paths = parser.paths
    if limit:
        paths = paths[:limit]
    return paths


def download_fivek_expert(expert: str, workers: int, limit: int | None) -> None:
    paths = list_fivek_expert(expert, limit)
    jobs = []
    for path in paths:
        url = urllib.parse.urljoin(FIVEK_BASE, urllib.parse.quote(path, safe="/"))
        name = Path(path).name
        jobs.append((url, ROOT / f"data/raw/fivek/expert_tiff/{expert}" / name))
    run_download(jobs, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "physics",
            "cie",
            "camera-spectral",
            "dorf",
            "filmset",
            "filmset-zip",
            "fivek-dng",
            "fivek-expert",
            "dped",
        ],
    )
    parser.add_argument("--expert", choices=["a", "b", "c", "d", "e"], default="c")
    parser.add_argument("--dped-part", choices=["sample", "patches", "original", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.dataset == "physics":
        download_physics()
    elif args.dataset == "cie":
        download_cie()
    elif args.dataset == "camera-spectral":
        download_camera_spectral()
    elif args.dataset == "dorf":
        download_dorf()
    elif args.dataset == "filmset":
        download_filmset(args.workers, args.limit)
    elif args.dataset == "filmset-zip":
        download_filmset_zip()
    elif args.dataset == "fivek-dng":
        download_fivek_dng()
    elif args.dataset == "dped":
        download_dped(args.dped_part)
    else:
        download_fivek_expert(args.expert, args.workers, args.limit)


if __name__ == "__main__":
    main()
