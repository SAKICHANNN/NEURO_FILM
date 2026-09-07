"""Bounded, create-only acquisition of a single licensed development series."""

import hashlib
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.urls.append(dict(attrs).get("src", ""))


def fetch(url, cap):
    with urlopen(
        Request(url, headers={"User-Agent": "K-MCFM research intake"}), timeout=40
    ) as r:
        if r.url != url:
            raise ValueError("unexpected redirect")
        data = r.read(cap + 1)
    if len(data) > cap:
        raise ValueError("download budget exceeded")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    config_bytes = (ROOT / "configs/ai_film_reference_italy_v1.json").read_bytes()
    c = json.loads(config_bytes)
    destination = ROOT / c["destination"]
    if destination.resolve().drive.upper() != "P:":
        raise ValueError("this acquisition requires canonical P-backed storage")
    if destination.exists():
        raise FileExistsError(destination)
    page = fetch(c["page"], 1024 * 1024)
    html = page.decode("utf-8")
    if (
        "creativecommons.org/licenses/by/4.0/" not in html
        or "The works above are licensed under" not in html
    ):
        raise ValueError("explicit works license missing")
    if "Captured with a Nikon FM-2 on Kodak Portra 400 film." not in html:
        raise ValueError("source attribution drift")
    parser = Images()
    parser.feed(html)
    urls = [u for u in parser.urls if u.startswith(c["image_prefix"])]
    expected = [f"{c['image_prefix']}{i:02d}.jpg" for i in c["ids"]]
    if urls != expected:
        raise ValueError("source image list drift")
    # URL-only historical comparison never decodes sealed assessment images.
    old = (
        ROOT
        / "data/real_film/sf3_a3u_portra_three_source_heldout_v1/nicknick/manifest.json"
    )
    history = json.loads(old.read_text(encoding="utf-8"))
    old_hashes = {r["sha256"] for r in history["rows"]}
    if set(urls) & {r["url"] for r in history["rows"]}:
        raise ValueError("historical URL overlap")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source.html").write_bytes(page)
    rows, seen, total = [], set(), 0
    for index, url in zip(c["ids"], urls, strict=True):
        payload = fetch(url, min(c["max_file_bytes"], c["max_total_bytes"] - total))
        digest = sha(payload)
        if digest in seen or digest in old_hashes:
            raise ValueError("duplicate or historical byte overlap")
        with Image.open(io.BytesIO(payload)) as im:
            if im.format != "JPEG" or im.width * im.height > 40_000_000:
                raise ValueError("invalid JPEG dimensions")
            size = list(im.size)
            im.verify()
        name = f"{index:02d}.jpg"
        with (destination / name).open("xb") as f:
            f.write(payload)
        total += len(payload)
        seen.add(digest)
        rows.append(
            {
                "path": name,
                "url": url,
                "sha256": digest,
                "bytes": len(payload),
                "size": size,
            }
        )
        print(f"{len(rows)}/{len(urls)} bytes={total}", flush=True)
    report = dict(
        c,
        config_sha256=sha(config_bytes),
        page_sha256=sha(page),
        rows=rows,
        total_bytes=total,
        historical_byte_overlap=0,
        historical_pixel_reads=0,
        perceptual_overlap="not_yet_checked",
    )
    with (destination / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("COMPLETE", sha((destination / "manifest.json").read_bytes()))


if __name__ == "__main__":
    main()
