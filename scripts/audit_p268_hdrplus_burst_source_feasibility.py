from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class P268Error(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "neuro-film-p268/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _visible_text(body: bytes) -> str:
    source = body.decode("utf-8", errors="strict")
    source = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    source = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", source)).split())


def _gcloud_path() -> str:
    executable = shutil.which("gcloud.cmd")
    if executable is None:
        raise P268Error("installed gcloud.cmd is unavailable")
    return executable


def _gcloud_json(*arguments: str) -> list[dict[str, Any]]:
    process = subprocess.run(
        [_gcloud_path(), "storage", "ls", "--json", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if process.returncode != 0:
        raise P268Error(
            f"public bucket metadata listing failed: {process.stderr.strip()}"
        )
    value = json.loads(process.stdout)
    if not isinstance(value, list):
        raise P268Error("gcloud JSON listing is not a list")
    return value


def _burst_ids(rows: list[dict[str, Any]], prefix: str) -> list[str]:
    ids = []
    for row in rows:
        if row.get("type") != "prefix":
            raise P268Error("burst root contains a non-prefix row")
        url = str(row.get("url", ""))
        if not url.startswith(prefix) or not url.endswith("/"):
            raise P268Error("burst prefix escaped the frozen root")
        burst_id = url[len(prefix) : -1]
        if not burst_id or "/" in burst_id or SAFE_ID.fullmatch(burst_id) is None:
            raise P268Error("unsafe burst ID")
        ids.append(burst_id)
    if len(ids) != len(set(ids)):
        raise P268Error("duplicate burst ID")
    return sorted(ids)


def _selected_burst(ids: list[str]) -> tuple[str, str]:
    ranked = sorted((_sha256_bytes(value.encode("utf-8")), value) for value in ids)
    if not ranked:
        raise P268Error("empty burst inventory")
    digest, burst_id = ranked[0]
    return burst_id, digest


def _objects(rows: list[dict[str, Any]], expected_prefix: str) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if row.get("type") == "prefix":
            continue
        if row.get("type") != "cloud_object" or not isinstance(
            row.get("metadata"), dict
        ):
            raise P268Error("unexpected selected-object listing row")
        metadata = row["metadata"]
        name = str(metadata.get("name", ""))
        if not name.startswith(expected_prefix) or name.endswith("/"):
            raise P268Error("selected object escaped its frozen prefix")
        relative = name[len(expected_prefix) :]
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise P268Error("unsafe selected-object relative path")
        normalized.append(
            {
                "relative_path": relative,
                "bytes": int(metadata["size"]),
                "generation": str(metadata["generation"]),
                "metageneration": str(metadata["metageneration"]),
                "crc32c": str(metadata["crc32c"]),
                "md5_base64": str(metadata.get("md5Hash", "")),
                "content_type": str(metadata.get("contentType", "")),
                "updated": str(metadata["updated"]),
            }
        )
    return sorted(normalized, key=lambda row: row["relative_path"])


def _page_facts(dataset_body: bytes, paper_body: bytes) -> dict[str, Any]:
    dataset_text = _visible_text(dataset_body)
    paper_text = _visible_text(paper_body)
    required_dataset = [
        "3640 bursts",
        "28461 images",
        "153 bursts, 37 GiB",
        "765 GiB",
        "payload_N<frame>.dng",
        "lens_shading_map_N<frame>.tiff",
        "rgb2rgb.txt",
        "merged.dng",
        "final.jpg",
        "reference_frame.txt",
        "Creative Commons license (CC-BY-SA)",
    ]
    required_paper = [
        "do not use bracketed exposures",
        "frames of constant exposure",
        "Bayer raw frames",
        "Camera2 API",
    ]
    return {
        "dataset_page_bytes": len(dataset_body),
        "dataset_page_sha256": _sha256_bytes(dataset_body),
        "paper_page_bytes": len(paper_body),
        "paper_page_sha256": _sha256_bytes(paper_body),
        "dataset_required_statements": {
            token: token in dataset_text for token in required_dataset
        },
        "paper_required_statements": {
            token: token in paper_text for token in required_paper
        },
        "cc_by_sa_4_link_present": (
            "https://creativecommons.org/licenses/by-sa/4.0/"
            in dataset_body.decode("utf-8")
        ),
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] not in {
        "SOURCE_SELECTION_LOCKED_BEFORE_FORMAL_OBJECT_AUDIT",
        "FORMAL_EXECUTION_LOCKED",
    }:
        raise P268Error("P268 source selection is not locked")
    official = config["official"]
    dataset_body = _fetch(official["dataset_url"])
    paper_body = _fetch(official["paper_url"])
    prefix_rows = _gcloud_json(official["burst_prefix"])
    if reverse:
        prefix_rows.reverse()
    burst_ids = _burst_ids(prefix_rows, official["burst_prefix"])
    selected, selected_sha = _selected_burst(burst_ids)
    if selected != config["selection"]["selected_burst_id"]:
        raise P268Error("hash-ranked selected burst differs")
    if selected_sha != config["selection"]["selected_burst_id_sha256"]:
        raise P268Error("selected burst hash differs")

    roots = {
        "burst": f"{official['burst_prefix']}{selected}/",
        "result_20161014": f"{official['result_prefixes'][0]}{selected}/",
        "result_20171023": f"{official['result_prefixes'][1]}{selected}/",
    }
    inventories = {}
    for role, url in roots.items():
        rows = _gcloud_json("--recursive", url)
        if reverse:
            rows.reverse()
        prefix = url.removeprefix("gs://hdrplusdata/")
        inventories[role] = _objects(rows, prefix)

    burst_names = [row["relative_path"] for row in inventories["burst"]]
    payload_names = [
        name for name in burst_names if re.fullmatch(r"payload_N\d{3}\.dng", name)
    ]
    lens_names = [
        name
        for name in burst_names
        if re.fullmatch(r"lens_shading_map_N\d{3}\.tiff", name)
    ]
    result_role_gates = {}
    for role in ("result_20161014", "result_20171023"):
        names = {row["relative_path"] for row in inventories[role]}
        result_role_gates[role] = all(
            name in names for name in config["required_result_roles"]
        )
    selected_total = sum(
        row["bytes"] for inventory in inventories.values() for row in inventory
    )
    page_facts = _page_facts(dataset_body, paper_body)
    gates = {
        "official_page_facts": (
            all(page_facts["dataset_required_statements"].values())
            and all(page_facts["paper_required_statements"].values())
            and page_facts["cc_by_sa_4_link_present"]
        ),
        "burst_inventory": (
            len(burst_ids) == official["curated_bursts"]
            and len(set(burst_ids)) == official["curated_bursts"]
        ),
        "hash_ranked_selection": selected == config["selection"]["selected_burst_id"],
        "burst_roles": (
            len(payload_names)
            >= config["required_burst_roles"]["minimum_payload_dng_count"]
            and bool(lens_names)
            and "rgb2rgb.txt" in burst_names
        ),
        "result_roles": all(result_role_gates.values()),
        "fixed_object_identity": all(
            row["bytes"] > 0
            and row["generation"]
            and row["metageneration"]
            and row["crc32c"]
            for inventory in inventories.values()
            for row in inventory
        ),
        "selected_total_within_budget": selected_total
        <= config["budgets"]["maximum_selected_burst_and_matching_results_bytes"],
        "zero_payload_or_pixel_reads": True,
    }
    status = (
        "PASS_PRIVATE_HDRPLUS_ONE_BURST_SOURCE_FEASIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_HDRPLUS_ONE_BURST_SOURCE_FEASIBILITY"
    )
    scientific = {
        "status": status,
        "official_pages": page_facts,
        "burst_inventory": {
            "count": len(burst_ids),
            "inventory_sha256": _sha256_bytes(_canonical_bytes(burst_ids)),
        },
        "selection": {
            "burst_id": selected,
            "burst_id_sha256": selected_sha,
        },
        "inventories": inventories,
        "roles": {
            "payload_dng_count": len(payload_names),
            "lens_shading_map_count": len(lens_names),
            "rgb2rgb_present": "rgb2rgb.txt" in burst_names,
            "timing_present": "timing.txt" in burst_names,
            "result_role_gates": result_role_gates,
        },
        "selected_object_count": sum(len(value) for value in inventories.values()),
        "selected_total_bytes": selected_total,
        "gates": gates,
    }
    return {
        "schema": "neuro-film.p268-hdrplus-burst-source-feasibility-result.v1",
        "experiment_id": "P268",
        "status": status,
        "scientific": scientific,
        "identities": {
            "config_bytes": config_path.stat().st_size,
            "config_sha256": _sha256_file(config_path),
            "contract_sha256": _sha256_file(
                ROOT / "docs/planning/P268_HDRPLUS_BURST_SOURCE_FEASIBILITY_CONTRACT.md"
            ),
            "runner_sha256": _sha256_file(Path(__file__)),
        },
        "network_request_classes": [
            "official_dataset_page",
            "official_paper_page",
            "public_gcs_metadata_api",
        ],
        "object_payload_body_bytes_read": 0,
        "dng_jpeg_tiff_pixel_decodes": 0,
        "fit_train_inference_score_reads": 0,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    value = execute(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
