import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_manifest = Path(config["source_manifest"])
    if digest(source_manifest) != config["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest hash mismatch")
    if config["pixel_decode_allowed"] or config["training_allowed"]:
        raise ValueError("This entry point admits metadata only")
    sources = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    if any(row["research_pool"] != "source_train" or row["domain"] != "input"
           or row["distributed_split"] != "train" for row in sources):
        raise ValueError("Only existing source_train identities are eligible")
    seed = config["selection_seed"]
    sources.sort(key=lambda row: hashlib.sha256(
        f"{seed}:{row['duplicate_cluster_id']}:{row['content_id']}".encode()
    ).hexdigest())
    selected = []
    clusters = set()
    total = config["development_train_groups"] + config["development_check_groups"]
    for row in sources:
        if row["duplicate_cluster_id"] not in clusters:
            selected.append(row)
            clusters.add(row["duplicate_cluster_id"])
        if len(selected) == total:
            break
    if len(selected) != total:
        raise ValueError("Insufficient distinct source_train clusters")
    desired_paths = set()
    for row in selected:
        name = Path(row["path"]).name
        desired_paths.add(f"train/input/{name}")
        desired_paths.update(f"train/{folder}/{name}" for folder in config["domains"].values())
    inventory = {}
    with Path(config["inventory_cache"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["path"] in desired_paths:
                inventory[row["path"]] = row
    if set(inventory) != desired_paths:
        raise ValueError("Selected paired path missing from existing inventory")
    root = Path(config["source_root"]).resolve()
    files = {}
    for relative in sorted(desired_paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "train"):
            raise ValueError("Path escapes distributed training data")
        cached = inventory[relative]
        sha = digest(path)
        if sha != cached["sha256"] or path.stat().st_size != cached["bytes"]:
            raise ValueError(f"Selected file integrity mismatch: {relative}")
        with Image.open(path) as header:
            if header.format != "PNG" or header.mode != "RGB":
                raise ValueError(f"Unexpected image header: {relative}")
            profile = header.info.get("icc_profile")
            files[relative] = {
                "path": relative, "sha256": sha, "bytes": cached["bytes"],
                "size": list(header.size), "mode": header.mode,
                "icc_sha256": hashlib.sha256(profile).hexdigest() if profile else None,
                "encoding_status": "ICC bytes recorded; colorimetric interpretation not verified",
            }
    pairs = []
    for index, source in enumerate(selected):
        name = Path(source["path"]).name
        before = files[f"train/input/{name}"]
        if before["sha256"] != source["sha256"]:
            raise ValueError("Source identity disagrees with frozen manifest")
        for edit, folder in config["domains"].items():
            after = files[f"train/{folder}/{name}"]
            if before["size"] != after["size"]:
                raise ValueError("Pair dimensions differ")
            pairs.append({
                "content_id": source["content_id"],
                "duplicate_cluster_id": source["duplicate_cluster_id"],
                "capture_group_status": config["capture_group_status"],
                "partition": "development_train" if index < config["development_train_groups"] else "development_check",
                "edit_id": f"filmset_capture_one:{edit}",
                "edit_identity_status": "paper-supported domain recipe; exact parameter equality unknown",
                "before": before, "after": after,
                "allowed_use": config["allowed_use"], "redistributable": False,
                "pair_alignment_status": "basename and dimensions matched; pixel alignment unverified",
                "portrait_status": "unreviewed", "appearance_status": "unreviewed",
            })
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "pairs.jsonl"
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs)
    if manifest.exists() and manifest.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("Refusing to replace a different admission manifest")
    manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    train_clusters = {row["duplicate_cluster_id"] for row in pairs if row["partition"] == "development_train"}
    check_clusters = {row["duplicate_cluster_id"] for row in pairs if row["partition"] == "development_check"}
    report = {
        "experiment_id": config["experiment_id"],
        "status": "METADATA_VERIFIED_NOT_TRAINING_ADMITTED",
        "config_sha256": digest(config_path), "script_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(source_manifest), "pairs_sha256": digest(manifest),
        "selected_groups": len(selected), "pair_rows": len(pairs), "verified_files": len(files),
        "development_train_groups": len(train_clusters), "development_check_groups": len(check_clusters),
        "cross_partition_duplicate_cluster_overlap": len(train_clusters & check_clusters),
        "icc_profile_hashes": sorted({row["icc_sha256"] or "absent" for row in files.values()}),
        "image_pixel_decodes": 0, "lockbox_manifests_opened": 0, "lockbox_image_files_opened": 0,
        "remaining": ["training-only portrait and target appearance review", "pixel alignment and encoding verification",
                      "capture lineage unavailable", "only three known recipe domains; no unseen-edit claim"],
        "claim_ceiling": config["claim_ceiling"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/filmset_paired_supervision_admission_v1.json"))
    print(json.dumps(prepare(parser.parse_args().config), indent=2))
