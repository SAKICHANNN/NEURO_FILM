import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tst100k_reference_switch_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    cfg = json.loads(CONFIG.read_text())
    assert (cfg["groups"], cfg["targets_per_source"], cfg["development_fit_groups"],
            cfg["development_check_groups"], cfg["seed"]) == (32, 2, 24, 8, 48)
    source, consumed = ROOT / cfg["manifest"], ROOT / cfg["consumed_candidates"]
    assert sha(source) == cfg["manifest_sha256"]
    assert sha(consumed) == cfg["consumed_candidates_sha256"]
    rows, old = json.loads(source.read_text()), json.loads(consumed.read_text())
    owners, labels = defaultdict(set), defaultdict(set)
    for row in rows:
        assert set(row) == {"content", "reference", "gt"}
        owners[row["gt"]].add(row["content"])
        labels[(row["content"], row["reference"])].add(row["gt"])
    assert all(len(v) == 1 for v in owners.values())
    owner = {k: next(iter(v)) for k, v in owners.items()}
    ambiguous = {key for key, targets in labels.items() if len(targets) > 1}
    assert len(ambiguous) == 108
    excluded = {r["content"] for r in old["triplets"]} | {owner[r["reference"]] for r in old["triplets"]}
    assert len(excluded) == 16
    by_source = defaultdict(list)
    for row in rows:
        if ((row["content"], row["reference"]) not in ambiguous
                and row["reference"] in owner and row["content"] not in excluded
                and owner[row["reference"]] not in excluded):
            by_source[row["content"]].append(row)

    def order(value):
        return hashlib.sha256((str(cfg["seed"]) + "|" + value).encode()).hexdigest()

    used, groups = set(excluded), []
    for content in sorted(by_source, key=order):
        if content in used:
            continue
        selected, targets, refowners = [], set(), set()
        for row in sorted(by_source[content], key=lambda r: order(json.dumps(r, sort_keys=True))):
            refowner = owner[row["reference"]]
            if refowner in used or refowner == content or refowner in refowners or row["gt"] in targets:
                continue
            selected.append({**row, "reference_owner": refowner})
            targets.add(row["gt"])
            refowners.add(refowner)
            if len(selected) == cfg["targets_per_source"]:
                break
        if len(selected) != cfg["targets_per_source"]:
            continue
        groups.append({"group_index": len(groups), "source_owner": content,
                       "split": "development_fit" if len(groups) < cfg["development_fit_groups"] else "development_check",
                       "triplets": selected})
        used |= {content} | refowners
        if len(groups) == cfg["groups"]:
            break
    assert len(groups) == 32
    role_owners = [{g["source_owner"], *(r["reference_owner"] for r in g["triplets"])} for g in groups]
    assert len(set.union(*role_owners)) == 96
    assert not set.union(*role_owners) & excluded
    fit_owners, check_owners = set.union(*role_owners[:24]), set.union(*role_owners[24:])
    assert not fit_owners & check_owners
    files = sorted({r[role] for g in groups for r in g["triplets"] for role in ("content", "reference", "gt")})
    existing = []
    for relative in files:
        path = ROOT / cfg["local_originals"] / relative
        if path.is_file():
            existing.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "groups.jsonl"
    manifest.write_text("".join(json.dumps(g, sort_keys=True) + "\n" for g in groups), encoding="utf-8")
    file_manifest = out / "files.json"
    file_manifest.write_text(json.dumps({"repository": cfg["repository"], "revision": cfg["revision"],
                                       "paths": files, "remote_file_sizes_and_sha256": "UNKNOWN_NO_HEAD_PERFORMED"}, indent=2) + "\n", encoding="utf-8")
    report = {"status": "METADATA_FROZEN_NOT_TRAINING_ADMISSION", "config_sha256": sha(CONFIG),
              "script_sha256": sha(Path(__file__)), "source_manifest_sha256": sha(source),
              "consumed_candidates_sha256": sha(consumed), "groups_sha256": sha(manifest),
              "files_sha256": sha(file_manifest), "groups": 32, "triplets": 64,
              "fit_groups": 24, "check_groups": 8, "distinct_role_owners": 96,
              "ambiguous_keys_excluded": len(ambiguous),
              "ambiguous_rows_excluded": sum((r["content"], r["reference"]) in ambiguous for r in rows),
              "consumed_owners_excluded": sorted(excluded), "cross_split_role_owner_overlap": 0,
              "required_distinct_files": len(files), "existing_in_admitted_local_originals": existing,
              "new_files_required": len(files) - len(existing), "new_bytes_required": None,
              "claim_ceiling": cfg["boundaries"], "new_pixels_read": 0, "downloads": 0, "head_requests": 0,
              "fits": 0, "training": 0}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare()
