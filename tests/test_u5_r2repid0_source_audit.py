from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u5_r2repid0_source import evaluate

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads((ROOT / "configs/u5_r2repid0_source_audit_v1.json").read_text())


def _fetcher(
    contract: dict, *, markup_drift: bool = False, repository_drift: bool = False
):
    source = contract["official_sources"]
    roles = ["original", "tiff16_a", "tiff16_b", "tiff16_c", "tiff16_d", "tiff16_e"]
    header = "name,left,right,mos\n"
    rows = []
    for scene_index in range(4981):
        pairs = [
            (left, right)
            for index, left in enumerate(roles)
            for right in roles[index + 1 :]
        ]
        if scene_index < 50:
            pairs = pairs[:10]
        rows.extend(
            f"scene-{scene_index:04d},{left},{right},{0.0 if pair_index == 0 else 1.0}\n"
            for pair_index, (left, right) in enumerate(pairs)
        )
    markup = (header + "".join(rows)).encode()
    expected = source["expected_files"]
    expected["processed_markup.csv"] = {
        "size": len(markup) + (1 if markup_drift else 0),
        "sha256": "drift"
        if markup_drift
        else __import__("hashlib").sha256(markup).hexdigest(),
    }
    license_md = (
        b"Photos are for research purposes only. Annotations use Creative Commons Attribution 4.0. "
        b"The underlying images remain restricted."
    )
    files_adobe = "\n".join(f"a-{index}" for index in range(2690)).encode()
    files_mit = "\n".join(f"m-{index}" for index in range(2310)).encode()
    payloads = {
        "README.md": b"README",
        "LICENSE.md": license_md,
        "LicenseAdobe.txt": b"Adobe",
        "LicenseAdobeMIT.txt": b"AdobeMIT",
        "filesAdobe.txt": files_adobe,
        "filesAdobeMIT.txt": files_mit,
        "processed_markup.csv": markup,
    }
    for name, data in payloads.items():
        if name != "processed_markup.csv":
            expected[name] = {
                "size": len(data),
                "sha256": __import__("hashlib").sha256(data).hexdigest(),
            }
    api = {
        "sha": "f" * 40 if repository_drift else source["expected_repository"]["sha"],
        "lastModified": source["expected_repository"]["last_modified"],
        "private": False,
        "gated": False,
        "cardData": {"license": "other"},
    }
    tree = []
    for name, facts in expected.items():
        tree.append(
            {
                "type": "file",
                "path": name,
                "size": facts["size"],
                "lfs": {"oid": facts["sha256"]},
            }
        )

    def fetch(url: str, maximum_bytes: int) -> bytes:
        if url == source["api_url"]:
            data = json.dumps(api).encode()
        elif url == source["tree_url"]:
            data = json.dumps(tree).encode()
        else:
            data = payloads[url.rsplit("/", 1)[-1]]
        assert len(data) <= maximum_bytes
        return data

    return fetch


def test_repid_audit_retains_aggregate_structure_only() -> None:
    contract = _contract()
    report = evaluate(contract, fetch=_fetcher(contract))
    assert report["automatic_pass"] is True
    assert report["annotation_structure"]["row_count"] == 74465
    assert report["annotation_structure"]["scene_count"] == 4981
    assert report["annotation_structure"]["annotation_rows_persisted"] is False
    assert report["requests"]["image_members"] == 0
    assert report["requests"]["operator_fits"] == 0
    assert report["rights_scope"]["product_dependency_allowed"] is False


def test_repid_audit_fails_on_markup_hash_drift() -> None:
    contract = _contract()
    report = evaluate(contract, fetch=_fetcher(contract, markup_drift=True))
    assert report["automatic_pass"] is False
    assert report["gates"]["exact_processed_markup_bytes"] is False


def test_repid_audit_fails_on_repository_revision_drift() -> None:
    contract = _contract()
    report = evaluate(contract, fetch=_fetcher(contract, repository_drift=True))
    assert report["automatic_pass"] is False
    assert report["gates"]["exact_repository_revision"] is False
