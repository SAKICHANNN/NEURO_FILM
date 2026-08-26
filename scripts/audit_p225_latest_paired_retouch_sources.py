"""Run the offline, metadata-only P225 latest paired-retouch source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.latest_paired_retouch_source_audit import (
    LatestPairedRetouchSourceAuditError,
    analyze_latest_paired_retouch_sources,
    canonical_json_bytes,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _verify_repo(repo: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != candidate["official_commit"]:
        raise LatestPairedRetouchSourceAuditError(
            f"official commit mismatch for {candidate['candidate_id']}"
        )
    blobs: dict[str, str] = {}
    for relative, expected in sorted(candidate["required_blobs"].items()):
        actual = _git(repo, "rev-parse", f"HEAD:{relative}")
        if actual != expected:
            raise LatestPairedRetouchSourceAuditError(
                f"Git blob mismatch for {candidate['candidate_id']}:{relative}"
            )
        blobs[relative] = actual
    return {"commit": commit, "blobs": blobs}


def _verify_pdf(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise LatestPairedRetouchSourceAuditError(f"PDF identity mismatch: {label}")


def _pdf_text(pdf: Path, work: Path) -> str:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise LatestPairedRetouchSourceAuditError("pdftotext is required")
    output = work / f"{pdf.stem}.txt"
    subprocess.run([executable, "-layout", str(pdf), str(output)], check=True)
    return output.read_text(encoding="utf-8", errors="replace")


def run(config_path: Path, source_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = {row["candidate_id"]: row for row in config["candidates"]}
    pixtalk = candidates["pixtalk_iccv2025"]
    retouchiq = candidates["retouchiq_cvpr2026"]
    instant = candidates["instantretouch_cvpr2026"]

    pixtalk_repo = source_root / "pixtalk"
    instant_repo = source_root / "instantretouch"
    source_identity = {
        "pixtalk_iccv2025": _verify_repo(pixtalk_repo, pixtalk),
        "instantretouch_cvpr2026": _verify_repo(instant_repo, instant),
    }

    pdf_paths = {
        "retouchiq_paper": source_root / "retouchiq.pdf",
        "retouchiq_supplement": source_root / "retouchiq_supp.pdf",
        "instantretouch_paper": source_root / "instantretouch.pdf",
        "instantretouch_supplement": source_root / "instantretouch_supp.pdf",
    }
    _verify_pdf(pdf_paths["retouchiq_paper"], retouchiq["paper_sha256"], "RetouchIQ paper")
    _verify_pdf(
        pdf_paths["retouchiq_supplement"],
        retouchiq["supplement_sha256"],
        "RetouchIQ supplement",
    )
    _verify_pdf(pdf_paths["instantretouch_paper"], instant["paper_sha256"], "InstantRetouch paper")
    _verify_pdf(
        pdf_paths["instantretouch_supplement"],
        instant["supplement_sha256"],
        "InstantRetouch supplement",
    )
    source_identity["retouchiq_cvpr2026"] = {
        "paper_sha256": retouchiq["paper_sha256"],
        "supplement_sha256": retouchiq["supplement_sha256"],
    }
    source_identity["instantretouch_cvpr2026"].update(
        {
            "paper_sha256": instant["paper_sha256"],
            "supplement_sha256": instant["supplement_sha256"],
        }
    )

    with tempfile.TemporaryDirectory(prefix="nf_p225_pdf_text_") as temp:
        text_root = Path(temp)
        scientific = analyze_latest_paired_retouch_sources(
            config=config,
            pixtalk_readme=(pixtalk_repo / "README.md").read_text(encoding="utf-8"),
            retouchiq_paper_text=_pdf_text(pdf_paths["retouchiq_paper"], text_root),
            retouchiq_supplement_text=_pdf_text(
                pdf_paths["retouchiq_supplement"], text_root
            ),
            instant_readme=(instant_repo / "README.md").read_text(encoding="utf-8"),
            instant_license=(instant_repo / "LICENSE").read_text(encoding="utf-8"),
            instant_dataset_loader=(instant_repo / "dataset/dataset_5kreq.py").read_text(
                encoding="utf-8"
            ),
            instant_paper_text=_pdf_text(pdf_paths["instantretouch_paper"], text_root),
            instant_supplement_text=_pdf_text(
                pdf_paths["instantretouch_supplement"], text_root
            ),
            instant_repository_tree=tuple(
                _git(instant_repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
            ),
        )
    return {
        **scientific,
        "source_identity": source_identity,
        "formal_source_root_persisted": False,
        "network_reads_during_formal_audit": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/p225_latest_paired_retouch_source_audit_v1.json",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
