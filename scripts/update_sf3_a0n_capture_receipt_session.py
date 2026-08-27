"""Record or inspect one copy-on-write SF3.A0N capture receipt session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_capture_session import (
    CONDITION_KIND,
    EXPOSURE_KIND,
    capture_session_progress,
    export_capture_session_csv,
    import_capture_session_csv,
    update_capture_session_batch,
    update_capture_session_row,
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a0n_three_stock_capture_receipts_v1.json",
    )
    parser.add_argument(
        "--stock",
        choices=("fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--status", action="store_true")
    action.add_argument("--condition-id")
    action.add_argument("--exposure-id")
    action.add_argument("--batch", type=Path)
    action.add_argument("--batch-csv", type=Path)
    action.add_argument("--export-csv", action="store_true")
    parser.add_argument("--values", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = _object(args.packet)
    if args.export_csv:
        if args.values is not None:
            parser.error("--export-csv does not accept --values")
        _write_new_text(
            args.output,
            export_capture_session_csv(
                args.contract, packet, root=ROOT, stock=args.stock
            ),
        )
        print(json.dumps({"format": "csv", "output": str(args.output)}, sort_keys=True))
        return 0
    if args.status:
        if args.values is not None:
            parser.error("--status does not accept --values")
        result = capture_session_progress(
            args.contract, packet, root=ROOT, stock=args.stock
        )
    elif args.batch is not None:
        if args.values is not None:
            parser.error("--batch does not accept --values")
        updates = json.loads(args.batch.read_bytes())
        if not isinstance(updates, list):
            parser.error("--batch must contain a JSON list")
        result = update_capture_session_batch(
            args.contract,
            packet,
            root=ROOT,
            stock=args.stock,
            updates=updates,
        )
    elif args.batch_csv is not None:
        if args.values is not None:
            parser.error("--batch-csv does not accept --values")
        result = import_capture_session_csv(
            args.contract,
            packet,
            args.batch_csv.read_text(encoding="utf-8-sig"),
            root=ROOT,
            stock=args.stock,
        )
    else:
        if args.values is None:
            parser.error("row update requires --values")
        result = update_capture_session_row(
            args.contract,
            packet,
            root=ROOT,
            stock=args.stock,
            kind=CONDITION_KIND if args.condition_id else EXPOSURE_KIND,
            slot_id=args.condition_id or args.exposure_id,
            values=_object(args.values),
        )
    _write_new(args.output, result)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
