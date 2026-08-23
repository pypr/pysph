#!/usr/bin/env python3
"""Archive daily closeouts and session logs older than the configured window."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_WEEKS = 12
NOW = datetime(2026, 6, 15)


def parse_date_from_name(path: Path):
    for part in path.stem.split("_"):
        try:
            return datetime.strptime(part[:10], "%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.strptime(path.stem[:10], "%Y-%m-%d")
    except ValueError:
        return None


def collect():
    cutoff = NOW - timedelta(weeks=ARCHIVE_WEEKS)
    pairs = [
        (ROOT / "updates" / "daily", ROOT / "updates" / "archive" / "daily"),
        (ROOT / "updates" / "session-logs", ROOT / "updates" / "archive" / "session-logs"),
    ]
    moves = []
    for src_dir, dst_dir in pairs:
        for path in sorted(src_dir.glob("*.md")):
            d = parse_date_from_name(path)
            if d and d < cutoff:
                moves.append((path, dst_dir / path.name))
    return moves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    moves = collect()
    if not moves:
        print("compact: nothing to archive")
        return 0
    for src, dst in moves:
        print(f"{'would move' if args.dry_run else 'move'} {src} -> {dst}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
