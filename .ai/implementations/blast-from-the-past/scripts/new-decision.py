#!/usr/bin/env python3
"""Create a new ADR without overwriting existing files."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "decisions"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "decision"


def next_id() -> tuple[str, str]:
    max_id = 0
    for path in DECISIONS.glob("*adr-*.md"):
        m = re.search(r"adr-(\d{4})", path.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    n = max_id + 1
    return f"ADR-{n:04d}", f"{n:04d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("--scope", default="global")
    parser.add_argument("--status", default="Proposed")
    parser.add_argument("--date", default="2026-06-15")
    parser.add_argument("--author", default="@kunalpuri-prediqt")
    args = parser.parse_args()

    DECISIONS.mkdir(parents=True, exist_ok=True)
    adr_id, nnnn = next_id()
    slug = slugify(args.title)
    path = DECISIONS / f"{args.date}_adr-{nnnn}_{slug}.md"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    text = f"""---
type: decision
id: {adr_id}
date: {args.date}
author: {args.author}
scope: {args.scope}
status: {args.status}
supersedes: []
relates_to: []
depends_on: []
conflicts_with: []
---

# {adr_id}: {args.title}

## Context

Confirm with team.

## Decision

Confirm with team.

## Rationale

Confirm with team.

## Alternatives considered

Confirm with team.

## Consequences

Confirm with team.

## Follow-ups

- Confirm with team.
"""
    path.write_text(text)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
