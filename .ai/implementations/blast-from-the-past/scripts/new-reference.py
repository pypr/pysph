#!/usr/bin/env python3
from __future__ import annotations

import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def slugify(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-") or "reference"
def main():
    p = argparse.ArgumentParser(); p.add_argument("title"); p.add_argument("--kind", default="primary")
    a = p.parse_args(); slug = slugify(a.title); d = ROOT / "references" / ("primary" if a.kind == "primary" else "secondary")
    path = d / f"{slug}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "reference-note-template.md").read_text().replace("{{slug}}", slug).replace("{{ISO_TIMESTAMP}}", "2026-06-15T07:19:08 CET").replace("{{Title}}", a.title).replace("kind: primary", f"kind: {a.kind}")
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
