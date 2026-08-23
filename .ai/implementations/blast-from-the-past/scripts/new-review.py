#!/usr/bin/env python3
from __future__ import annotations

import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def slugify(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-") or "review"
def main():
    p = argparse.ArgumentParser(); p.add_argument("title"); p.add_argument("--date", default="2026-06-15")
    a = p.parse_args(); path = ROOT / "reviews" / f"{a.date}_{slugify(a.title)}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "review-template.md").read_text().replace("{{Title}}", a.title).replace("{{YYYY-MM-DD}}", a.date).replace("{{AGENT_ID}}", "codex")
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
