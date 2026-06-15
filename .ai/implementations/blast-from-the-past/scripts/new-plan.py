#!/usr/bin/env python3
from __future__ import annotations

import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def slugify(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-") or "plan"

def main():
    p = argparse.ArgumentParser(); p.add_argument("title"); p.add_argument("--date", default="2026-06-15")
    a = p.parse_args(); slug = slugify(a.title)
    path = ROOT / "plans" / f"{a.date}_{slug}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "plan-template.md").read_text().replace("{{YYYY-MM-DD}}_{{slug}}", f"{a.date}_{slug}").replace("{{Title}}", a.title).replace("{{ISO_TIMESTAMP}}", f"{a.date}T00:00:00 CET").replace("{{AGENT_ID}}", "codex")
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
