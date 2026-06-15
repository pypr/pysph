#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--week", default="2026-W25"); p.add_argument("--range", default="2026-06-15 to 2026-06-21")
    a = p.parse_args(); path = ROOT / "updates" / "weekly" / f"{a.week}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "weekly-closeout-template.md").read_text().replace("{{YYYY-Www}}", a.week).replace("{{YYYY-MM-DD}} to {{YYYY-MM-DD}}", a.range)
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
