#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--date", default="2026-06-15")
    a = p.parse_args(); path = ROOT / "updates" / "daily" / f"{a.date}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "daily-closeout-template.md").read_text().replace("{{YYYY-MM-DD}}", a.date).replace("{{AGENT_ID}}", "codex").replace("{{N}}", "0")
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
