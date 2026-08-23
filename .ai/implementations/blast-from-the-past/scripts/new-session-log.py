#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--date", default="2026-06-15"); p.add_argument("--start", default="07:19")
    a = p.parse_args(); path = ROOT / "updates" / "session-logs" / f"{a.date}_{a.start.replace(':','')}.md"
    if path.exists(): raise SystemExit(f"Refusing to overwrite {path}")
    text = (ROOT / "templates" / "session-log-template.md").read_text().replace("{{YYYY-MM-DD}}", a.date).replace("{{HH:MM}}", a.start, 1).replace("{{HH:MM}}", a.start).replace("{{AGENT_ID}}", "codex")
    path.write_text(text); print(path); return 0
if __name__ == "__main__": raise SystemExit(main())
