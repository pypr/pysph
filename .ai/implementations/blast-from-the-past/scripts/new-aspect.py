#!/usr/bin/env python3
from __future__ import annotations

import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def slugify(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-") or "aspect"
def main():
    p = argparse.ArgumentParser(); p.add_argument("name")
    a = p.parse_args(); name = slugify(a.name); d = ROOT / "aspects" / name
    if d.exists(): raise SystemExit(f"Refusing to overwrite {d}")
    (d / "notes").mkdir(parents=True)
    text = (ROOT / "templates" / "aspect-context-template.md").read_text().replace("{{name}}", name).replace("{{ISO_TIMESTAMP}}", "2026-06-15T07:19:08 CET")
    (d / "context.md").write_text(text)
    (d / "open-questions.md").write_text(f"# Open Questions - {name}\n\n- (none yet)\n")
    (d / "known-issues.md").write_text(f"# Known Issues - {name}\n\n- (none yet)\n")
    print(d); return 0
if __name__ == "__main__": raise SystemExit(main())
