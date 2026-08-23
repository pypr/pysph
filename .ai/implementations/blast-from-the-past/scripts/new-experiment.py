#!/usr/bin/env python3
from __future__ import annotations

import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def slugify(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-") or "experiment"
def main():
    p = argparse.ArgumentParser(); p.add_argument("title"); p.add_argument("--aspect", default="validation-benchmarks"); p.add_argument("--date", default="2026-06-15")
    a = p.parse_args(); slug = slugify(a.title); d = ROOT / "experiments" / f"{a.date}_{slug}"
    if d.exists(): raise SystemExit(f"Refusing to overwrite {d}")
    (d / "plots").mkdir(parents=True)
    text = (ROOT / "templates" / "experiment-template.md").read_text().replace("{{YYYY-MM-DD}}_{{slug}}", f"{a.date}_{slug}").replace("{{ISO_TIMESTAMP}}", f"{a.date}T07:19:08 CET").replace("{{aspect-name}}", a.aspect).replace("{{Title}}", a.title)
    (d / "experiment.md").write_text(text); print(d); return 0
if __name__ == "__main__": raise SystemExit(main())
