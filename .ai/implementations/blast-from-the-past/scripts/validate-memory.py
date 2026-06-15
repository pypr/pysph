#!/usr/bin/env python3
"""Validate the .ai memory system."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_USERS = {"@kunalpuri-prediqt", "kunalpuri-prediqt"}
REQUIRED = {
    "daily-closeout": ["type", "date", "user", "agent", "duration_minutes", "aspects_touched"],
    "weekly-closeout": ["type", "week", "range", "user"],
    "session-log": ["type", "date", "start", "end", "user", "agent", "aspects_touched", "memory_consulted"],
    "plan": ["type", "id", "author", "agent", "created", "status", "aspects", "host_files", "within_boundary"],
    "review": ["type", "date", "user", "agent", "plan", "adrs", "aspects_touched", "host_files", "status"],
    "decision": ["type", "id", "date", "author", "scope", "status", "supersedes", "relates_to", "depends_on", "conflicts_with"],
    "experiment": ["type", "id", "created", "author", "aspect", "status", "last_checked"],
    "reference-note": ["type", "id", "created", "author", "kind", "status", "aspects"],
}


def parse_scalar(value: str):
    value = value.strip()
    if value in ("[]", ""):
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [x.strip().strip("'\"") for x in inner.split(",")]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value.strip("'\"")


def read_fm(path: Path):
    text = path.read_text(errors="replace")
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end == -1:
        return None, text
    data = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = parse_scalar(v)
    return data, text


def all_markdown():
    return sorted(p for p in ROOT.rglob("*.md") if "/updates/archive/" not in str(p))


def parse_boundary_prefixes():
    path = ROOT / "implementation.md"
    text = path.read_text()
    m = re.search(r"## Integration boundary\n(?P<body>.*?)(?:\n## |\Z)", text, re.S)
    prefixes = []
    if not m:
        return prefixes
    for line in m.group("body").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].split(" - ", 1)[0].strip("` ")
        if item.endswith("/**/*.pxd"):
            prefixes.append((item[:-8], ".pxd"))
        elif item.endswith("/**/*.pyx"):
            prefixes.append((item[:-8], ".pyx"))
        else:
            prefixes.append((item, None))
    return prefixes


def in_boundary(path: str, prefixes) -> bool:
    for prefix, suffix in prefixes:
        if suffix:
            if path.startswith(prefix) and path.endswith(suffix):
                return True
        elif path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def parse_datetime(value: str):
    if not isinstance(value, str):
        return None
    cleaned = value.replace(" CET", "").replace(" CEST", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned[:19] if "T" in cleaned else cleaned[:10], fmt)
        except ValueError:
            pass
    return None


def validate() -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    artifacts = []

    for path in all_markdown():
        fm, text = read_fm(path)
        if not fm:
            continue
        typ = fm.get("type")
        if typ:
            artifacts.append((path, fm, text))
            for field in REQUIRED.get(typ, []):
                if field not in fm:
                    errors.append(f"{path}: missing required frontmatter field {field}")
            if typ in ("daily-closeout", "weekly-closeout", "session-log", "review"):
                if fm.get("user") not in ALLOWED_USERS:
                    errors.append(f"{path}: invalid user {fm.get('user')!r}")
            if typ == "decision" and fm.get("author") not in ALLOWED_USERS:
                errors.append(f"{path}: invalid author {fm.get('author')!r}")

    aspect_names = {p.name for p in (ROOT / "aspects").iterdir() if p.is_dir()}
    adr_ids = {}
    for path, fm, _ in artifacts:
        if fm.get("type") == "decision":
            adr_ids[fm.get("id")] = path

    boundary = parse_boundary_prefixes()
    for path, fm, text in artifacts:
        typ = fm.get("type")
        if typ == "plan":
            for aspect in fm.get("aspects", []):
                if aspect not in aspect_names:
                    errors.append(f"{path}: unknown aspect {aspect}")
            outside = [f for f in fm.get("host_files", []) if not in_boundary(f, boundary)]
            if outside and fm.get("within_boundary") is not False:
                errors.append(f"{path}: outside-boundary host_files require within_boundary: false: {outside}")
            if fm.get("status") == "approved" and ">" not in text.split("## Approval", 1)[-1]:
                errors.append(f"{path}: approved plan lacks verbatim quote block")
        if typ == "review":
            plan = fm.get("plan")
            if isinstance(plan, str) and plan.startswith(".ai/implementations/blast-from-the-past/plans/"):
                plan_path = ROOT.parents[2] / plan
                pfm, _ = read_fm(plan_path) if plan_path.exists() else (None, "")
                if not pfm:
                    errors.append(f"{path}: review plan does not resolve: {plan}")
                elif pfm.get("status") != "approved":
                    errors.append(f"{path}: review plan is not approved: {plan}")
            for adr in fm.get("adrs", []):
                if adr not in adr_ids:
                    errors.append(f"{path}: unknown ADR {adr}")
            if fm.get("status") == "lgtm" and ">" not in text.split("## Sign-off", 1)[-1]:
                errors.append(f"{path}: lgtm review lacks verbatim quote block")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "update-decision-graph.py"), "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        errors.append("decision graph check failed:\n" + proc.stdout.strip())

    secret_patterns = [
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{24,}"),
        re.compile(r"://[^/\s:@]+:[^/\s:@]+@"),
    ]
    for path in ROOT.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="ignore")
            for pat in secret_patterns:
                if pat.search(text):
                    errors.append(f"{path}: possible secret matched by validator")
                    break

    current = ROOT / "current.md"
    if current.exists():
        current_text = current.read_text()
        updated_match = re.search(r"Updated:\s*([^\n]+)", current_text)
        updated = parse_datetime(updated_match.group(1)) if updated_match else None
        dailies = list((ROOT / "updates" / "daily").glob("*.md"))
        if updated and dailies:
            newest_daily = max((parse_datetime(read_fm(p)[0].get("date")) for p in dailies if read_fm(p)[0]), default=None)
            if newest_daily and updated < newest_daily:
                warnings.append("current.md Updated timestamp predates newest daily closeout")

    now = datetime(2026, 6, 15, 7, 19, 8)
    for path, fm, _ in artifacts:
        if fm.get("type") == "experiment" and fm.get("status") == "running":
            last = parse_datetime(fm.get("last_checked"))
            if last and now - last > timedelta(hours=48):
                warnings.append(f"{path}: running experiment last_checked older than 48h")

    return errors, warnings


def main() -> int:
    errors, warnings = validate()
    if errors:
        print("validate-memory: FAILED")
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print("validate-memory: PASS")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
