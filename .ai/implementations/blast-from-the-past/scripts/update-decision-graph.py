#!/usr/bin/env python3
"""Regenerate and validate the ADR decision index and graph."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "decisions"
INDEX = DECISIONS / "index.json"
GRAPH = DECISIONS / "graph.md"
ADR_RE = re.compile(r"^ADR-\d{4}$")


def parse_scalar(value: str):
    value = value.strip()
    if value in ("[]", ""):
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [x.strip().strip("'\"") for x in inner.split(",")]
    return value.strip("'\"")


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter")
    data = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}: invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def scan():
    nodes = []
    for path in sorted(DECISIONS.glob("*.md")):
        if path.name in ("graph.md", "index.md"):
            continue
        fm = frontmatter(path)
        fm["file"] = str(path.relative_to(ROOT))
        nodes.append(fm)
    return nodes


def validate(nodes: list[dict]) -> list[str]:
    errors = []
    ids = [n.get("id") for n in nodes]
    seen = set()
    aspect_scopes = {p.name for p in (ROOT / "aspects").iterdir() if p.is_dir()}

    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not ADR_RE.match(node_id):
            errors.append(f"{node.get('file')}: invalid ADR id {node_id!r}")
        if node_id in seen:
            errors.append(f"duplicate ADR id {node_id}")
        seen.add(node_id)
        scope = node.get("scope")
        if scope != "global" and scope not in aspect_scopes:
            errors.append(f"{node_id}: invalid scope {scope!r}")
        if node.get("status") not in ("Proposed", "Accepted", "Superseded", "Rejected"):
            errors.append(f"{node_id}: invalid status {node.get('status')!r}")

    id_set = set(ids)
    by_id = {n.get("id"): n for n in nodes}
    edge_fields = ("supersedes", "relates_to", "depends_on", "conflicts_with")
    for node in nodes:
        node_id = node.get("id")
        for field in edge_fields:
            values = node.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                errors.append(f"{node_id}: {field} must be a list")
                continue
            for target in values:
                if target not in id_set:
                    errors.append(f"{node_id}: {field} target {target} does not exist")
                if field == "depends_on" and target in by_id:
                    if by_id[target].get("status") != "Accepted":
                        errors.append(
                            f"{node_id}: depends_on target {target} is "
                            f"{by_id[target].get('status')}, not Accepted"
                        )

    # Supersedes cycle check.
    supersedes = {}
    for node in nodes:
        vals = node.get("supersedes", [])
        if isinstance(vals, str):
            vals = [vals]
        supersedes[node.get("id")] = vals

    def visit(start, node_id, stack):
        for nxt in supersedes.get(node_id, []):
            if nxt == start or nxt in stack:
                errors.append(f"supersedes cycle involving {start}")
                return
            visit(start, nxt, stack | {nxt})

    for node_id in list(supersedes):
        visit(node_id, node_id, {node_id})

    return sorted(set(errors))


def render_index(nodes: list[dict]) -> str:
    out = {"nodes": [], "edges": []}
    for node in sorted(nodes, key=lambda n: n.get("id", "")):
        out["nodes"].append({
            "id": node.get("id"),
            "file": node.get("file"),
            "scope": node.get("scope"),
            "status": node.get("status"),
            "date": node.get("date"),
        })
        for field in ("supersedes", "relates_to", "depends_on", "conflicts_with"):
            vals = node.get(field, [])
            if isinstance(vals, str):
                vals = [vals]
            for target in vals:
                out["edges"].append({"from": node.get("id"), "to": target, "type": field})
    return json.dumps(out, indent=2, sort_keys=True) + "\n"


def render_graph(nodes: list[dict]) -> str:
    lines = [
        "# Decision Graph",
        "",
        "Generated from ADR frontmatter. Do not hand-edit.",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    scopes = sorted({n.get("scope") for n in nodes})
    for scope in scopes:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", str(scope))
        lines.append(f"  subgraph {safe}[{scope}]")
        for node in sorted([n for n in nodes if n.get("scope") == scope], key=lambda n: n.get("id")):
            node_id = node.get("id")
            status = node.get("status")
            label = f"{node_id}<br/>{status}"
            lines.append(f"    {node_id.replace('-', '_')}[\"{label}\"]")
        lines.append("  end")
    for node in sorted(nodes, key=lambda n: n.get("id", "")):
        src = node.get("id", "").replace("-", "_")
        edge_defs = {
            "depends_on": "-- depends_on -->",
            "relates_to": "-. relates_to .->",
            "supersedes": "-. supersedes .->",
            "conflicts_with": "-. conflicts_with .->",
        }
        for field, arrow in edge_defs.items():
            vals = node.get(field, [])
            if isinstance(vals, str):
                vals = [vals]
            for target in vals:
                lines.append(f"  {src} {arrow} {target.replace('-', '_')}")
    lines.extend([
        "  classDef Accepted fill:#d5f5d5,stroke:#2c7a2c;",
        "  classDef Proposed fill:#fff3bf,stroke:#9a7500;",
        "  classDef Superseded fill:#e5e7eb,stroke:#6b7280;",
        "  classDef Rejected fill:#ffd6d6,stroke:#b91c1c;",
    ])
    for node in nodes:
        lines.append(f"  class {node.get('id').replace('-', '_')} {node.get('status')};")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    DECISIONS.mkdir(parents=True, exist_ok=True)
    try:
        nodes = scan()
        errors = validate(nodes)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    index_text = render_index(nodes)
    graph_text = render_graph(nodes)
    if args.check:
        ok = True
        if not INDEX.exists() or INDEX.read_text() != index_text:
            print("ERROR: decisions/index.json is stale", file=sys.stderr)
            ok = False
        if not GRAPH.exists() or GRAPH.read_text() != graph_text:
            print("ERROR: decisions/graph.md is stale", file=sys.stderr)
            ok = False
        return 0 if ok else 1
    INDEX.write_text(index_text)
    GRAPH.write_text(graph_text)
    print(f"Generated {INDEX.relative_to(ROOT)} and {GRAPH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
