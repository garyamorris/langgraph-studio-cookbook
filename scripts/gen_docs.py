"""Generate docs/CATALOG.md and docs/examples/<id>.md from the code itself.

    .venv/bin/python scripts/gen_docs.py

For every graph in langgraph.json this imports the module and renders: the META block, the module docstring's
sections (PROBLEM, DESIGN, ...), the example input (and resume values), and a Mermaid diagram drawn from the
*compiled graph* - so the pages cannot drift from the code. docs/CATALOG.md groups examples by category and adds
an index by LangGraph feature and a list of the examples that use the OpenAI Decisions API.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DEEPDIVE_OFFLINE", "1")  # importing must never need an API key

from deepdive.common.meta import CATEGORY_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SECTION_RE = re.compile(r"^([A-Z][A-Z0-9 &/\-]+)\s*$")
TITLES = {
    "PROBLEM": "Problem",
    "DESIGN": "Design",
    "GRAPH": "Graph",
    "LANGGRAPH FEATURES": "LangGraph features",
    "MODELS AND DATA": "Models and data",
    "TRY IT IN STUDIO": "Try it in Studio",
    "EXTEND IT": "Extend it",
}


def load_registry() -> dict[str, str]:
    return json.loads((ROOT / "langgraph.json").read_text())["graphs"]


def module_of(target: str):
    path = target.split(":")[0].removeprefix("./src/").removesuffix(".py").replace("/", ".")
    return importlib.import_module(path)


def parse_sections(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary line, {SECTION: body}). Sections are ALL-CAPS lines at column 0."""
    lines = (doc or "").strip("\n").splitlines()
    summary = lines[0].strip() if lines else ""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        m = SECTION_RE.match(line)
        if m and not line.startswith(" "):
            current = m.group(1).strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    out = {}
    for k, v in sections.items():
        body = "\n".join(v)
        # dedent by the smallest indentation of non-blank lines
        indents = [len(x) - len(x.lstrip()) for x in v if x.strip()]
        cut = min(indents) if indents else 0
        out[k] = "\n".join(x[cut:] if x.strip() else "" for x in v).strip("\n")
    return summary, out


def mermaid(mod) -> str:
    try:
        g = mod.graph.get_graph(xray=1)
    except Exception:  # noqa: BLE001 - entrypoints / exotic graphs may not support xray
        g = mod.graph.get_graph()
    return g.draw_mermaid()


def render_example(gid: str, mod) -> str:
    meta = mod.META
    summary, sections = parse_sections(mod.__doc__ or "")
    out = [f"# {meta.title}", "", f"> {meta.problem}", ""]
    out += [
        "| | |", "|---|---|",
        f"| **Example id** | `{gid}` |",
        f"| **Category** | {meta.category} |",
        f"| **Difficulty** | {meta.difficulty} |",
        f"| **Models** | {', '.join(meta.models)} |",
        f"| **Data** | {', '.join(f'`{d}`' for d in meta.datasets) or '-'} |",
        f"| **LangGraph features** | {', '.join(meta.features)} |",
        f"| **Source** | [`src/deepdive/examples/{gid}.py`](../../src/deepdive/examples/{gid}.py) |",
        "", "---", "",
    ]
    for name, body in sections.items():
        out.append(f"## {TITLES.get(name, name.capitalize())}")
        out.append("")
        if name == "GRAPH":
            out += ["```text", body, "```"]
        else:
            out.append(body.replace("``", "`"))  # docstrings use RST-style double backticks; Markdown wants one
        out.append("")
    out += ["## Compiled graph", "", "Drawn from the compiled graph by `scripts/gen_docs.py` (one level of subgraphs expanded).", "",
            "```mermaid", mermaid(mod), "```", ""]
    out += ["## Example input", "", "```json", json.dumps(mod.EXAMPLE_INPUT, indent=2, default=str), "```", ""]
    resumes = getattr(mod, "EXAMPLE_RESUMES", None)
    if resumes:
        out += ["Resume values (answers to the interrupts, in order):", "", "```json", json.dumps(resumes, indent=2, default=str), "```", ""]
    out += ["Run it: `uv run langgraph dev`, open Studio, choose `" + gid + "`, paste the input above.", ""]
    return "\n".join(out)


def main() -> int:
    registry = load_registry()
    mods = {gid: module_of(t) for gid, t in sorted(registry.items())}
    (DOCS / "examples").mkdir(parents=True, exist_ok=True)
    for old in (DOCS / "examples").glob("*.md"):
        old.unlink()

    by_cat: dict[str, list[str]] = defaultdict(list)
    for gid, mod in mods.items():
        (DOCS / "examples" / f"{gid}.md").write_text(render_example(gid, mod), encoding="utf-8")
        by_cat[mod.META.category].append(gid)

    lines = [
        "# Example catalog", "",
        f"{len(mods)} examples. Each one solves a concrete problem; open its page for the problem statement, the design "
        "rationale, a diagram of the compiled graph, and the exact input to paste into Studio.", "",
    ]
    for cat in CATEGORY_ORDER:
        ids = by_cat.get(cat)
        if not ids:
            continue
        lines += [f"## {cat}", "", "| Example | Problem it solves | Key LangGraph features | Models |", "|---|---|---|---|"]
        for gid in ids:
            m = mods[gid].META
            lines.append(f"| [`{gid}`](examples/{gid}.md) | {m.problem} | {', '.join(m.features)} | {', '.join(m.models)} |")
        lines.append("")

    feats: dict[str, list[str]] = defaultdict(list)
    for gid, mod in mods.items():
        for f in mod.META.features:
            feats[f.lower()].append(gid)
    lines += ["## Index by LangGraph feature", "", "| Feature | Examples |", "|---|---|"]
    for f in sorted(feats):
        lines.append(f"| {f} | {', '.join(f'[`{g}`](examples/{g}.md)' for g in feats[f])} |")
    lines.append("")

    dec = [g for g, m in mods.items() if any("decisions" in x.lower() for x in m.META.models)]
    lines += ["## Examples that use the OpenAI Decisions API", ""]
    lines += [f"* [`{g}`](examples/{g}.md): {mods[g].META.problem}" for g in dec] or ["(none)"]
    lines.append("")
    (DOCS / "CATALOG.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote docs/CATALOG.md and {len(mods)} pages under docs/examples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
