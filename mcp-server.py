from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Iterable

from mcp.server.fastmcp import FastMCP
from typing_extensions import TypedDict

ROOT = Path(__file__).parent.resolve()
DOCS_DIR = Path(ROOT / "docs" / "source")
# EXAMPLES_DIR = Path(os.getenv("BERMUDA_EXAMPLES_DIR", ROOT / "examples")).resolve()

mcp = FastMCP("bermuda-mcp")


# ---------- Resources ----------
# Let Claude attach docs with: @bermuda-mcp:bermuda-doc://path/to/file.md
@mcp.resource("bermuda-doc://{relpath}")
def read_doc(relpath: str) -> str:
    p = (DOCS_DIR / relpath).resolve()
    if DOCS_DIR not in p.parents and p != DOCS_DIR:
        raise FileNotFoundError("Path outside DOCS_DIR")
    return p.read_text(encoding="utf-8")


# # Let Claude attach examples with: @bermuda-mcp:bermuda-example://name.py
# @mcp.resource("bermuda-example://{name}")
# def read_example(name: str) -> str:
#     p = (EXAMPLES_DIR / name).with_suffix(".py").resolve()
#     if EXAMPLES_DIR not in p.parents:
#         raise FileNotFoundError("Path outside EXAMPLES_DIR")
#     return p.read_text(encoding="utf-8")
#


# ---------- Tools ----------
class SearchHit(TypedDict):
    path: str
    line: int
    snippet: str


def _walk_files(base: Path, exts: Iterable[str]) -> Iterable[Path]:
    for ext in exts:
        yield from base.rglob(f"*{ext}")


@mcp.tool()
def list_docs() -> list[str]:
    """List relative doc paths available as bermuda-doc resources."""
    return [
        str(p.relative_to(DOCS_DIR)) for p in _walk_files(DOCS_DIR, [".rst", ".txt"])
    ]


# @mcp.tool()
# def list_examples() -> list[str]:
#     """List example names available as bermuda-example resources (without .py)."""
#     return [p.stem for p in _walk_files(EXAMPLES_DIR, [".py"])]
#
#
# @mcp.tool()
# def get_example(name: str) -> str:
#     """Read a bermuda example by name (without .py)."""
#     return read_example(name)
#


@mcp.tool()
def search_docs(query: str, k: int = 12, context: int = 120) -> list[SearchHit]:
    """
    Very simple full-text search across docs. Returns best 'k' line hits.
    """
    q = re.compile(re.escape(query), re.IGNORECASE)
    hits: list[SearchHit] = []
    for path in _walk_files(DOCS_DIR, [".rst", ".txt"]):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines, start=1):
            if q.search(line):
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                snippet = "\n".join(lines[start:end])
                hits.append(
                    {
                        "path": str(path.relative_to(DOCS_DIR)),
                        "line": i,
                        "snippet": snippet[:context],
                    }
                )
    return hits[:k]


@mcp.tool()
def explain_symbol(qualified_name: str) -> str:
    """
    Import a symbol from the installed 'bermuda' package and return its docstring.
    Example: 'bermuda.utils.aggregate'
    """
    parts = qualified_name.split(".")
    for j in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:j])
        try:
            mod = importlib.import_module(mod_name)
            obj = mod
            for p in parts[j:]:
                obj = getattr(obj, p)
            doc = inspect.getdoc(obj) or "(no docstring found)"
            return f"{qualified_name}\n\n{doc}"
        except Exception:
            continue
    return f"Could not import {qualified_name}"


# ---------- Prompts (optional; become slash-commands in Claude Code) ----------
# from mcp.server.fastmcp.prompts import base
#
#
# @mcp.prompt(title="Bermuda: write code from task")
# def bermuda_snippet(task: str) -> list[base.Message]:
#     return [
#         base.SystemMessage(
#             "You are a bermuda coding assistant. Prefer APIs from 'bermuda'. "
#             "Use attached docs/examples when present. Include minimal runnable code."
#         ),
#         base.UserMessage(f"Task: {task}"),
#     ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
