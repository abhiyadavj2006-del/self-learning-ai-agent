"""Tools the agent can call. Stdlib only.

Tool 1 -- calculator: safe evaluation of arithmetic via `ast`.
  No eval(), no exec(), no names, no attribute access: only numbers
  and + - * / // % ** ( ) and a small whitelist of math functions.
Tool 2 -- note search: full-text search over the local notes/
  directory (.txt / .md), returning file, line number and snippet.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from pathlib import Path

# ---------------------------------------------------------------- calculator
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculate(expr: str) -> float:
    """Evaluate an arithmetic expression safely. Raises ValueError."""
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")
    if not re.fullmatch(r"[0-9\s+\-*/%().,a-zA-Z]+", expr):
        raise ValueError("invalid characters in expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse expression: {e}") from e
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


# ---------------------------------------------------------------- note search
def search_notes(query: str, notes_dir: str | Path = "notes") -> list[dict]:
    """Case-insensitive search of *.txt / *.md under notes_dir.

    Returns [{file, line, text, hits}] sorted by hits desc, then file.
    An empty query returns [].
    """
    query = query.strip().lower()
    if not query:
        return []
    words = query.split()
    base = Path(notes_dir)
    results: list[dict] = []
    if not base.is_dir():
        return []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            hits = sum(low.count(w) for w in words)
            if hits:
                results.append(
                    {"file": str(path), "line": i, "text": line.strip(), "hits": hits}
                )
    results.sort(key=lambda r: (-r["hits"], r["file"], r["line"]))
    return results


def list_notes(notes_dir: str | Path = "notes") -> list[str]:
    base = Path(notes_dir)
    if not base.is_dir():
        return []
    return sorted(
        str(p) for p in base.rglob("*")
        if p.suffix.lower() in {".txt", ".md"} and p.is_file()
    )
