"""Small shared tools. Dataset-specific tools live in the example that uses them."""

from __future__ import annotations

import ast
import operator as op

from langchain_core.tools import tool

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic only (numbers, + - * / ** % //, parentheses). Never uses eval()."""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = _ev(node.left), _ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](_ev(node.operand))
        raise ValueError("only arithmetic expressions are allowed")

    return _ev(ast.parse(expression.strip(), mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(129 * 3) * 0.9'. Supports + - * / ** % // and parentheses."""
    try:
        return str(round(safe_eval(expression), 6))
    except Exception as e:  # noqa: BLE001 - return the error to the model so it can self-correct
        return f"error: {e}"
