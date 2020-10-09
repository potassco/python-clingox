"""
This is going to be merged into the ast module.
"""
from typing import Any

from clingo.ast import AST, ASTType


def _encode(x: Any) -> Any:
    return f"todo encode: {x}"

def as_dict(x: AST) -> dict:
    """
    Convert the given ast node into a dictionary representation whose elements
    only involve the data structures: `dict`, `list`, `int`, and `str`.
    """
    ret = {"type": str(x.type)}
    for key, value in x.items():
        ret[key] = _encode(value)
    return ret


def _decode(x: Any) -> Any:
    return f"todo decode: {x}"

def from_dict(x: dict) -> AST:
    """
    Convert the dictionary representation of an AST node into an AST node.
    """
    return AST(getattr(ASTType, x['type']), **{key: value for key, value in x.items() if key != "type"})
