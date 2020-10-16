"""
This is either going to be merged into the ast module or put into a submodule.
"""
from typing import Any
from functools import singledispatch

from clingo.ast import AST, ASTType

from .ast import location_to_str, str_to_location

@singledispatch
def _encode(x: Any) -> Any:
    return f"todo encode: {x}"

@_encode.register
def _encode_str(x: str) -> Any:
    return x

@_encode.register
def _encode_loc(x: dict) -> Any:
    return location_to_str(x)

@_encode.register
def _encode_list(x: list) -> Any:
    return [_encode(y) for y in x]

@_encode.register
def _encode_none(x: None) -> Any:
    return x

@_encode.register
def _encode_ast(x: AST) -> Any:
    return as_dict(x)

def as_dict(x: AST) -> dict:
    """
    Convert the given ast node into a dictionary representation whose elements
    only involve the data structures: `dict`, `list`, `int`, and `str`.
    """
    ret = {"type": str(x.type)}
    for key, value in x.items():
        # TODO: there should probably be a method inbetween that also recieves
        #       the key...
        ret[key] = _encode(value)
    return ret


def _decode(x: Any) -> Any:
    return f"todo decode: {x}"

def from_dict(x: dict) -> AST:
    """
    Convert the dictionary representation of an AST node into an AST node.
    """
    return AST(getattr(ASTType, x['type']), **{key: value for key, value in x.items() if key != "type"})
