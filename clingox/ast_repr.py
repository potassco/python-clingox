"""
This is either going to be merged into the ast module or put into a submodule.
"""
from typing import List, Any
from functools import singledispatch

from clingo import Symbol
from clingo.ast import AST, ASTType, Sign

from .ast import location_to_str, str_to_location

@singledispatch
def _encode(x: Any) -> Any:
    raise RuntimeError(f"todo encode: {x}")

@_encode.register
def _encode_str(x: str) -> str:
    return x

@_encode.register
def _encode_symbol(x: Symbol) -> str:
    return str(x)

@_encode.register
def _encode_bool(x: bool) -> bool:
    return x

@_encode.register
def _encode_sign(x: Sign) -> str:
    if x == Sign.NoSign:
        return 'NoSign'
    if x == Sign.Negation:
        return 'Negation'
    assert x == Sign.DoubleNegation
    return 'DoubleNegation'

@_encode.register
def _encode_list(x: list) -> List[Any]:
    return [_encode(y) for y in x]

@_encode.register
def _encode_none(x: None) -> None:
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
    for key, val in x.items():
        if key == 'location':
            enc = location_to_str(val)
        else:
            enc = _encode(val)
        ret[key] = enc
    return ret


def _decode(x: Any) -> Any:
    return f"todo decode: {x}"

def from_dict(x: dict) -> AST:
    """
    Convert the dictionary representation of an AST node into an AST node.
    """
    return AST(getattr(ASTType, x['type']), **{key: value for key, value in x.items() if key != "type"})
