"""
This module is a replacement for Python`s pprint module for pretty printing
clingo objects.
"""

# pylint: disable=protected-access

import pprint as _pp
from typing import IO, Any, Dict, Optional, Sequence, Tuple

from clingo.ast import AST, ASTSequence, Location, Position
from clingo.symbol import Symbol, SymbolType

__all__ = [
    "PrettyPrinter",
    "isreadable",
    "isrecursive",
    "pformat",
    "pp",
    "pprint",
    "saferepr",
]


def pprint(
    obj: Any,
    stream: Optional[IO[str]] = None,
    indent: int = 1,
    width: int = 80,
    depth: Optional[int] = None,
    **kwargs
):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth, **kwargs
    )
    printer.pprint(obj)


def pformat(
    obj: Any, indent: int = 1, width: int = 80, depth: Optional[int] = None, **kwargs
) -> str:
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(
        stream=None, indent=indent, width=width, depth=depth, **kwargs
    ).pformat(obj)


def pp(obj: Any, *args, sort_dicts: bool = False, **kwargs):
    """Pretty-print a Python object."""
    # pylint: disable=invalid-name
    pprint(obj, *args, sort_dicts=sort_dicts, **kwargs)  # nocoverage


def saferepr(obj: Any) -> str:
    """Version of repr() which can handle recursive data structures."""
    if hasattr(PrettyPrinter, "_safe_repr"):
        return PrettyPrinter()._safe_repr(obj, {}, None, 0)[0]  # type: ignore
    return _pp.saferepr(obj)  # nocoverage


def isreadable(obj: Any) -> bool:
    """Determine if saferepr(object) is readable by eval()."""
    if hasattr(PrettyPrinter, "_safe_repr"):
        return PrettyPrinter()._safe_repr(obj, {}, None, 0)[1]  # type: ignore
    return _pp.isreadable(obj)  # nocoverage


def isrecursive(obj: Any) -> bool:
    """Determine if object requires a recursive representation."""
    if hasattr(PrettyPrinter, "_safe_repr"):
        return PrettyPrinter()._safe_repr(obj, {}, None, 0)[2]  # type: ignore
    return _pp.isrecursive(obj)  # nocoverage


class _DummyLoc:
    def __repr__(self):
        return "LOC"


class PrettyPrinter(_pp.PrettyPrinter):
    """
    A pretty printer extending the standard `PrettyPrinter` class with
    functions to format `clingo.ast.AST` objects.
    """

    _hide_location: bool

    def __init__(self, *args, **kwargs):
        hide_location = kwargs.pop("hide_location", False)
        super().__init__(*args, **kwargs)
        self._hide_location = hide_location

    if _pp.PrettyPrinter.__init__.__doc__ is not None:
        __init__.__doc__ = (
            _pp.PrettyPrinter.__init__.__doc__
            + """hide_location
            Replace locations in `clingo.ast.AST` objects by placeholder LOC.
        """
        )

    def _pprint_namedtuple(
        self,
        obj: Any,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        # Note: adjusted _pprint_dataclass from python 3.10
        cls_name = obj.__class__.__name__
        indent += len(cls_name) + 1
        stream.write(cls_name + "(")
        self._format_kwargs_items(
            obj._asdict().items(), stream, indent, allowance, context, level
        )
        stream.write(")")

    def _format_kwargs_items(
        self,
        items: Sequence[Tuple[str, Any]],
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        # Note: copied _pprint_namespace_items from python 3.10
        write = stream.write
        delimnl = ",\n" + " " * indent
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key)
            write("=")
            if id(ent) in context:
                write("...")  # nocoverage
            else:
                self._format(  # type: ignore
                    ent,
                    stream,
                    indent + len(key) + 1,
                    allowance if last else 1,
                    context,
                    level,
                )
            if not last:
                write(delimnl)

    def _format_args_items(
        self,
        items: Sequence[Any],
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        write = stream.write
        delimnl = ",\n" + " " * indent
        last_index = len(items) - 1
        for i, ent in enumerate(items):
            last = i == last_index
            if id(ent) in context:
                write("...")  # nocoverage
            else:
                self._format(  # type: ignore
                    ent,
                    stream,
                    indent,
                    allowance if last else 1,
                    context,
                    level,
                )
            if not last:
                write(delimnl)

    _dispatch = _pp.PrettyPrinter._dispatch.copy()  # type: ignore

    def _pprint_pos(
        self,
        obj: Position,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        self._pprint_namedtuple(obj, stream, indent, allowance, context, level)

    _dispatch[Position.__repr__] = _pprint_pos

    def _pprint_loc(
        self,
        obj: Location,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        self._pprint_namedtuple(obj, stream, indent, allowance, context, level)

    _dispatch[Location.__repr__] = _pprint_loc

    def _pprint_ast(
        self,
        obj: AST,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        name = str(obj.ast_type).replace("ASTType", "ast")
        indent += len(name) + 1
        items = [
            (key, _DummyLoc() if self._hide_location and key == "location" else val)
            for key, val in obj.items()
        ]
        stream.write(name + "(")
        self._format_kwargs_items(items, stream, indent, allowance, context, level)
        stream.write(")")

    _dispatch[AST.__repr__] = _pprint_ast
    _dispatch[ASTSequence.__repr__] = _pp.PrettyPrinter._pprint_list  # type: ignore

    def _pprint_sym(
        self,
        obj: Symbol,
        stream: IO[str],
        indent: int,
        allowance: int,
        context: Dict[int, Any],
        level: int,
    ):
        if obj.type == SymbolType.Function:
            indent += 9
            items = [obj.name, obj.arguments, obj.positive]

            stream.write("Function(")
            self._format_args_items(items, stream, indent, allowance, context, level)
            stream.write(")")
        else:
            stream.write(repr(obj))

    _dispatch[Symbol.__repr__] = _pprint_sym
