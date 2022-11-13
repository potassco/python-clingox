'''
This module provides highlevel functions to work with clingo's AST.

Theory Parsing Examples
-----------------------

The following examples shows how to construct and use a theory parser:

```python-repl
>>> from clingo.ast import TheoryAtomType, parse_string
>>> from clingox.ast import Arity, Associativity, TheoryParser
>>>
>>> terms = {"term":
...     {("-", Arity.Unary): (3, Associativity.NoAssociativity),
...      ("**", Arity.Binary): (2, Associativity.Right),
...      ("*", Arity.Binary): (1, Associativity.Left),
...      ("+", Arity.Binary): (0, Associativity.Left),
...      ("-", Arity.Binary): (0, Associativity.Left)}}
>>> atoms = {("eval", 0): (TheoryAtomType.Head, "term", None)}
>>> parser = TheoryParser(terms, atoms)
>>>
>>> parse_string('&eval{ -1 * 2 + 3 }.', print)
#program base.
&eval { (- 1 * 2 + 3) }.
>>> parse_string('&eval{ -1 * 2 + 3 }.', lambda x: print(parser(x)))
#program base.
&eval { +(*(-(1),2),3) }.
```

The same parser can also be constructed from a theory:

```python-repl
>>> from clingo.ast import parse_string, ASTType
>>> from clingox.ast import theory_parser_from_definition
>>>
>>> theory = """\\
... #theory test {
...     term {
...         -  : 3, unary;
...         ** : 2, binary, right;
...         *  : 1, binary, left;
...         +  : 0, binary, left;
...         -  : 0, binary, left
...     };
...     &eval/0 : term, head
... }.
... """
>>>
>>> parsers = []
>>> def extract(stm):
...     if stm.ast_type == ASTType.TheoryDefinition:
...         parsers.append(theory_parser_from_definition(stm))
...
>>> parse_string(theory, extract)
>>> parse_string('&eval{ -1 * 2 + 3 }.', print)
#program base.
&eval { (- 1 * 2 + 3) }.
>>> parse_string('&eval{ -1 * 2 + 3 }.', lambda x: print(parsers[0](x)))
#program base.
&eval { +(*(-(1),2),3) }.
```

AST to dict Conversion Example
------------------------------

Another interesting feature is to convert ASTs to YAML:

```python-repl
>>> from json import dumps
>>> from clingo.ast import parse_string
>>> from clingox.ast import ast_to_dict
>>>
>>> prg = []
>>> parse_string('a.', lambda x: prg.append(ast_to_dict(x)))
>>>
>>> print(dumps(prg, indent=2))
[
  {
    "ast_type": "Program",
    "location": "<string>:1:1",
    "name": "base",
    "parameters": []
  },
  {
    "ast_type": "Rule",
    "location": "<string>:1:1-3",
    "head": {
      "ast_type": "Literal",
      "location": "<string>:1:1-2",
      "sign": 0,
      "atom": {
        "ast_type": "SymbolicAtom",
        "symbol": {
          "ast_type": "Function",
          "location": "<string>:1:1-2",
          "name": "a",
          "arguments": [],
          "external": 0
        }
      }
    },
    "body": []
  }
]
```
'''
from copy import copy
from enum import Enum, auto
from functools import lru_cache, partial, singledispatch
from re import fullmatch
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import clingo
from clingo import ast
from clingo.ast import (
    AST,
    ASTSequence,
    ASTType,
    Function,
    Location,
    Position,
    Sign,
    StrSequence,
    SymbolicAtom,
    SymbolicTerm,
    TheoryAtomType,
    TheoryFunction,
    TheoryOperatorType,
    Transformer,
    UnaryOperation,
    parse_string,
)

from .theory import is_operator

__all__ = [
    "Arity",
    "Associativity",
    "ASTPredicate",
    "AtomTable",
    "OperatorTable",
    "TheoryParser",
    "TheoryTermParser",
    "TheoryUnparsedTermParser",
    "ast_to_dict",
    "clingo_literal_parser",
    "clingo_term_parser",
    "dict_to_ast",
    "filter_body_literals",
    "location_to_str",
    "negate_sign",
    "normalize_symbolic_terms",
    "parse_theory",
    "partition_body_literals",
    "prefix_symbolic_atoms",
    "reify_symbolic_atoms",
    "rename_symbolic_atoms",
    "str_to_location",
    "theory_parser_from_definition",
    "theory_term_to_literal",
    "theory_term_to_term",
]


class Arity(Enum):
    """
    Enumeration of operator arities.
    """

    # pylint:disable=invalid-name
    Unary = 1
    Binary = 2


class Associativity(Enum):
    """
    Enumeration of operator associativities.
    """

    # pylint: disable=invalid-name
    Left = auto()
    Right = auto()
    NoAssociativity = auto()


def _s(m, a: str, b: str):
    """
    Select the match group b if not None and group a otherwise.
    """
    return m[a] if m[b] is None else m[b]


def _quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:")


def _unquote(s: str) -> str:
    return s.replace("\\:", ":").replace("\\\\", "\\")


def location_to_str(loc: Location) -> str:
    """
    This function transfroms a loctation object into a readable string.

    Colons in the location will be quoted ensuring that the resulting is
    parsable using `str_to_location`.

    Parameters
    ----------
    loc
        The location to transform.

    Returns
    -------
    The string representation of the given location.
    """
    begin, end = loc.begin, loc.end
    bf, ef = _quote(begin.filename), _quote(end.filename)
    ret = f"{bf}:{begin.line}:{begin.column}"
    dash, eq = True, bf == ef
    if not eq:
        ret += f"{'-' if dash else ':'}{ef}"
        dash = False
    eq = eq and begin.line == end.line
    if not eq:
        ret += f"{'-' if dash else ':'}{end.line}"
        dash = False
    eq = eq and begin.column == end.column
    if not eq:
        ret += f"{'-' if dash else ':'}{end.column}"
        dash = False
    return ret


def str_to_location(loc: str) -> Location:
    """
    This function parses a location from its string representation.

    Parameters
    ----------
    loc
        The string to parse.

    Returns
    -------
    The parsed location.

    See Also
    --------
    location_to_str
    """
    m = fullmatch(
        r"(?P<bf>([^\\:]|\\\\|\\:)*):(?P<bl>[0-9]*):(?P<bc>[0-9]+)"
        r"(-(((?P<ef>([^\\:]|\\\\|\\:)*):)?(?P<el>[0-9]*):)?(?P<ec>[0-9]+))?",
        loc,
    )
    if not m:
        raise RuntimeError("could not parse location")
    begin = Position(_unquote(m["bf"]), int(m["bl"]), int(m["bc"]))
    end = Position(
        _unquote(_s(m, "bf", "ef")), int(_s(m, "bl", "el")), int(_s(m, "bc", "ec"))
    )
    return Location(begin, end)


OperatorTable = Mapping[Tuple[str, Arity], Tuple[int, Associativity]]
AtomTable = Mapping[
    Tuple[str, int], Tuple[TheoryAtomType, str, Optional[Tuple[List[str], str]]]
]


class TheoryUnparsedTermParser:
    """
    Parser for unparsed theory terms in clingo's AST that works like the
    inbuilt one.

    Note that associativity for unary operators is ignored and binary
    operators must use either `Associativity.Left` or `Associativity.Right`.

    Parameters
    ----------
    table
        Mapping of operator/arity pairs to priority/associativity pairs.
    """

    _stack: List[Tuple[str, Arity]]
    _terms: List[AST]
    _table: OperatorTable

    def __init__(self, table: OperatorTable):
        self._stack = []
        self._terms = []
        self._table = table

    def _priority_and_associativity(self, operator: str) -> Tuple[int, Associativity]:
        """
        Get priority and associativity of the given binary operator.
        """
        return self._table[(operator, Arity.Binary)]

    def _priority(self, operator: str, arity: Arity) -> int:
        """
        Get priority of the given unary or binary operator.
        """
        return self._table[(operator, arity)][0]

    def _check(self, operator: str) -> bool:
        """
        Returns true if the stack has to be reduced because of the precedence
        of the given binary operator is lower than the preceeding operator on
        the stack.
        """
        if not self._stack:
            return False
        priority, associativity = self._priority_and_associativity(operator)
        previous_priority = self._priority(*self._stack[-1])
        return previous_priority > priority or (
            previous_priority == priority and associativity == Associativity.Left
        )

    def _reduce(self) -> None:
        """
        Combines the last unary or binary term on the stack.
        """
        b = self._terms.pop()
        operator, arity = self._stack.pop()
        if arity == Arity.Unary:
            self._terms.append(TheoryFunction(b.location, operator, [b]))
        else:
            a = self._terms.pop()
            loc = Location(a.location.begin, b.location.end)
            self._terms.append(TheoryFunction(loc, operator, [a, b]))

    def check_operator(self, operator: str, arity: Arity, location: Location) -> None:
        """
        Check if the given operator is in the parse table raising a runtime
        error if absent.

        Parameters
        ----------
        operator
            The operator name.
        arity
            The arity of the operator.
        location
            Location of the operator for error reporting.
        """
        if (operator, arity) not in self._table:
            raise RuntimeError(
                f"cannot parse operator `{operator}`: {location_to_str(location)}"
            )

    def parse(self, x: AST) -> AST:
        """
        Parses the given unparsed term, replacing it by nested theory
        functions.

        Parameters
        ----------
        x
            The AST to parse.

        Returns
        -------
        The rewritten AST.
        """
        del self._stack[:]
        del self._terms[:]

        arity = Arity.Unary

        for element in x.elements:
            for operator in element.operators:
                self.check_operator(operator, arity, x.location)

                while arity == Arity.Binary and self._check(operator):
                    self._reduce()

                self._stack.append((operator, arity))
                arity = Arity.Unary

            self._terms.append(element.term)
            arity = Arity.Binary

        while self._stack:
            self._reduce()

        return self._terms[0]


class TheoryTermParser(Transformer):
    """
    Parser for theory terms in clingo's AST that works like the inbuilt one.

    This is implemented as a transformer that traverses the AST replacing all
    terms found.

    Parameters
    ----------
    table
        This must either be a table of operators or a `TheoryUnparsedTermParser`.

    See Also
    --------
    TheoryUnparsedTermParser
    """

    # pylint: disable=invalid-name

    def __init__(self, table: Union[OperatorTable, TheoryUnparsedTermParser]):
        self._parser = (
            table
            if isinstance(table, TheoryUnparsedTermParser)
            else TheoryUnparsedTermParser(table)
        )

    def visit_TheoryFunction(self, x) -> AST:
        """
        Parse the theory function and check if it agrees with the grammar.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        arity = None
        if len(x.arguments) == 1:
            arity = Arity.Unary
        if len(x.arguments) == 2:
            arity = Arity.Binary
        if arity is not None and is_operator(x.name):
            self._parser.check_operator(x.name, arity, x.location)

        return x.update(**self.visit_children(x))

    def visit_TheoryUnparsedTerm(self, x: AST) -> AST:
        """
        Parse the given unparsed term.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return cast(AST, self(self._parser.parse(x)))


_clingo_term_table = {
    ("-", Arity.Unary): (5, Associativity.NoAssociativity),
    ("~", Arity.Unary): (5, Associativity.NoAssociativity),
    ("**", Arity.Binary): (4, Associativity.Right),
    ("*", Arity.Binary): (3, Associativity.Left),
    ("/", Arity.Binary): (3, Associativity.Left),
    ("\\", Arity.Binary): (3, Associativity.Left),
    ("+", Arity.Binary): (2, Associativity.Left),
    ("-", Arity.Binary): (2, Associativity.Left),
    ("&", Arity.Binary): (1, Associativity.Left),
    ("?", Arity.Binary): (1, Associativity.Left),
    ("^", Arity.Binary): (1, Associativity.Left),
    ("..", Arity.Binary): (0, Associativity.Left),
}


@lru_cache(maxsize=None)
def clingo_term_parser() -> TheoryTermParser:
    """
    Return a theory term parser that parses theory terms like clingo terms.

    Note that for technical reasons pools and the absolute function are not
    supported.
    """
    return TheoryTermParser(_clingo_term_table)


@lru_cache(maxsize=None)
def clingo_literal_parser() -> TheoryTermParser:
    """
    Return a theory term parser that parses theory literals similar to clingo's
    parser for symbolic literals.

    Note that for technical reasons pools and the absolute function are not
    supported.
    """
    clingo_literal_table = _clingo_term_table.copy()
    clingo_literal_table.update(
        {
            ("-", Arity.Unary): (0, Associativity.NoAssociativity),
            ("not", Arity.Unary): (0, Associativity.NoAssociativity),
        }
    )
    return TheoryTermParser(clingo_literal_table)


class TheoryParser(Transformer):
    """
    This class parses theory atoms in the same way as clingo's internal parser.

    Parameters
    ----------
    terms
        Mapping from term identifiers to `TheoryTermParser`s. If an operator
        table is given, the `TheoryTermParser` is constructed from this table.

    atoms
        Mapping from atom name/arity pairs to tuples defining the acceptable
        structure of the theory atom.
    """

    # pylint: disable=invalid-name
    _table: Mapping[
        Tuple[str, int],
        Tuple[
            TheoryAtomType,
            TheoryTermParser,
            Optional[Tuple[Set[str], TheoryTermParser]],
        ],
    ]
    _in_body: bool
    _in_head: bool
    _is_directive: bool

    def __init__(
        self,
        terms: Mapping[str, Union[OperatorTable, TheoryTermParser]],
        atoms: AtomTable,
    ):
        self._reset()

        term_parsers = {}
        for term_key, parser in terms.items():
            if isinstance(parser, TheoryTermParser):
                term_parsers[term_key] = parser
            else:
                term_parsers[term_key] = TheoryTermParser(parser)

        self._table = {}
        for atom_key, (atom_type, term_key, guard) in atoms.items():
            guard_table = None
            if guard is not None:
                guard_table = (set(guard[0]), term_parsers[guard[1]])
            self._table[atom_key] = (atom_type, term_parsers[term_key], guard_table)

    def _reset(self, in_head=True, in_body=True, is_directive=True):
        """
        Set state information about active scope.
        """
        self._in_head = in_head
        self._in_body = in_body
        self._is_directive = is_directive

    def _visit_body(self, x: AST) -> AST:
        try:
            self._reset(False, True, False)
            old = x.body
            new = self.visit_sequence(old)
            return x if new is old else x.update(body=new)
        finally:
            self._reset()

    def visit_Rule(self, x: AST) -> AST:
        """
        Parse theory atoms in body and head.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        ret = self._visit_body(x)
        try:
            self._reset(True, False, not x.body)
            head = self(x.head)
            if head is not x.head:
                if ret is x:
                    ret = copy(ret)
                ret.head = head
        finally:
            self._reset()

        return ret

    def visit_ShowTerm(self, x: AST) -> AST:
        """
        Parse theory atoms in body.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return self._visit_body(x)

    def visit_Minimize(self, x: AST) -> AST:
        """
        Parse theory atoms in body.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return self._visit_body(x)

    def visit_Edge(self, x: AST) -> AST:
        """
        Parse theory atoms in body.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return self._visit_body(x)

    def visit_Heuristic(self, x: AST) -> AST:
        """
        Parse theory atoms in body.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return self._visit_body(x)

    def visit_ProjectAtom(self, x: AST) -> AST:
        """
        Parse theory atoms in body.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        return self._visit_body(x)

    def visit_TheoryAtom(self, x: AST) -> AST:
        """
        Parse the given theory atom.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        name = x.term.name
        arity = len(x.term.arguments)
        if (name, arity) not in self._table:
            raise RuntimeError(
                f"theory atom definiton not found: {location_to_str(x.location)}"
            )

        type_, element_parser, guard_table = self._table[(name, arity)]
        if type_ == TheoryAtomType.Head and not self._in_head:
            raise RuntimeError(
                f"theory atom only accepted in head: {location_to_str(x.location)}"
            )
        if type_ == TheoryAtomType.Body and not self._in_body:
            raise RuntimeError(
                f"theory atom only accepted in body: {location_to_str(x.location)}"
            )
        if type_ == TheoryAtomType.Directive and not (
            self._in_head and self._is_directive
        ):
            raise RuntimeError(
                f"theory atom must be a directive: {location_to_str(x.location)}"
            )

        x = copy(x)
        x.term = element_parser(x.term)
        x.elements = element_parser.visit_sequence(x.elements)

        if x.guard is not None:
            if guard_table is None:
                raise RuntimeError(
                    f"unexpected guard in theory atom: {location_to_str(x.location)}"
                )

            guards, guard_parser = guard_table
            if x.guard.operator_name not in guards:
                raise RuntimeError(
                    f"unexpected guard in theory atom: {location_to_str(x.location)}"
                )

            x.guard = copy(x.guard)
            x.guard.term = guard_parser(x.guard.term)

        return x


def theory_parser_from_definition(x: AST) -> TheoryParser:
    """
    Turn an AST node of type TheoryDefinition into a TheoryParser.

    Parameters
    ----------
    x
        An AST representing a theory definition.

    Returns
    -------
    The corresponding `TheoryParser`.
    """
    assert x.ast_type == ASTType.TheoryDefinition

    terms = {}
    atoms = {}

    for term_def in x.terms:
        term_table = {}

        for op_def in term_def.operators:
            op_assoc: Associativity
            if op_def.operator_type == TheoryOperatorType.BinaryLeft:
                op_type = Arity.Binary
                op_assoc = Associativity.Left
            elif op_def.operator_type == TheoryOperatorType.BinaryRight:
                op_type = Arity.Binary
                op_assoc = Associativity.Right
            else:
                op_type = Arity.Unary
                op_assoc = Associativity.NoAssociativity

            term_table[(op_def.name, op_type)] = (op_def.priority, op_assoc)

        terms[term_def.name] = term_table

    for atom_def in x.atoms:
        guard = None
        if atom_def.guard is not None:
            guard = (atom_def.guard.operators, atom_def.guard.term)

        atoms[(atom_def.name, atom_def.arity)] = (
            atom_def.atom_type,
            atom_def.term,
            guard,
        )

    return TheoryParser(terms, atoms)


def parse_theory(s: str) -> TheoryParser:
    """
    Turn the given theory into a parser.
    """
    parser = None

    def extract(stm):
        nonlocal parser
        if stm.ast_type == ASTType.TheoryDefinition:
            if parser is not None:
                raise ValueError("multiple theory definitions")
            parser = theory_parser_from_definition(stm)
        else:
            assert (
                stm.ast_type == ASTType.Program
                and stm.name == "base"
                and not stm.parameters
            )

    parse_string(f"{s}.", extract)
    if parser is None:
        raise ValueError("no theory definition found")
    return cast(TheoryParser, parser)


class _SymbolicAtomTransformer(Transformer):
    """
    Transforms symbolic atoms with the given function.
    """

    # pylint: disable=invalid-name

    def __init__(self, transformer_function: Callable[[AST], AST]):
        self._transformer_function = transformer_function

    def visit_SymbolicAtom(self, x: AST) -> AST:
        """
        Transform the given symbolic.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """
        term = x.symbol
        new_term = self._transformer_function(term)
        return x if new_term is term else SymbolicAtom(new_term)


def rewrite_symbolic_atoms(x: AST, rewrite_function: Callable[[AST], AST]) -> AST:
    """
    Rewrite all symbolic atoms in the given AST node with the given function.

    Parameters
    ----------
    x
        The ast in which to rename symbolic atoms.
    rename_function
        A function applied to the term representation of the symbolic atom. The
        function has to return a term compatible with symbolic atoms.

    Returns
    -------
    The rewritten AST.
    """
    return cast(AST, _SymbolicAtomTransformer(rewrite_function)(x))


def rename_symbolic_atoms(x: AST, rename_function: Callable[[str], str]) -> AST:
    """
    Rename all symbolic atoms in the given AST node with the given function.

    Parameters
    ----------
    x
        The ast in which to rename symbolic atoms.
    rename_function
        A function for renaming symbols.

    Returns
    -------
    The rewritten AST.
    """

    def renamer(term: AST):
        if term.ast_type == ASTType.UnaryOperation:
            return UnaryOperation(
                term.location, term.operator_type, renamer(term.argument)
            )
        if term.ast_type == ASTType.SymbolicTerm:
            sym = term.symbol
            new_name = rename_function(sym.name)
            return SymbolicTerm(
                term.location, clingo.Function(new_name, sym.arguments, sym.positive)
            )
        if term.ast_type == ASTType.Function:
            return Function(
                term.location, rename_function(term.name), term.arguments, term.external
            )
        return term

    return rewrite_symbolic_atoms(x, renamer)


def prefix_symbolic_atoms(x: AST, prefix: str) -> AST:
    """
    Prefix all symbolic atoms in the given AST with the given string.

    Parameters
    ----------
    x
        The ast in which to prefix symbolic atom names.
    prefix
        The prefix to add.

    Returns
    -------
    The rewritten AST.

    See Also
    --------
    rename_symbolic_atoms
    """
    return rename_symbolic_atoms(x, lambda s: prefix + s)


def reify_symbolic_atoms(
    x: AST,
    name: str,
    argument_extender: Optional[Callable[[AST], Sequence[AST]]] = None,
    reify_strong_negation: bool = False,
) -> AST:
    """
    Reify all symbolic atoms in the given AST node with the given name and
    function.

    Parameters
    ----------
    x
        The ast in which to rename symbolic atoms.
    name
        A string to serve as name of the new symbolic atom.
    argument_extender
        A function to provide extra arguments. If not provided, no extra
        arguments are added. The term passed as argument should be placed in
        the correct position.
    reify_strong_negation
        Boolean indicating how to encode strong negation. If false, `-p(X)` is
        reified as `-name(p(X))`. If true, then `-p(X)` is reified as
        `name(-p(X))`. In the latter case, this means that stable models
        containing both `name(p(a))` and `name(-p(a))` are possible. Clingo
        style consistency can be restored by adding the constraint
        `:- name(X), name(-X), X<-X.`

    Returns
    -------
    The rewritten AST.
    """

    def reifier(term: AST):
        if term.ast_type == ASTType.UnaryOperation and not reify_strong_negation:
            return UnaryOperation(
                term.location, term.operator_type, reifier(term.argument)
            )
        arguments = argument_extender(term) if argument_extender else [term]
        return Function(term.location, name, arguments, False)

    return rewrite_symbolic_atoms(x, reifier)


@singledispatch
def _encode(x: Any) -> Any:
    assert False, f"unknown value to encode: {x}"


@_encode.register(str)
def _encode_str(x: str) -> str:
    return x


@_encode.register(clingo.Symbol)
def _encode_symbol(x: clingo.Symbol) -> str:
    return str(x)


@_encode.register(int)
def _encode_int(x: int) -> int:
    return x


@_encode.register(ASTSequence)
def _encode_ast_seq(x: ASTSequence) -> List[Any]:
    return [_encode(y) for y in x]


@_encode.register(StrSequence)
def _encode_str_seq(x: StrSequence) -> List[Any]:
    return [_encode(y) for y in x]


@_encode.register(type(None))
def _encode_none(x: None) -> None:
    return x


@_encode.register(AST)
def _encode_ast(x: AST) -> Any:
    return ast_to_dict(x)


def ast_to_dict(x: AST) -> dict:
    """
    Convert the given ast node into a dictionary representation whose elements
    only involve the data structures: `dict`, `list`, `int`, and `str`.

    The resulting value can be used with other Python modules like the `yaml`
    or `pickle` modules.

    Parameters
    ----------
    x
        The ast to transform.

    Returns
    -------
    The corresponding Python representation.

    See Also
    --------
    dict_to_ast
    """
    ret = {"ast_type": str(x.ast_type).replace("ASTType.", "")}
    for key, val in x.items():
        if key == "location":
            assert isinstance(val, Location)
            enc = location_to_str(val)
        else:
            enc = _encode(val)
        ret[key] = enc
    return ret


@singledispatch
def _decode(x: Any, key: str) -> Any:
    raise RuntimeError(f"unknown key/value to decode: {key}: {x}")


@_decode.register(str)
def _decode_str(x: str, key: str) -> Any:
    if key == "location":
        return str_to_location(x)

    if key == "symbol":
        return clingo.parse_term(x)

    assert key in ("name", "id", "code", "elements", "term", "list", "operator_name")
    return x


@_decode.register(int)
def _decode_int(x: int, key: str) -> Any:
    # pylint: disable=unused-argument
    return x


@_decode.register(type(None))
def _decode_none(x: None, key: str) -> Any:
    # pylint: disable=unused-argument
    return x


@_decode.register(list)
def _decode_list(x: list, key_: str) -> Any:
    # pylint: disable=unused-argument
    return [_decode(y, "list") for y in x]


@_decode.register(dict)
def _decode_dict(x: dict, key_: str) -> Any:
    # pylint: disable=unused-argument
    return dict_to_ast(x)


def dict_to_ast(x: dict) -> AST:
    """
    Convert the Python dict representation of an AST node into an AST node.

    Parameters
    ----------
    x
        The Python representation of the AST.

    Returns
    -------
    The corresponding AST.

    See Also
    --------
    ast_to_dict
    """
    return getattr(ast, x["ast_type"])(
        **{key: _decode(value, key) for key, value in x.items() if key != "ast_type"}
    )


ASTPredicate = Union[Callable[[AST], bool], bool]


def _eval_predicate(predicate: ASTPredicate, arg: AST) -> bool:
    if callable(predicate):
        return predicate(arg)
    return predicate


def _body_literal_predicate(
    lit: AST,
    symbolic_atom_predicate: ASTPredicate = True,
    theory_atom_predicate: ASTPredicate = True,
    aggregate_predicate: ASTPredicate = True,
    conditional_literal_predicate: ASTPredicate = True,
    signs: Container[Sign] = (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
) -> bool:
    if lit.ast_type == ASTType.Literal:
        atom = lit.atom
        if lit.sign not in signs:
            return False
        if atom.ast_type == ASTType.SymbolicAtom:
            return _eval_predicate(symbolic_atom_predicate, atom.symbol)
        if atom.ast_type in (ASTType.Aggregate, ASTType.BodyAggregate):
            return _eval_predicate(aggregate_predicate, atom)
        if atom.ast_type == ASTType.TheoryAtom:
            return _eval_predicate(theory_atom_predicate, atom)
    elif lit.ast_type == ASTType.ConditionalLiteral:
        return lit.literal.sign in signs and _eval_predicate(
            conditional_literal_predicate, lit
        )
    return True


def filter_body_literals(
    body: Iterable[AST],
    symbolic_atom_predicate: ASTPredicate = True,
    theory_atom_predicate: ASTPredicate = True,
    aggregate_predicate: ASTPredicate = True,
    conditional_literal_predicate: ASTPredicate = True,
    signs: Container[Sign] = (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
) -> Iterable[AST]:
    """
    Filters the given body literals according to the given predicates.

    Parameters
    ----------
    body
        An iterable of `AST`s for body literals.
    symbolic_atom_predicate
        Predicate to filter symbolic atoms.
    theory_atom_predicate
        Predicate to filter theory atoms.
    aggregate_predicate
        Predicate to filter aggregates.
    conditional_literal_predicate
        Predicate to filter conditional literals.
    signs
        Only include literals with the given signs.

    Returns
    -------
    An iterarable of body literals.

    Notes
    -----
    An `ASTPredicate` is a callable that takes an `AST` and returns a Boolean.
    Booleans `True` and `False` are also accepted, meaning that the predicate
    is always `True` or `False`, respectively.
    """
    pred = partial(
        _body_literal_predicate,
        symbolic_atom_predicate=symbolic_atom_predicate,
        theory_atom_predicate=theory_atom_predicate,
        aggregate_predicate=aggregate_predicate,
        conditional_literal_predicate=conditional_literal_predicate,
        signs=signs,
    )
    return filter(pred, body)


def partition_body_literals(
    body: Iterable[AST],
    symbolic_atom_predicate: ASTPredicate = True,
    theory_atom_predicate: ASTPredicate = True,
    aggregate_predicate: ASTPredicate = True,
    conditional_literal_predicate: ASTPredicate = True,
    signs: Container[Sign] = (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
) -> Tuple[List[AST], List[AST]]:
    """
    Partition the given body literals according to the given predicates.

    Parameters
    ----------
    body
        An iterable of `AST` that represents a body.
    symbolic_atom_predicate
        Predicate to partition symbolic atoms.
    theory_atom_predicate
        Predicate to partition theory atoms.
    aggregate_predicate
        Predicate to partition aggregates.
    conditional_literal_predicate
        Predicate to partition conditional literals.
    signs
        Only include literals with the given signs in the first list.

    Returns
    -------
    A pair of lists of body literals. The first iterable yields the literals
    that satisfy the predicate while the second one yields the ones that do
    not.

    Notes
    -----
    An `ASTPredicate` is a callable that takes an `AST` and returns a Boolean.
    Booleans `True` and `False` are also accepted, meaning that the predicate
    is always `True` or `False`, respectively.
    """
    pred = partial(
        _body_literal_predicate,
        symbolic_atom_predicate=symbolic_atom_predicate,
        theory_atom_predicate=theory_atom_predicate,
        aggregate_predicate=aggregate_predicate,
        conditional_literal_predicate=conditional_literal_predicate,
        signs=signs,
    )
    part_a: List[AST] = []
    part_b: List[AST] = []
    for lit in body:
        if pred(lit):
            part_a.append(lit)
        else:
            part_b.append(lit)
    return part_a, part_b


_unary_operator_map = {
    "-": ast.UnaryOperator.Minus,
    "~": ast.UnaryOperator.Negation,
    "|": ast.UnaryOperator.Absolute,
}

_binary_operator_map = {
    "+": ast.BinaryOperator.Plus,
    "-": ast.BinaryOperator.Minus,
    "*": ast.BinaryOperator.Multiplication,
    "/": ast.BinaryOperator.Division,
    "\\": ast.BinaryOperator.Modulo,
    "**": ast.BinaryOperator.Power,
    "&": ast.BinaryOperator.And,
    "?": ast.BinaryOperator.Or,
    "^": ast.BinaryOperator.XOr,
}


def _theory_term_to_term(x: AST) -> AST:
    """
    Convert a given theory term into a plain clingo term.
    """
    if x.ast_type in (ASTType.SymbolicTerm, ASTType.Variable):
        return x

    if x.ast_type == ASTType.TheoryFunction:
        if len(x.arguments) == 1 and x.name in _unary_operator_map:
            arg = _theory_term_to_term(x.arguments[0])
            uop = _unary_operator_map[x.name]

            return ast.UnaryOperation(x.location, uop, arg)

        if len(x.arguments) == 2:
            lhs = _theory_term_to_term(x.arguments[0])
            rhs = _theory_term_to_term(x.arguments[1])

            if x.name in _binary_operator_map:
                bop = _binary_operator_map[x.name]
                return ast.BinaryOperation(x.location, bop, lhs, rhs)

            if x.name == "..":
                return ast.Interval(x.location, lhs, rhs)

        if not is_operator(x.name):
            return ast.Function(
                x.location,
                x.name,
                [_theory_term_to_term(a) for a in x.arguments],
                False,
            )

    elif x.ast_type == ASTType.TheorySequence:
        if x.sequence_type == ast.TheorySequenceType.Tuple:
            return ast.Function(
                x.location, "", [_theory_term_to_term(a) for a in x.terms], False
            )

    raise RuntimeError(f"{location_to_str(x.location)}: invalid term `{x}`")


def theory_term_to_term(x: AST, parse: bool = True) -> AST:
    """
    Convert the given theory term into a plain clingo term.

    If argument `parse` is set to true, occurences of unparsed theory terms are
    parsed using `clingo_term_parser()`.
    """
    if parse:
        x = clingo_term_parser()(x)
    return _theory_term_to_term(x)


def _build_atom(
    location: ast.Location, positive: bool, name: str, arguments: List
) -> ast.AST:
    """
    Helper function to create an atom.

    Arguments:
    location  -- Location to use.
    positive  -- Classical sign of the atom.
    name      -- The name of the atom.
    arguments -- The arguments of the atom.
    """
    ret = ast.Function(location, name, arguments, False)
    if not positive:
        ret = ast.UnaryOperation(location, ast.UnaryOperator.Minus, ret)
    return ast.SymbolicAtom(ret)


def negate_sign(sign: ast.Sign) -> ast.Sign:
    """
    Negate the given sign.
    """
    if sign == ast.Sign.Negation:
        return ast.Sign.DoubleNegation
    return ast.Sign.Negation


def _theory_term_to_literal(
    x: AST, positive: bool = True, sign: ast.Sign = ast.Sign.NoSign
) -> AST:
    """
    Convert a given theory term into a symbolic clingo literal.
    """
    if x.ast_type == ASTType.TheoryFunction:
        if x.name == "-":
            return _theory_term_to_literal(x.arguments[0], not positive, sign)

        if x.name == "not":
            sign = negate_sign(sign)
            if not positive:
                sign = negate_sign(sign)
            return _theory_term_to_literal(x.arguments[0], True, sign)

        if not is_operator(x.name):
            atom = _build_atom(
                x.location,
                positive,
                x.name,
                [theory_term_to_term(a) for a in x.arguments],
            )
            return ast.Literal(x.location, sign, atom)

    elif (
        x.ast_type == ASTType.SymbolicTerm
        and x.symbol.type == clingo.SymbolType.Function
        and x.symbol.name
    ):
        atom = _build_atom(
            x.location,
            (positive == x.symbol.positive),
            x.symbol.name,
            [ast.SymbolicTerm(x.location, a) for a in x.symbol.arguments],
        )
        return ast.Literal(x.location, sign, atom)

    raise RuntimeError(f"{location_to_str(x.location)}: invalid literal `{x}`")


def theory_term_to_literal(x: AST, parse: bool = True) -> AST:
    """
    Convert the given theory term into a symbolic clingo literal.

    If argument `parse` is set to true, occurences of unparsed theory terms are
    parsed using `clingo_literal_parser()`.

    Literals can use an arbitrary number of classical and default negation
    signs. They are normalized using the following equivalences:

    - `- - lit = lit`
    - `- not lit = not not lit`
    - `not not not lit = not lit`
    """
    if parse:
        x = clingo_literal_parser()(x)
    return _theory_term_to_literal(x, True, ast.Sign.NoSign)


def normalize_symbolic_terms(x: AST):
    """
    Replaces all occurrences of objects of the class clingo.Function in an AST
    by the corresponding object of the class ast.Function.

    Parameters
    ----------
    x
        The AST to rewrite.

    Returns
    -------
    The rewritten AST.
    """
    return _NormalizeSymbolicTermTransformer().visit(x)


def _symbol_to_ast(x: clingo.Symbol, location: ast.Location) -> AST:
    """
    Convert the given symbol into an AST.

    Parameters
    ----------
    x
        The symbol to convert.
    location
        The location to use.

    Returns
    -------
    The converted AST.
    """
    if x.type != clingo.SymbolType.Function:
        return SymbolicTerm(location, x)
    return ast.Function(
        location,
        x.name,
        [_symbol_to_ast(a, location) for a in x.arguments],
        external=False,
    )


class _NormalizeSymbolicTermTransformer(Transformer):
    """Transforms a SymbolicTerm AST of type Function into an AST of type ast.Function."""

    def visit_SymbolicTerm(self, x: AST):  # pylint: disable=invalid-name
        """
        Transform the given symbolic term.

        Parameters
        ----------
        x
            The AST to rewrite.

        Returns
        -------
        The rewritten AST.
        """

        symbol = x.symbol

        if symbol.type != clingo.SymbolType.Function:
            return x

        return _symbol_to_ast(symbol, x.location)
