'''
This module provides highlevel functions to work with clingo's AST.

TODO:
- unpooling as in clingcon
'''

from typing import Any, Callable, cast, Iterator, List, Mapping, Optional, Set, Tuple, TypeVar, Union
from functools import singledispatch
from copy import copy
from re import fullmatch

import clingo
from clingo.ast import (
    AggregateFunction, AST, ASTType, BinaryOperator, ComparisonOperator, Function, ScriptType, Sign,
    Symbol, SymbolicAtom, TheoryAtomType, TheoryFunction, TheoryOperatorType, Transformer, UnaryOperator)
from .theory import is_operator


UNARY: bool = True
BINARY: bool = not UNARY
LEFT: bool = True
RIGHT: bool = not LEFT
NONE: bool = RIGHT

def _s(m, a: str, b: str):
    '''
    Select the match group b if not None and group a otherwise.
    '''
    return m[a] if m[b] is None else m[b]

def _quote(s: str) -> str:
    return s.replace('\\', '\\\\').replace(':', '\\:')

def _unquote(s: str) -> str:
    return s.replace('\\:', ':').replace('\\\\', '\\')

def location_to_str(loc: dict) -> str:
    """
    This function takes a location from a clingo AST and transforms it into a
    readable format.

    Colons in the location will be quoted ensuring that the location is
    parsable.
    """
    begin, end = loc["begin"], loc["end"]
    bf, ef = _quote(begin['filename']), _quote(end['filename'])
    ret = "{}:{}:{}".format(bf, begin["line"], begin["column"])
    dash, eq = True, bf == ef
    if not eq:
        ret += "{}{}".format("-" if dash else ":", ef)
        dash = False
    eq = eq and begin["line"] == end["line"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["line"])
        dash = False
    eq = eq and begin["column"] == end["column"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["column"])
        dash = False
    return ret

def str_to_location(s: str) -> dict:
    """
    This function parses a location string and returns it as a dictionary as
    accepted by clingo's AST.
    """
    m = fullmatch(
        r'(?P<bf>([^\\:]|\\\\|\\:)*):(?P<bl>[0-9]*):(?P<bc>[0-9]+)'
        r'(-(((?P<ef>([^\\:]|\\\\|\\:)*):)?(?P<el>[0-9]*):)?(?P<ec>[0-9]+))?', s)
    if not m:
        raise RuntimeError('could not parse location')
    ret = {'begin': {'filename': _unquote(m['bf']),
                     'line': int(m['bl']),
                     'column': int(m['bc'])},
           'end': {'filename': _unquote(_s(m, 'bf', 'ef')),
                   'line': int(_s(m, 'bl', 'el')),
                   'column':  int(_s(m, 'bc', 'ec'))}}
    return ret


class TheoryUnparsedTermParser:
    """
    Parser for unparsed theory terms in clingo's AST that works like the
    inbuilt one.
    """
    _stack: List[Tuple[str, bool]]
    _terms: List[AST]

    def __init__(self, table: Mapping[Tuple[str, bool], Tuple[int, bool]]):
        """
        Initializes the parser with the given operators.

        Example table
        -------------
        { ("-",  UNARY):  (3, RIGHT), # associativity is ignored
          ("**", BINARY): (2, RIGHT),
          ("*",  BINARY): (1, LEFT),
          ("+",  BINARY): (0, LEFT),
          ("-",  BINARY): (0, LEFT) }
        """
        self._stack = []
        self._terms = []
        self._table = table

    def _priority_and_associativity(self, operator: str) -> Tuple[int, bool]:
        """
        Get priority and associativity of the given binary operator.
        """
        return self._table[(operator, BINARY)]

    def _priority(self, operator: str, unary: bool) -> int:
        """
        Get priority of the given unary or binary operator.
        """
        return self._table[(operator, unary)][0]

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
        return previous_priority > priority or (previous_priority == priority and associativity)

    def _reduce(self) -> None:
        """
        Combines the last unary or binary term on the stack.
        """
        b = self._terms.pop()
        operator, unary = self._stack.pop()
        if unary:
            self._terms.append(TheoryFunction(b.location, operator, [b]))
        else:
            a = self._terms.pop()
            l = {"begin": a.location["begin"], "end": b.location["end"]}
            self._terms.append(TheoryFunction(l, operator, [a, b]))

    def check_operator(self, operator: str, unary: bool, location: Any):
        """
        Check if the given operator is in the parse table.
        """
        if not (operator, unary) in self._table:
            raise RuntimeError("cannot parse operator `{}`: {}".format(operator, location_to_str(location)))

    def parse(self, x: AST) -> AST:
        """
        Parses the given unparsed term, replacing it by nested theory
        functions.
        """
        del self._stack[:]
        del self._terms[:]

        unary = True

        for element in x.elements:
            for operator in element.operators:
                self.check_operator(operator, unary, x.location)

                while not unary and self._check(operator):
                    self._reduce()

                self._stack.append((operator, unary))
                unary = True

            self._terms.append(element.term)
            unary = False

        while self._stack:
            self._reduce()

        return self._terms[0]

TermTable = Mapping[Tuple[str, bool],
                    Tuple[int, bool]]
AtomTable = Mapping[Tuple[str, int],
                    Tuple[TheoryAtomType,
                          str,
                          Optional[Tuple[List[str], str]]]]

class TheoryTermParser(Transformer):
    """
    Parser for theory terms in clingo's AST that works like the inbuilt one.

    This is implemented as a transformer that traverses the AST replacing all
    terms found.
    """
    # pylint: disable=invalid-name

    def __init__(self, table: TermTable):
        """
        Initializes the parser with the given operators.
        """
        self._parser = TheoryUnparsedTermParser(table)

    def visit_TheoryFunction(self, x) -> AST:
        """
        Parse the theory function and check if it agrees with the grammar.
        """
        unary = len(x.arguments) == 1
        binary = len(x.arguments) == 2
        if (unary or binary) and is_operator(x.name):
            self._parser.check_operator(x.name, unary, x.location)

        return self.visit_children(x)

    def visit_TheoryUnparsedTerm(self, x: AST) -> AST:
        """
        Parse the given unparsed term.
        """
        return cast(AST, self(self._parser.parse(x)))


class TheoryParser(Transformer):
    """
    This class parses theory atoms in the same way as clingo's internal parser.
    """
    # pylint: disable=invalid-name
    _table: Mapping[Tuple[str, int],
                    Tuple[TheoryAtomType,
                          TheoryTermParser,
                          Optional[Tuple[Set[str], TheoryTermParser]]]]
    in_body: bool
    in_head: bool
    is_directive: bool

    def __init__(self, terms: Mapping[str, Union[TermTable, TheoryTermParser]], atoms: AtomTable):
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
        self.in_head = in_head
        self.in_body = in_body
        self.is_directive = is_directive

    def _visit_body(self, x: AST) -> AST:
        try:
            self._reset(False, True, False)
            body = self.visit_list(x.body)
            if body is not x.body:
                x = copy(x)
                x.body = body
        finally:
            self._reset()
        return x

    def visit_Rule(self, x: AST) -> AST:
        """
        Parse theory atoms in body and head.
        """
        try:
            ret = self._visit_body(x)
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
        """
        return self._visit_body(x)

    def visit_Minimize(self, x: AST) -> AST:
        """
        Parse theory atoms in body.
        """
        return self._visit_body(x)

    def visit_Edge(self, x: AST) -> AST:
        """
        Parse theory atoms in body.
        """
        return self._visit_body(x)

    def visit_Heuristic(self, x: AST) -> AST:
        """
        Parse theory atoms in body.
        """
        return self._visit_body(x)

    def visit_ProjectAtom(self, x: AST) -> AST:
        """
        Parse theory atoms in body.
        """
        return self._visit_body(x)

    def visit_TheoryAtom(self, x: AST) -> AST:
        """
        Parse the given theory atom.
        """
        name = x.term.name
        arity = len(x.term.arguments)
        if (name, arity) not in self._table:
            raise RuntimeError(f"theory atom definiton not found: {location_to_str(x.location)}")

        type_, element_parser, guard_table = self._table[(name, arity)]
        if type_ is TheoryAtomType.Head and not self.in_head:
            raise RuntimeError(f"theory atom only accepted in head: {location_to_str(x.location)}")
        if type_ is TheoryAtomType.Body and not self.in_body:
            raise RuntimeError(f"theory atom only accepted in body: {location_to_str(x.location)}")
        if type_ is TheoryAtomType.Directive and not (self.in_head and self.is_directive):
            raise RuntimeError(f"theory atom must be a directive: {location_to_str(x.location)}")

        x = copy(x)
        x.term = element_parser(x.term)
        x.elements = element_parser.visit_list(x.elements)

        if x.guard is not None:
            if guard_table is None:
                raise RuntimeError(f"unexpected guard in theory atom: {location_to_str(x.location)}")

            guards, guard_parser = guard_table
            if x.guard.operator_name not in guards:
                raise RuntimeError(f"unexpected guard in theory atom: {location_to_str(x.location)}")

            x.guard = copy(x.guard)
            x.guard.term = guard_parser(x.guard.term)

        return x

def theory_parser_from_definition(x: AST) -> TheoryParser:
    """
    Turn an AST node of type TheoryDefinition into a TheoryParser.
    """
    assert x.type is ASTType.TheoryDefinition

    terms = {}
    atoms = {}

    for term_def in x.terms:
        term_table = {}

        for op_def in term_def.operators:
            if op_def.operator_type is TheoryOperatorType.BinaryLeft:
                op_type = BINARY
                op_assoc = LEFT
            elif op_def.operator_type is TheoryOperatorType.BinaryRight:
                op_type = BINARY
                op_assoc = RIGHT
            else:
                op_type = UNARY
                op_assoc = NONE

            term_table[(op_def.name, op_type)] = (op_def.priority, op_assoc)

        terms[term_def.name] = term_table

    for atom_def in x.atoms:
        guard = None
        if atom_def.guard is not None:
            guard = (atom_def.guard.operators, atom_def.guard.term)

        atoms[(atom_def.name, atom_def.arity)] = (atom_def.atom_type, atom_def.elements, guard)

    return TheoryParser(terms, atoms)


class SymbolicAtomRenamer(Transformer):
    '''
    A transformer to rename symbolic atoms.
    '''

    def __init__(self, rename_function: Callable[[str], str]):
        '''
        Initialize the transformer with the given function to rename symbolic
        atoms.
        '''
        self.rename_function = rename_function

    def visit_SymbolicAtom(self, x: AST) -> AST:
        '''
        Rename the given symbolic atom and the renamed version.
        '''
        term = x.term
        if term.type == ASTType.Symbol:
            sym = term.symbol
            term = Symbol(term.location, clingo.Function(self.rename_function(sym.name), sym.arguments, sym.positive))
        elif term.type == ASTType.Function:
            term = Function(term.location, self.rename_function(term.name), term.arguments, term.external)
        return SymbolicAtom(term)

def rename_symbolic_atoms(x: AST, rename_function: Callable[[str], str]) -> AST:
    '''
    Rename all symbolic atoms in the given AST node with the given function.
    '''
    return cast(AST, SymbolicAtomRenamer(rename_function)(x))

def prefix_symbolic_atoms(x: AST, prefix: str) -> AST:
    '''
    Prefix all symbolic atoms in the given AST with the given string.
    '''
    return rename_symbolic_atoms(x, lambda s: prefix + s)


@singledispatch
def _encode(x: Any) -> Any:
    raise RuntimeError(f"unknown value to encode: {x}")

@_encode.register
def _encode_str(x: str) -> str:
    return x

@_encode.register
def _encode_symbol(x: clingo.Symbol) -> str:
    return str(x)

@_encode.register
def _encode_bool(x: bool) -> bool:
    return x

@_encode.register
def _encode_int(x: int) -> int:
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
def _encode_theoryoptype(x: TheoryOperatorType) -> str:
    if x == TheoryOperatorType.Unary:
        return 'Unary'
    if x == TheoryOperatorType.BinaryLeft:
        return 'BinaryLeft'
    assert x == TheoryOperatorType.BinaryRight
    return 'BinaryRight'

@_encode.register
def _encode_afun(x: AggregateFunction) -> str:
    if x == AggregateFunction.Count:
        return 'Count'
    if x == AggregateFunction.Sum:
        return 'Sum'
    if x == AggregateFunction.SumPlus:
        return 'SumPlus'
    if x == AggregateFunction.Min:
        return 'Min'
    assert x == AggregateFunction.Max
    return 'Max'

@_encode.register
def _encode_comp(x: ComparisonOperator) -> str:
    if x == ComparisonOperator.GreaterThan:
        return 'GreaterThan'
    if x == ComparisonOperator.LessThan:
        return 'LessThan'
    if x == ComparisonOperator.LessEqual:
        return 'LessEqual'
    if x == ComparisonOperator.GreaterEqual:
        return 'GreaterEqual'
    if x == ComparisonOperator.NotEqual:
        return 'NotEqual'
    assert x == ComparisonOperator.Equal
    return 'Equal'

@_encode.register
def _encode_tatype(x: TheoryAtomType) -> str:
    if x == TheoryAtomType.Any:
        return 'Any'
    if x == TheoryAtomType.Head:
        return 'Head'
    if x == TheoryAtomType.Body:
        return 'Body'
    assert x == TheoryAtomType.Directive
    return 'Directive'

@_encode.register
def _encode_sctype(x: ScriptType) -> str:
    if x == ScriptType.Python:
        return 'Python'
    assert x == ScriptType.Lua
    return 'Lua'

@_encode.register
def _encode_unop(x: UnaryOperator) -> str:
    if x == UnaryOperator.Negation:
        return 'Negation'
    if x == UnaryOperator.Minus:
        return 'UnaryMinus'
    assert x == UnaryOperator.Absolute
    return 'Absolute'

@_encode.register
def _encode_binop(x: BinaryOperator) -> str:
    if x == BinaryOperator.And:
        return 'And'
    if x == BinaryOperator.Division:
        return 'Division'
    if x == BinaryOperator.Minus:
        return 'Minus'
    if x == BinaryOperator.Modulo:
        return 'Modulo'
    if x == BinaryOperator.Multiplication:
        return 'Multiplication'
    if x == BinaryOperator.Or:
        return 'Or'
    if x == BinaryOperator.Plus:
        return 'Plus'
    if x == BinaryOperator.Power:
        return 'Power'
    assert x == BinaryOperator.XOr
    return 'XOr'

@_encode.register
def _encode_list(x: list) -> List[Any]:
    return [_encode(y) for y in x]

@_encode.register
def _encode_none(x: None) -> None:
    return x

@_encode.register
def _encode_ast(x: AST) -> Any:
    return ast_to_dict(x)

def ast_to_dict(x: AST) -> dict:
    """
    Convert the given ast node into a dictionary representation whose elements
    only involve the data structures: `dict`, `list`, `int`, and `str`.

    The resulting value can be used with other python modules like the `yaml`
    or `pickle` modules.
    """
    ret = {"type": str(x.type)}
    for key, val in x.items():
        if key == 'location':
            enc = location_to_str(val)
        else:
            enc = _encode(val)
        ret[key] = enc
    return ret


@singledispatch
def _decode(x: Any, key: str) -> Any:
    raise RuntimeError(f"unknown key/value to decode: {key}: {x}")

@_decode.register
def _decode_str(x: str, key: str) -> Any:
    if key == "location":
        return str_to_location(x)

    if key == "symbol":
        return clingo.parse_term(x)

    if key == "sign":
        return getattr(Sign, x)

    if key == "comparison":
        return getattr(ComparisonOperator, x)

    if key == "script_type":
        return getattr(ScriptType, x)

    if key == "function":
        return getattr(AggregateFunction, x)

    if key == "operator":
        if x == "UnaryMinus":
            return UnaryOperator.Minus
        if hasattr(BinaryOperator, x):
            return getattr(BinaryOperator, x)
        return getattr(UnaryOperator, x)

    if key == "operator_type":
        return getattr(TheoryOperatorType, x)

    if key == "atom_type":
        return getattr(TheoryAtomType, x)

    assert key in ("name", "id", "code", "elements", "term", "list", "operator_name")
    return x

@_decode.register
def _decode_int(x: int, key: str) -> Any:
    # pylint: disable=unused-argument
    return x

@_decode.register
def _decode_none(x: None, key: str) -> Any:
    # pylint: disable=unused-argument
    return x

@_decode.register
def _decode_list(x: list, key_: str) -> Any:
    # pylint: disable=unused-argument
    return [_decode(y, "list") for y in x]

@_decode.register
def _decode_dict(x: dict, key_: str) -> Any:
    # pylint: disable=unused-argument
    return dict_to_ast(x)

def dict_to_ast(x: dict) -> AST:
    """
    Convert the dictionary representation of an AST node into an AST node.
    """
    return AST(getattr(ASTType, x['type']), **{key: _decode(value, key) for key, value in x.items() if key != "type"})
