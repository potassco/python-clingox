'''
This module provides highlevel functions to work with clingo's AST.

TODO:
- unpooling as in clingcon
'''

from typing import Any, Callable, cast, Iterator, List, Mapping, Optional, Set, Tuple, TypeVar, Union
from copy import copy

import clingo
from clingo.ast import AST, ASTType, Function, Symbol, SymbolicAtom, TheoryFunction, TheoryAtomType, TheoryOperatorType
from .theory import is_operator


UNARY: bool = True
BINARY: bool = not UNARY
LEFT: bool = True
RIGHT: bool = not LEFT
NONE: bool = RIGHT

class Visitor:
    '''
    A visitor for clingo's abstart syntaxt tree.

    This class should be derived from. Implementing functions with name
    `visit_<type>` can be used to visit nodes of the given type.

    Implements: `Callable[[AST], None]`.
    '''
    def visit_children(self, x: AST, *args: Any, **kwargs: Any):
        '''
        Visit the children of an AST node.
        '''
        for key in x.child_keys:
            y = getattr(x, key)
            if isinstance(y, AST):
                self.visit(y, *args, **kwargs)
            elif isinstance(y, List): # pylint: disable=all
                self.visit_list(y, *args, **kwargs)
            else:
                assert y is None

    def visit_list(self, x: List[AST], *args: Any, **kwargs: Any):
        '''
        Visit a sequence of AST nodes.
        '''
        for y in x:
            self.visit(y, *args, **kwargs)

    def visit(self, x: AST, *args: Any, **kwargs: Any):
        '''
        Generic visit method dispatching to specific member functions to visit
        child nodes.
        '''
        attr = "visit_" + str(x.type)
        if hasattr(self, attr):
            getattr(self, attr)(x, *args, **kwargs)
        else:
            self.visit_children(x, *args, **kwargs)

    def __call__(self, x: AST, *args: Any, **kwargs: Any):
        '''
        Alternative to call visit.
        '''
        self.visit(x, *args, **kwargs)


class Transformer:
    '''
    This class is similar to the `Visitor` but allows for mutating the AST by
    returning modified AST nodes from the visit methods.

    Implements: `Callable[[AST], Optional[AST]]`.
    '''
    def visit_children(self, x: AST, *args: Any, **kwargs: Any) -> AST:
        '''
        Visit the children of an AST node.
        '''
        copied = False
        for key in x.child_keys:
            y = getattr(x, key)
            z: Union[AST, List, None]
            if isinstance(y, AST):
                z = self.visit(y, *args, **kwargs)
            elif isinstance(y, List): # pylint: disable=all
                z = self.visit_list(y, *args, **kwargs)
            else:
                z = None
            if y is z:
                continue
            if not copied:
                copied = True
                x = copy(x)
            setattr(x, key, z)
        return x

    def _seq(self, i: int, z: Optional[AST], x: List[AST], *args: Any, **kwargs: Any) -> Iterator[Optional[AST]]:
        for y in x[:i]:
            yield y
        yield z
        for y in x[i+1:]:
            yield self.visit(y, *args, **kwargs)

    def visit_list(self, x: List[AST], *args: Any, **kwargs: Any) -> List[AST]:
        '''
        Visit a sequence of AST nodes.

        If a transformer returns None, the element is removed from the list.
        '''
        for i, y in enumerate(x):
            z = self.visit(y, *args, **kwargs)
            if y is not z:
                return list(w for w in self._seq(i, z, x, *args, **kwargs) if w is not None)
        return x

    def visit(self, x: AST, *args: Any, **kwargs: Any) -> Optional[AST]:
        '''
        Generic visit method dispatching to specific member functions to visit
        child nodes.
        '''
        attr = "visit_" + str(x.type)
        if hasattr(self, attr):
            return getattr(self, attr)(x, *args, **kwargs)
        return self.visit_children(x, *args, **kwargs)

    def __call__(self, x: AST, *args: Any, **kwargs: Any) -> Optional[AST]:
        '''
        Alternative to call visit.
        '''
        return self.visit(x, *args, **kwargs)


def str_location(loc: Any) -> str:
    """
    This function takes a location from a clingo AST and transforms it into a
    readable format.
    """
    begin = loc["begin"]
    end = loc["end"]
    ret = "{}:{}:{}".format(begin["filename"], begin["line"], begin["column"])
    dash = True
    eq = begin["filename"] == end["filename"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["filename"])
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
            raise RuntimeError("cannot parse operator `{}`: {}".format(operator, str_location(location)))

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
            raise RuntimeError(f"theory atom definiton not found: {str_location(x.location)}")

        type_, element_parser, guard_table = self._table[(name, arity)]
        if type_ is TheoryAtomType.Head and not self.in_head:
            raise RuntimeError(f"theory atom only accepted in head: {str_location(x.location)}")
        if type_ is TheoryAtomType.Body and not self.in_body:
            raise RuntimeError(f"theory atom only accepted in body: {str_location(x.location)}")
        if type_ is TheoryAtomType.Directive and not (self.in_head and self.is_directive):
            raise RuntimeError(f"theory atom must be a directive: {str_location(x.location)}")

        x = copy(x)
        x.term = element_parser(x.term)
        x.elements = element_parser.visit_list(x.elements)

        if x.guard is not None:
            if guard_table is None:
                raise RuntimeError(f"unexpected guard in theory atom: {str_location(x.location)}")

            guards, guard_parser = guard_table
            if x.guard.operator_name not in guards:
                raise RuntimeError(f"unexpected guard in theory atom: {str_location(x.location)}")

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
