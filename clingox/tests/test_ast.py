"""
Simple tests for ast manipulation.
"""

from unittest import TestCase
from typing import List, Optional, cast

import clingo
from clingo import Function
from clingo.ast import AST, ASTType, Location, Position, Transformer, parse_string, Variable
from .. import ast
from ..ast import (
    Arity, Associativity, TheoryTermParser, TheoryParser, TheoryAtomType,
    ast_to_dict, dict_to_ast, location_to_str, prefix_symbolic_atoms, str_to_location, theory_parser_from_definition,
    reify_symbolic_atoms)

TERM_TABLE = {"t": {("-", Arity.Unary): (3, Associativity.NoAssociativity),
                    ("**", Arity.Binary): (2, Associativity.Right),
                    ("*", Arity.Binary): (1, Associativity.Left),
                    ("+", Arity.Binary): (0, Associativity.Left),
                    ("-", Arity.Binary): (0, Associativity.Left)}}

ATOM_TABLE = {("p", 0): (TheoryAtomType.Head, "t", None),
              ("q", 1): (TheoryAtomType.Body, "t", None),
              ("r", 0): (TheoryAtomType.Directive, "t", (["<"], "t"))}

TEST_THEORY = """\
#theory test {
    t {
        -  : 3, unary;
        ** : 2, binary, right;
        *  : 1, binary, left;
        +  : 0, binary, left;
        -  : 0, binary, left
    };
    &p/0 : t, head;
    &q/1 : t, body;
    &r/0 : t, { < }, t, directive
}\
"""

LOC = Location(Position("a", 1, 2),
               Position("a", 1, 2))


class Extractor(Transformer):
    '''
    Simple visitor returning the first theory term in a program.
    '''
    # pylint: disable=invalid-name
    atom: Optional[AST]

    def __init__(self):
        self.atom = None

    def visit_TheoryAtom(self, x: AST):
        '''
        Extract theory atom.
        '''
        self.atom = x
        return x


def theory_atom(s: str) -> AST:
    """
    Convert string to theory term.
    """
    v = Extractor()

    def visit(stm):
        v(stm)

    parse_string(f"{s}.", visit)
    return cast(AST, v.atom)


def last_stm(s: str) -> AST:
    """
    Convert string to rule.
    """
    v = Extractor()
    stm = None

    def set_stm(x):
        nonlocal stm
        stm = x
        v(stm)

    parse_string(s, set_stm)

    return cast(AST, stm)


def parse_term(s: str) -> str:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return str(TheoryTermParser(TERM_TABLE["t"])(theory_atom(f"&p {{{s}}}").elements[0].terms[0]))


def parse_atom(s: str, parser: Optional[TheoryParser] = None) -> str:
    """
    Parse the given theory atom using a simple parse table for testing.
    """
    if parser is None:
        parser = TheoryParser(TERM_TABLE, ATOM_TABLE)

    return str(parser(theory_atom(s)))


def parse_stm(s: str, parser: Optional[TheoryParser] = None) -> str:
    """
    Parse the given theory atom using a simple parse table for testing.
    """
    if parser is None:
        parser = TheoryParser(TERM_TABLE, ATOM_TABLE)

    return str(parser(last_stm(s)))


def parse_theory(s: str) -> TheoryParser:
    """
    Turn the given theory into a parser.
    """
    parser = None

    def extract(stm):
        nonlocal parser
        if stm.ast_type == ASTType.TheoryDefinition:
            parser = theory_parser_from_definition(stm)

    parse_string(f"{s}.", extract)
    return cast(TheoryParser, parser)


def test_apply_function(s: str, f=lambda x: x):
    '''
    Parse the given program and apply f to it.
    '''
    prg: List[str]
    prg = []

    def append(stm):
        nonlocal prg
        ret = f(stm)
        if ret is not None:
            prg.append(ret)

    parse_string(s, append)
    return prg


def test_apply_function_str(s: str, f):
    '''
    Parse the given program and apply f to it.
    Returns a list of string
    '''
    return [str(x) for x in test_apply_function(s, f)]


def test_rename(s: str, f=lambda s: prefix_symbolic_atoms(s, "u_")):
    '''
    Parse the given program and rename symbolic atoms in it.
    '''
    return test_apply_function_str(s, f)


def test_reifier(s: str, f=None, sn=None):
    '''
    Parse the given program and rename symbolic atoms in it.
    '''
    def fun(x):
        if f is None and sn is None:
            return reify_symbolic_atoms(x, 'u')
        if sn is None:
            return reify_symbolic_atoms(x, 'u', f)
        if f is None:
            return reify_symbolic_atoms(x, 'u', reifing_strong_negation_name=sn)
        return reify_symbolic_atoms(x, 'u', f, sn)

    return test_apply_function_str(s, fun)


def test_ast_dict(tc: TestCase, s: str):
    '''
    Parse and transform a program to its dictionary representation.
    '''
    prg: list = []
    parse_string(s, prg.append)
    ret = [ast_to_dict(x) for x in prg]
    preamble = {'ast_type': 'Program', 'location': '<string>:1:1', 'name': 'base', 'parameters': []}
    tc.assertEqual(ret[0], preamble)
    tc.assertEqual(prg, [dict_to_ast(x) for x in ret])
    return ret[1:]


class TestAST(TestCase):
    '''
    Tests for AST manipulation.
    '''

    def test_loc(self):
        '''
        Test string representation of location.
        '''
        loc = LOC
        self.assertEqual(location_to_str(loc), "a:1:2")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position(loc.end.filename, loc.end.line, 4))
        self.assertEqual(location_to_str(loc), "a:1:2-4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position(loc.end.filename, 3, loc.end.column))
        self.assertEqual(location_to_str(loc), "a:1:2-3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position('b', loc.end.line, loc.end.column))
        self.assertEqual(location_to_str(loc), "a:1:2-b:3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(Position(r'a:1:2-3\:', loc.begin.line, loc.begin.column),
                       Position('b:1:2-3', loc.end.line, loc.end.column))
        self.assertEqual(location_to_str(loc), r'a\:1\:2-3\\\::1:2-b\:1\:2-3:3:4')
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        self.assertRaises(RuntimeError, str_to_location, 'a:1:2-')

    def test_parse_term(self):
        '''
        Test parsing of theory terms.
        '''
        self.assertEqual(parse_term("1+2"), "+(1,2)")
        self.assertEqual(parse_term("1+2+3"), "+(+(1,2),3)")
        self.assertEqual(parse_term("1+2*3"), "+(1,*(2,3))")
        self.assertEqual(parse_term("1**2**3"), "**(1,**(2,3))")
        self.assertEqual(parse_term("-1+2"), "+(-(1),2)")
        self.assertEqual(parse_term("f(1+2)+3"), "+(f(+(1,2)),3)")
        self.assertRaises(RuntimeError, parse_term, "1++2")

    def test_parse_atom(self):
        '''
        Test parsing of theory atoms.
        '''
        self.assertEqual(parse_atom("&p {1+2}"), "&p { +(1,2) }")
        self.assertEqual(parse_atom("&p {1+2+3}"), "&p { +(+(1,2),3) }")
        self.assertEqual(parse_atom("&q(1+2+3) { }"), "&q(((1+2)+3)) { }")
        self.assertEqual(parse_atom("&r { } < 1+2+3"), "&r { } < +(+(1,2),3)")
        # for coverage
        p = TheoryParser({'t': TheoryTermParser(TERM_TABLE["t"])}, ATOM_TABLE)
        self.assertEqual(parse_atom("&p {1+2}", p), "&p { +(1,2) }")

    def test_parse_atom_occ(self):
        """
        Test parsing of different theory atom types.
        """
        self.assertEqual(parse_stm("&p {1+2}."), "&p { +(1,2) }.")
        self.assertRaises(RuntimeError, parse_stm, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, parse_stm, "&q(1+2+3) { }.")
        self.assertEqual(parse_stm(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) { }.")
        self.assertEqual(parse_stm("&r { } < 1+2+3."), "&r { } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, parse_stm, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, parse_stm, ":- &r { } < 1+2+3.")

    def test_parse_theory(self):
        """
        Test creating parsers from theory definitions.
        """
        parser = parse_theory(TEST_THEORY)
        pa = lambda s: parse_atom(s, parser)  # noqa: E731
        pr = lambda s: parse_stm(s, parser)  # noqa: E731

        self.assertEqual(parse_atom("&p {1+2}", pa), "&p { +(1,2) }")
        self.assertEqual(parse_atom("&p {1+2+3}", pa), "&p { +(+(1,2),3) }")
        self.assertEqual(parse_atom("&q(1+2+3) { }", pa), "&q(((1+2)+3)) { }")
        self.assertEqual(parse_atom("&r { } < 1+2+3", pa), "&r { } < +(+(1,2),3)")

        self.assertEqual(pr("&p {1+2}."), "&p { +(1,2) }.")
        self.assertEqual(pr("#show x : &q(0) {1+2}."), "#show x : &q(0) { +(1,2) }.")
        self.assertEqual(pr(":~ &q(0) {1+2}. [0]"), ":~ &q(0) { +(1,2) }. [0@0]")
        self.assertEqual(pr("#edge (u, v) : &q(0) {1+2}."), "#edge (u,v) : &q(0) { +(1,2) }.")
        self.assertEqual(pr("#heuristic a : &q(0) {1+2}. [sign,true]"),
                         "#heuristic a : &q(0) { +(1,2) }. [sign@0,true]")
        self.assertEqual(pr("#project a : &q(0) {1+2}."), "#project a : &q(0) { +(1,2) }.")
        self.assertRaises(RuntimeError, pr, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, pr, "&q(1+2+3) { }.")
        self.assertEqual(pr(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) { }.")
        self.assertEqual(pr("&r { } < 1+2+3."), "&r { } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, pr, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, pr, ":- &r { } < 1+2+3.")
        self.assertRaises(RuntimeError, pr, "&s(1+2+3) { }.")
        self.assertRaises(RuntimeError, pr, "&p { } <= 3.")
        self.assertRaises(RuntimeError, pr, "&r { } <= 3.")

    def test_rename(self):
        '''
        Test renaming symbolic atoms.
        '''
        self.assertEqual(
            test_rename("a :- b(X,Y), not c(f(3,b))."),
            ['#program base.', 'u_a :- u_b(X,Y); not u_c(f(3,b)).'])
        sym = clingo.ast.SymbolicAtom(
                clingo.ast.UnaryOperation(LOC, clingo.ast.UnaryOperator.Minus,
                                          clingo.ast.Function(LOC, 'a', [], 0)))
        self.assertEqual(
            str(prefix_symbolic_atoms(sym, 'u_')),
            '-u_a')
        self.assertEqual(
            test_rename("-a :- -b(X,Y), not -c(f(3,b))."),
            ['#program base.', '-u_a :- -u_b(X,Y); not -u_c(f(3,b)).'])
        sym = ast.SymbolicAtom(ast.SymbolicTerm(LOC, Function('a', [Function('b')])))
        self.assertEqual(
            str(prefix_symbolic_atoms(sym, 'u_')),
            'u_a(b)')
        sym = ast.SymbolicAtom(Variable(LOC, 'B'))
        self.assertEqual(
            prefix_symbolic_atoms(sym, 'u'),
            sym)

    def test_reifier(self):
        '''
        Test reifying symbolic atoms.
        '''
        self.assertEqual(
            test_reifier("a."),
            ['#program base.', 'u(a).'])
        self.assertEqual(
            test_reifier("a :- b(X,Y), not c(f(3,b))."),
            ['#program base.', 'u(a) :- u(b(X,Y)); not u(c(f(3,b))).'])
        sym = clingo.ast.SymbolicAtom(
                clingo.ast.UnaryOperation(LOC, clingo.ast.UnaryOperator.Minus,
                                          clingo.ast.Function(LOC, 'a', [], 0)))
        self.assertEqual(
            str(reify_symbolic_atoms(sym, 'u')),
            '-u(a)')
        self.assertEqual(
            str(reify_symbolic_atoms(sym, 'u', reifing_strong_negation_name='s')),
            's(a)')
        self.assertEqual(
            test_reifier("-a :- -b(X,Y), not -c(f(3,b))."),
            ['#program base.', '-u(a) :- -u(b(X,Y)); not -u(c(f(3,b))).'])
        self.assertEqual(
            test_reifier("-a :- b(X,Y), not -c(f(3,b)). a :- -b(X,Y), not c(f(3,b)).", sn='s'),
            ['#program base.', 's(a) :- u(b(X,Y)); not s(c(f(3,b))).', 'u(a) :- s(b(X,Y)); not u(c(f(3,b))).'])
        self.assertEqual(
            test_reifier("a :- b(X,Y), not c(f(3,b)).", lambda x: [Variable(LOC, 'T'), Variable(LOC, 'I')]),
            ['#program base.', 'u(a,T,I) :- u(b(X,Y),T,I); not u(c(f(3,b)),T,I).'])
        self.assertEqual(
            test_reifier("a :- -b(X,Y), not c(f(3,b)).", lambda x: [Variable(LOC, 'T'), Variable(LOC, 'I')], sn='s'),
            ['#program base.', 'u(a,T,I) :- s(b(X,Y),T,I); not u(c(f(3,b)),T,I).'])
        sym = ast.SymbolicAtom(ast.SymbolicTerm(LOC, Function('a', [Function('b')])))
        self.assertEqual(
            str(reify_symbolic_atoms(sym, 'u')),
            'u(a(b))')
        sym = ast.SymbolicAtom(Variable(LOC, 'B'))
        self.assertEqual(
            prefix_symbolic_atoms(sym, 'u'),
            sym)

    def test_encode_term(self):
        '''
        Test encoding of terms in AST.
        '''
        self.assertEqual(
            test_ast_dict(self, 'a(1).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-6',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-5', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-5', 'name': 'a',
                                           'arguments': [{'ast_type': 'SymbolicTerm', 'location': '<string>:1:3-4',
                                                          'symbol': '1'}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(X).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-6',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-5', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-5', 'name': 'a',
                                           'arguments': [{'ast_type': 'Variable', 'location': '<string>:1:3-4',
                                                          'name': 'X'}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(-1).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-6', 'name': 'a',
                                           'arguments': [{'ast_type': 'UnaryOperation', 'location': '<string>:1:3-5',
                                                          'operator_type': 0,
                                                          'argument': {'ast_type': 'SymbolicTerm',
                                                                       'location': '<string>:1:4-5', 'symbol': '1'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(~1).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-6', 'name': 'a',
                                           'arguments': [{'ast_type': 'UnaryOperation', 'location': '<string>:1:3-5',
                                                          'operator_type': 1,
                                                          'argument': {'ast_type': 'SymbolicTerm',
                                                                       'location': '<string>:1:4-5', 'symbol': '1'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(|1|).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'UnaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 2,
                                                          'argument': {'ast_type': 'SymbolicTerm',
                                                                       'location': '<string>:1:4-5', 'symbol': '1'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1+2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 3,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1-2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 4,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1*2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 5,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1/2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 6,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1\\2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 7,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1**2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-9',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-8', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-8', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-7',
                                                          'operator_type': 8,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:6-7', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1^2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 0,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1?2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 1,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1&2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                           'arguments': [{'ast_type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                          'operator_type': 2,
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:5-6', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1..2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-9',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-8', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-8', 'name': 'a',
                                           'arguments': [{'ast_type': 'Interval', 'location': '<string>:1:3-7',
                                                          'left': {'ast_type': 'SymbolicTerm',
                                                                   'location': '<string>:1:3-4', 'symbol': '1'},
                                                          'right': {'ast_type': 'SymbolicTerm',
                                                                    'location': '<string>:1:6-7', 'symbol': '2'}}],
                                           'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a(1;2).'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Pool', 'location': '<string>:1:1-7',
                                           'arguments': [{'ast_type': 'Function', 'location': '<string>:1:1-7',
                                                          'name': 'a',
                                                          'arguments': [{'ast_type': 'SymbolicTerm',
                                                                         'location': '<string>:1:3-4', 'symbol': '1'}],
                                                          'external': 0},
                                                         {'ast_type': 'Function', 'location': '<string>:1:1-7',
                                                          'name': 'a',
                                                          'arguments': [{'ast_type': 'SymbolicTerm',
                                                                         'location': '<string>:1:5-6', 'symbol': '2'}],
                                                          'external': 0}]}}},
              'body': []}])

    def test_encode_literal(self):
        '''
        Tests for converting between python and ast representation of literals.
        '''
        self.assertEqual(
            test_ast_dict(self, 'a.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-3',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-2', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-2', 'name': 'a',
                                           'arguments': [], 'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'not a.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 1,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:5-6', 'name': 'a',
                                           'arguments': [], 'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'not not a.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-11',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-10', 'sign': 2,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:9-10', 'name': 'a',
                                           'arguments': [], 'external': 0}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a <= b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 2,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:6-7', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a < b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 1,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:5-6', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a >= b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 3,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:6-7', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a > b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 0,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:5-6', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a = b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-6', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 5,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:5-6', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a != b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-7', 'sign': 0,
                       'atom': {'ast_type': 'Comparison', 'comparison': 4,
                                'left': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-2', 'symbol': 'a'},
                                'right': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:6-7', 'symbol': 'b'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, 'a : b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'ast_type': 'Disjunction', 'location': '<string>:1:1-6',
                       'elements': [{'ast_type': 'ConditionalLiteral', 'location': '<string>:1:1-2',
                                     'literal': {'ast_type': 'Literal', 'location': '<string>:1:1-2', 'sign': 0,
                                                 'atom': {'ast_type': 'SymbolicAtom',
                                                          'symbol': {'ast_type': 'Function',
                                                                     'location': '<string>:1:1-2', 'name': 'a',
                                                                     'arguments': [], 'external': 0}}},
                                     'condition': [{'ast_type': 'Literal', 'location': '<string>:1:5-6', 'sign': 0,
                                                    'atom': {'ast_type': 'SymbolicAtom',
                                                             'symbol': {'ast_type': 'Function',
                                                                        'location': '<string>:1:5-6', 'name': 'b',
                                                                        'arguments': [], 'external': 0}}}]}]},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, ':- a : b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-10',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-10', 'sign': 0,
                       'atom': {'ast_type': 'BooleanConstant', 'value': 0}},
              'body': [{'ast_type': 'ConditionalLiteral', 'location': '<string>:1:4-9',
                        'literal': {'ast_type': 'Literal', 'location': '<string>:1:4-5', 'sign': 0,
                                    'atom': {'ast_type': 'SymbolicAtom',
                                             'symbol': {'ast_type': 'Function', 'location': '<string>:1:4-5',
                                                        'name': 'a', 'arguments': [], 'external': 0}}},
                        'condition': [{'ast_type': 'Literal', 'location': '<string>:1:8-9', 'sign': 0,
                                       'atom': {'ast_type': 'SymbolicAtom',
                                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:8-9',
                                                           'name': 'b', 'arguments': [], 'external': 0}}}]}]}])
        self.assertEqual(
            test_ast_dict(self, '#sum {1:a:b} <= 2.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-19',
              'head': {'ast_type': 'HeadAggregate', 'location': '<string>:1:1-18',
                       'left_guard': {'ast_type': 'AggregateGuard', 'comparison': 3,
                                      'term': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:17-18',
                                               'symbol': '2'}},
                       'function': 1,
                       'elements': [{'ast_type': 'HeadAggregateElement',
                                     'terms': [{'ast_type': 'SymbolicTerm', 'location': '<string>:1:7-8',
                                                'symbol': '1'}],
                                     'condition': {'ast_type': 'ConditionalLiteral', 'location': '<string>:1:9-10',
                                                   'literal': {'ast_type': 'Literal', 'location': '<string>:1:9-10',
                                                               'sign': 0,
                                                               'atom': {'ast_type': 'SymbolicAtom',
                                                                        'symbol': {'ast_type': 'Function',
                                                                                   'location': '<string>:1:9-10',
                                                                                   'name': 'a', 'arguments': [],
                                                                                   'external': 0}}},
                                                   'condition': [{'ast_type': 'Literal', 'location': '<string>:1:11-12',
                                                                  'sign': 0,
                                                                  'atom': {'ast_type': 'SymbolicAtom',
                                                                           'symbol': {'ast_type': 'Function',
                                                                                      'location': '<string>:1:11-12',
                                                                                      'name': 'b', 'arguments': [],
                                                                                      'external': 0}}}]}}],
                       'right_guard': None},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, ':- #sum {1:b} <= 2.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-20',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-20', 'sign': 0,
                       'atom': {'ast_type': 'BooleanConstant', 'value': 0}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:4-19', 'sign': 0,
                        'atom': {'ast_type': 'BodyAggregate', 'location': '<string>:1:4-19',
                                 'left_guard': {'ast_type': 'AggregateGuard', 'comparison': 3,
                                                'term': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:18-19',
                                                         'symbol': '2'}},
                                 'function': 1,
                                 'elements': [{'ast_type': 'BodyAggregateElement',
                                               'terms': [{'ast_type': 'SymbolicTerm', 'location': '<string>:1:10-11',
                                                          'symbol': '1'}],
                                               'condition': [{'ast_type': 'Literal', 'location': '<string>:1:12-13',
                                                              'sign': 0,
                                                              'atom': {'ast_type': 'SymbolicAtom',
                                                                       'symbol': {'ast_type': 'Function',
                                                                                  'location': '<string>:1:12-13',
                                                                                  'name': 'b', 'arguments': [],
                                                                                  'external': 0}}}]}],
                                 'right_guard': None}}]}])
        self.assertEqual(
            test_ast_dict(self, '#count {}.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-11',
              'head': {'ast_type': 'HeadAggregate', 'location': '<string>:1:1-10', 'left_guard': None,
                       'function': 0, 'elements': [], 'right_guard': None},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, '#min {}.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-9',
              'head': {'ast_type': 'HeadAggregate', 'location': '<string>:1:1-8', 'left_guard': None,
                       'function': 3, 'elements': [], 'right_guard': None},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, '#max {}.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-9',
              'head': {'ast_type': 'HeadAggregate', 'location': '<string>:1:1-8', 'left_guard': None,
                       'function': 4, 'elements': [], 'right_guard': None},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, '#sum+ {}.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-10',
              'head': {'ast_type': 'HeadAggregate', 'location': '<string>:1:1-9', 'left_guard': None,
                       'function': 2, 'elements': [], 'right_guard': None},
              'body': []}])

    def test_encode_theory(self):
        '''
        Tests for converting between python and ast representation of theory
        releated constructs.
        '''
        self.assertEqual(
            test_ast_dict(self, '#theory t { }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-15', 'name': 't', 'terms': [], 'atoms': []}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { t { + : 1, unary } }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-34', 'name': 't',
              'terms': [{'ast_type': 'TheoryTermDefinition', 'location': '<string>:1:13-31', 'name': 't',
                         'operators': [{'ast_type': 'TheoryOperatorDefinition', 'location': '<string>:1:17-29',
                                        'name': '+', 'priority': 1, 'operator_type': 0}]}],
              'atoms': []}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { t { + : 1, binary, left } }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-41', 'name': 't',
              'terms': [{'ast_type': 'TheoryTermDefinition', 'location': '<string>:1:13-38', 'name': 't',
                         'operators': [{'ast_type': 'TheoryOperatorDefinition', 'location': '<string>:1:17-36',
                                        'name': '+', 'priority': 1, 'operator_type': 1}]}],
              'atoms': []}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { t { + : 1, binary, right } }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-42', 'name': 't',
              'terms': [{'ast_type': 'TheoryTermDefinition', 'location': '<string>:1:13-39', 'name': 't',
                         'operators': [{'ast_type': 'TheoryOperatorDefinition', 'location': '<string>:1:17-37',
                                        'name': '+', 'priority': 1, 'operator_type': 2}]}],
              'atoms': []}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { &p/0 : t, any }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-29', 'name': 't', 'terms': [],
              'atoms': [{'ast_type': 'TheoryAtomDefinition', 'location': '<string>:1:13-26', 'atom_type': 2,
                         'name': 'p', 'arity': 0, 'term': 't', 'guard': None}]}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { &p/0 : t, head }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-30', 'name': 't', 'terms': [],
              'atoms': [{'ast_type': 'TheoryAtomDefinition', 'location': '<string>:1:13-27', 'atom_type': 0,
                         'name': 'p', 'arity': 0, 'term': 't', 'guard': None}]}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { &p/1 : t, body }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-30', 'name': 't', 'terms': [],
              'atoms': [{'ast_type': 'TheoryAtomDefinition', 'location': '<string>:1:13-27', 'atom_type': 1,
                         'name': 'p', 'arity': 1, 'term': 't', 'guard': None}]}])
        self.assertEqual(
            test_ast_dict(self, '#theory t { &p/2 : t, { < }, t, directive }.'),
            [{'ast_type': 'TheoryDefinition', 'location': '<string>:1:1-45', 'name': 't', 'terms': [],
              'atoms': [{'ast_type': 'TheoryAtomDefinition', 'location': '<string>:1:13-42', 'atom_type': 3,
                         'name': 'p', 'arity': 2, 'term': 't',
                         'guard': {'ast_type': 'TheoryGuardDefinition', 'operators': ['<'], 'term': 't'}}]}])
        self.assertEqual(
            test_ast_dict(self, '&p { }.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'TheoryAtom', 'location': '<string>:1:2-3',
                       'term': {'ast_type': 'Function', 'location': '<string>:1:2-3', 'name': 'p',
                                'arguments': [], 'external': 0},
                       'elements': [], 'guard': None}, 'body': []}])
        self.assertEqual(
            test_ast_dict(self, ':- &p { }.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-11',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-11', 'sign': 0,
                       'atom': {'ast_type': 'BooleanConstant', 'value': 0}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:4-10', 'sign': 0,
                        'atom': {'ast_type': 'TheoryAtom', 'location': '<string>:1:5-6',
                                 'term': {'ast_type': 'Function', 'location': '<string>:1:5-6', 'name': 'p',
                                          'arguments': [], 'external': 0},
                                 'elements': [], 'guard': None}}]}])
        self.assertEqual(
            test_ast_dict(self, '&p { } > 2.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-12',
              'head': {'ast_type': 'TheoryAtom', 'location': '<string>:1:2-3',
                       'term': {'ast_type': 'Function', 'location': '<string>:1:2-3', 'name': 'p', 'arguments': [],
                                'external': 0},
                       'elements': [], 'guard': {'ast_type': 'TheoryGuard', 'operator_name': '>',
                                                 'term': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:10-11',
                                                          'symbol': '2'}}},
              'body': []}])
        self.assertEqual(
            test_ast_dict(self, '&p { a,b: q }.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-15',
              'head': {'ast_type': 'TheoryAtom', 'location': '<string>:1:2-3',
                       'term': {'ast_type': 'Function', 'location': '<string>:1:2-3', 'name': 'p', 'arguments': [],
                                'external': 0},
                       'elements': [{'ast_type': 'TheoryAtomElement',
                                     'terms': [{'ast_type': 'SymbolicTerm', 'location': '<string>:1:6-7',
                                                'symbol': 'a'},
                                               {'ast_type': 'SymbolicTerm', 'location': '<string>:1:8-9',
                                                'symbol': 'b'}],
                                     'condition': [{'ast_type': 'Literal', 'location': '<string>:1:11-12', 'sign': 0,
                                                    'atom': {'ast_type': 'SymbolicAtom',
                                                             'symbol': {'ast_type': 'Function',
                                                                        'location': '<string>:1:11-12', 'name': 'q',
                                                                        'arguments': [], 'external': 0}}}]}],
                       'guard': None},
              'body': []}])

    def test_encode_statement(self):
        '''
        Tests for converting between python and ast representation of statements.
        '''
        self.assertEqual(
            test_ast_dict(self, 'a :- b.'),
            [{'ast_type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'ast_type': 'Literal', 'location': '<string>:1:1-2', 'sign': 0,
                       'atom': {'ast_type': 'SymbolicAtom',
                                'symbol': {'ast_type': 'Function', 'location': '<string>:1:1-2',
                                           'name': 'a', 'arguments': [], 'external': 0}}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:6-7', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:6-7',
                                            'name': 'b', 'arguments': [], 'external': 0}}}]}])
        self.assertEqual(
            test_ast_dict(self, '#defined x/0.'),
            [{'ast_type': 'Defined', 'location': '<string>:1:1-14', 'name': 'x', 'arity': 0, 'positive': True}])
        self.assertEqual(
            test_ast_dict(self, '#show a : b.'),
            [{'ast_type': 'ShowTerm', 'location': '<string>:1:1-13',
              'term': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:7-8', 'symbol': 'a'},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:11-12', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:11-12', 'name': 'b',
                                            'arguments': [], 'external': 0}}}],
              'csp': 0}])
        self.assertEqual(
            test_ast_dict(self, '#show a/0.'),
            [{'ast_type': 'ShowSignature', 'location': '<string>:1:1-11', 'name': 'a', 'arity': 0, 'positive': True,
              'csp': 0}])
        self.assertEqual(
            test_ast_dict(self, '#minimize { 1@2,a : b }.'),
            [{'ast_type': 'Minimize', 'location': '<string>:1:13-22',
              'weight': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:13-14', 'symbol': '1'},
              'priority': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:15-16', 'symbol': '2'},
              'terms': [{'ast_type': 'SymbolicTerm', 'location': '<string>:1:17-18', 'symbol': 'a'}],
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:21-22', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:21-22', 'name': 'b',
                                            'arguments': [], 'external': 0}}}]}])
        self.assertEqual(
            test_ast_dict(self, '#script (python) blub! #end.'),
            [{'ast_type': 'Script', 'location': '<string>:1:1-29', 'name': 'python',
              'code': 'blub!'}])
        self.assertEqual(
            test_ast_dict(self, '#script (lua) blub! #end.'),
            [{'ast_type': 'Script', 'location': '<string>:1:1-26', 'code': 'blub!', 'name': 'lua'}])
        self.assertEqual(
            test_ast_dict(self, '#program x(y).'),
            [{'ast_type': 'Program', 'location': '<string>:1:1-15', 'name': 'x',
              'parameters': [{'ast_type': 'Id', 'location': '<string>:1:12-13', 'name': 'y'}]}])
        self.assertEqual(
            test_ast_dict(self, '#project a/0.'),
            [{'ast_type': 'ProjectSignature', 'location': '<string>:1:1-14', 'name': 'a', 'arity': 0,
              'positive': True}])
        self.assertEqual(
            test_ast_dict(self, '#project a : b.'),
            [{'ast_type': 'ProjectAtom', 'location': '<string>:1:1-16',
              'atom': {'ast_type': 'SymbolicAtom',
                       'symbol': {'ast_type': 'Function', 'location': '<string>:1:10-11', 'name': 'a',
                                  'arguments': [], 'external': 0}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:14-15', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:14-15', 'name': 'b',
                                            'arguments': [], 'external': 0}}}]}])
        self.assertEqual(
            test_ast_dict(self, '#external x : y. [X]'),
            [{'ast_type': 'External', 'location': '<string>:1:1-21',
              'atom': {'ast_type': 'SymbolicAtom',
                       'symbol': {'ast_type': 'Function', 'location': '<string>:1:11-12', 'name': 'x',
                                  'arguments': [], 'external': 0}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:15-16', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:15-16', 'name': 'y',
                                            'arguments': [], 'external': 0}}}],
              'external_type': {'ast_type': 'Variable', 'location': '<string>:1:19-20', 'name': 'X'}}])
        self.assertEqual(
            test_ast_dict(self, '#edge (u,v) : b.'),
            [{'ast_type': 'Edge', 'location': '<string>:1:1-17',
              'node_u': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:8-9', 'symbol': 'u'},
              'node_v': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:10-11', 'symbol': 'v'},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:15-16', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:15-16', 'name': 'b',
                                            'arguments': [], 'external': 0}}}]}])
        self.assertEqual(
            test_ast_dict(self, '#heuristic a : b. [p,X]'),
            [{'ast_type': 'Heuristic', 'location': '<string>:1:1-24',
              'atom': {'ast_type': 'SymbolicAtom',
                       'symbol': {'ast_type': 'Function', 'location': '<string>:1:12-13', 'name': 'a',
                                  'arguments': [], 'external': 0}},
              'body': [{'ast_type': 'Literal', 'location': '<string>:1:16-17', 'sign': 0,
                        'atom': {'ast_type': 'SymbolicAtom',
                                 'symbol': {'ast_type': 'Function', 'location': '<string>:1:16-17', 'name': 'b',
                                            'arguments': [], 'external': 0}}}],
              'bias': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:20-21', 'symbol': 'p'},
              'priority': {'ast_type': 'SymbolicTerm', 'location': '<string>:1:1-24', 'symbol': '0'},
              'modifier': {'ast_type': 'Variable', 'location': '<string>:1:22-23', 'name': 'X'}}])

    def test_dict_ast_error(self):
        '''
        Test error condititons when converting between ast and dict.
        '''
        self.assertRaises(RuntimeError, dict_to_ast, {"ast_type": "Rule", "body": set()})
