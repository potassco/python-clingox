"""
Simple tests for ast manipulation.
"""

from copy import deepcopy
from unittest import TestCase
from typing import List, Optional, cast

from clingo import parse_program, Function
from clingo.ast import AST, ASTType, Variable
from .. import ast
from ..ast import (Visitor, Transformer, TheoryTermParser, TheoryParser, TheoryAtomType,
                   location_to_str, prefix_symbolic_atoms, str_to_location, theory_parser_from_definition)
from ..ast_repr import as_dict, from_dict

TERM_TABLE = {"t": {("-", ast.UNARY):  (3, ast.NONE),
                    ("**", ast.BINARY): (2, ast.RIGHT),
                    ("*", ast.BINARY): (1, ast.LEFT),
                    ("+", ast.BINARY): (0, ast.LEFT),
                    ("-", ast.BINARY): (0, ast.LEFT)}}

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

LOC = {"begin": {"filename": "a",
                 "line": 1,
                 "column": 2},
       "end": {"filename": "a",
               "line": 1,
               "column": 2}}

class TestVisitor(Visitor):
    '''
    Simple visitor marking what was visited in the string result.
    '''
    # pylint: disable=invalid-name
    result: str

    def __init__(self):
        self.result = ""

    def visit_Variable(self, x: AST):
        '''
        Visit a variable and add mark "v".
        '''
        self.result += "v"
        self.visit_children(x)

    def visit_SymbolicAtom(self, x: AST):
        '''
        Visit an atom and add mark "a".
        '''
        self.result += "a"
        self.visit_children(x)

    def visit_Literal(self, x: AST):
        '''
        Visit a literal and add mark "l".
        '''
        self.result += "l"
        self.visit_children(x)

    def visit_Rule(self, x: AST):
        '''
        Visit a rule and add mark "r".
        '''
        self.result += "r"
        self.visit_children(x)

def test_visit(s: str) -> str:
    '''
    Test the visitor by parsing the given program and using the TestVistor on
    it.
    '''
    prg: List[AST]
    prg = []
    parse_program(s, prg.append)
    visitor = TestVisitor()
    visitor.visit_list(prg)
    return visitor.result


class TestTransformer(Transformer):
    '''
    Simple transformer renaming variables and dropping program statements and
    guards of theory atoms.
    '''
    # pylint: disable=invalid-name, unused-argument
    result: str

    def __init__(self):
        self.result = ""

    def visit_Program(self, x: AST, suffix: str) -> Optional[AST]:
        '''
        Remove program parts.
        '''
        return None

    def visit_Variable(self, x: AST, suffix: str) -> Optional[AST]:
        '''
        Add suffix to variable.
        '''
        return Variable(x.location, x.name + suffix)

    def visit_TheoryGuard(self, x: AST, suffix: str) -> Optional[AST]:
        '''
        Drop guard of theory atom.
        '''
        return None

def test_transform(s: str) -> str:
    '''
    Test the transformer by parsing the given program and using the
    TestTransformer on it.
    '''
    prg: List[AST]
    prg = []
    parse_program(s, prg.append)
    v = TestTransformer()
    return "\n".join(str(x) for x in v.visit_list(prg, "_x"))


class Extractor(Visitor):
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

def theory_atom(s: str) -> AST:
    """
    Convert string to theory term.
    """
    v = Extractor()
    parse_program(f"{s}.", v)
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

    parse_program(s, set_stm)

    return cast(AST, stm)

def parse_term(s: str) -> str:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return str(TheoryTermParser(TERM_TABLE["t"])(theory_atom(f"&p {{{s}}}").elements[0].tuple[0]))

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
        if stm.type == ASTType.TheoryDefinition:
            parser = theory_parser_from_definition(stm)
    parse_program(f"{s}.", extract)
    return cast(TheoryParser, parser)


class TestAST(TestCase):
    '''
    Tests for AST manipulation.
    '''

    def test_loc(self):
        '''
        Test string representation of location.
        '''
        loc = deepcopy(LOC)
        self.assertEqual(location_to_str(loc), "a:1:2")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc['end']['column'] = 4
        self.assertEqual(location_to_str(loc), "a:1:2-4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc['end']['line'] = 3
        self.assertEqual(location_to_str(loc), "a:1:2-3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc['end']['filename'] = 'b'
        self.assertEqual(location_to_str(loc), "a:1:2-b:3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc['begin']['filename'] = r'a:1:2-3\:'
        loc['end']['filename'] = 'b:1:2-3'
        self.assertEqual(location_to_str(loc), r'a\:1\:2-3\\\::1:2-b\:1\:2-3:3:4')
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        self.assertRaises(RuntimeError, str_to_location, 'a:1:2-')

    def test_visit(self):
        '''
        Test the visitor.
        '''
        self.assertEqual(test_visit("a(X) :- p(X)."), "rlavlav")
        self.assertEqual(test_visit("a(X) :- &p { }."), "rlavl")

    def test_transform(self):
        '''
        Test the transformer.
        '''
        self.assertEqual(test_transform("a(X) :- p(X)."), "a(X_x) :- p(X_x).")
        self.assertEqual(test_transform("a(X) :- p(X), q; r; s."), "a(X_x) :- p(X_x); q; r; s.")
        self.assertEqual(test_transform("a(X) :- p, q(X), r; s."), "a(X_x) :- p; q(X_x); r; s.")
        self.assertEqual(test_transform("a(X) :- p, q, r(X); s."), "a(X_x) :- p; q; r(X_x); s.")
        self.assertEqual(test_transform("&p{} < 0."), "&p {  }.")
        self.assertEqual(test_transform("&p{}."), "&p {  }.")

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
        self.assertEqual(parse_atom("&p {1+2}"), "&p { +(1,2) :  }")
        self.assertEqual(parse_atom("&p {1+2+3}"), "&p { +(+(1,2),3) :  }")
        self.assertEqual(parse_atom("&q(1+2+3) { }"), "&q(((1+2)+3)) {  }")
        self.assertEqual(parse_atom("&r { } < 1+2+3"), "&r {  } < +(+(1,2),3)")
        # for coverage
        p = TheoryParser({'t': TheoryTermParser(TERM_TABLE["t"])}, ATOM_TABLE)
        self.assertEqual(parse_atom("&p {1+2}", p), "&p { +(1,2) :  }")

    def test_parse_atom_occ(self):
        """
        Test parsing of different theory atom types.
        """
        self.assertEqual(parse_stm("&p {1+2}."), "&p { +(1,2) :  }.")
        self.assertRaises(RuntimeError, parse_stm, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, parse_stm, "&q(1+2+3) { }.")
        self.assertEqual(parse_stm(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) {  }.")
        self.assertEqual(parse_stm("&r { } < 1+2+3."), "&r {  } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, parse_stm, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, parse_stm, ":- &r { } < 1+2+3.")

    def test_parse_theory(self):
        """
        Test creating parsers from theory definitions.
        """
        parser = parse_theory(TEST_THEORY)
        pa = lambda s: parse_atom(s, parser)
        pr = lambda s: parse_stm(s, parser)

        self.assertEqual(parse_atom("&p {1+2}", pa), "&p { +(1,2) :  }")
        self.assertEqual(parse_atom("&p {1+2+3}", pa), "&p { +(+(1,2),3) :  }")
        self.assertEqual(parse_atom("&q(1+2+3) { }", pa), "&q(((1+2)+3)) {  }")
        self.assertEqual(parse_atom("&r { } < 1+2+3", pa), "&r {  } < +(+(1,2),3)")

        self.assertEqual(pr("&p {1+2}."), "&p { +(1,2) :  }.")
        self.assertEqual(pr("#show x : &q(0) {1+2}."), "#show x : &q(0) { +(1,2) :  }.")
        self.assertEqual(pr(":~ &q(0) {1+2}. [0]"), ":~ &q(0) { +(1,2) :  }. [0@0]")
        self.assertEqual(pr("#edge (u, v) : &q(0) {1+2}."), "#edge (u,v) : &q(0) { +(1,2) :  }.")
        self.assertEqual(pr("#heuristic a : &q(0) {1+2}. [sign,true]"),
                         "#heuristic a : &q(0) { +(1,2) :  }. [sign@0,true]")
        self.assertEqual(pr("#project a : &q(0) {1+2}."), "#project a : &q(0) { +(1,2) :  }.")
        self.assertRaises(RuntimeError, pr, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, pr, "&q(1+2+3) { }.")
        self.assertEqual(pr(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) {  }.")
        self.assertEqual(pr("&r { } < 1+2+3."), "&r {  } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, pr, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, pr, ":- &r { } < 1+2+3.")
        self.assertRaises(RuntimeError, pr, "&s(1+2+3) { }.")
        self.assertRaises(RuntimeError, pr, "&p { } <= 3.")
        self.assertRaises(RuntimeError, pr, "&r { } <= 3.")


def test_rename(s: str, f=lambda s: prefix_symbolic_atoms(s, "u_")):
    '''
    Parse the given program and rename symbolic atoms in it.
    '''
    prg: List[str]
    prg = []
    def append(stm):
        nonlocal prg
        ret = f(stm)
        if ret is not None:
            prg.append(str(ret))
    parse_program(s, append)
    return prg

class TestRenameSymbolicAtoms(TestCase):
    '''
    Tests for renaming symbolic atoms.
    '''
    def test_rename(self):
        '''
        Test renaming symbolic atoms.
        '''
        self.assertEqual(
            test_rename("a :- b(X,Y), not c(f(3,b))."),
            ['#program base.', 'u_a :- u_b(X,Y); not u_c(f(3,b)).'])

        sym = ast.SymbolicAtom(ast.Symbol(LOC, Function('a', [Function('b')])))
        self.assertEqual(
            str(prefix_symbolic_atoms(sym, 'u_')),
            'u_a(b)')


def test_as_dict(s: str):
    '''
    Parse and transform a program to its dictionary representation.
    '''
    prg = []
    parse_program(s, lambda x: prg.append(as_dict(x)))
    preamble = {'type': 'Program', 'location': '<string>:1:1', 'name': 'base', 'parameters': []}
    assert prg[0] == preamble
    return prg[1:]

class TestRepr(TestCase):
    '''
    Tests for converting between python and ast representation of terms.
    '''
    def test_encode_term(self):
        '''
        Test encoding of terms in AST.
        '''
        self.assertEqual(
            test_as_dict('a(1).'),
            [{'type': 'Rule', 'location': '<string>:1:1-6',
              'head': {'type': 'Literal', 'location': '<string>:1:1-5', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-5', 'name': 'a',
                                         'arguments': [{'type': 'Symbol', 'location': '<string>:1:3-4', 'symbol': '1'}],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('a(X).'),
            [{'type': 'Rule', 'location': '<string>:1:1-6',
              'head': {'type': 'Literal', 'location': '<string>:1:1-5', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-5', 'name': 'a',
                                         'arguments': [{'type': 'Variable', 'location': '<string>:1:3-4', 'name': 'X'}],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('a(-1).'),
            [{'type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'type': 'Literal', 'location': '<string>:1:1-6', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-6', 'name': 'a',
                                         'arguments': [{'type': 'UnaryOperation', 'location': '<string>:1:3-5',
                                                        'operator': 'Minus',
                                                        'argument': {'type': 'Symbol', 'location': '<string>:1:4-5',
                                                                     'symbol': '1'}}],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('a(1+2).'),
            [{'type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'type': 'Literal', 'location': '<string>:1:1-7', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                         'arguments': [{'type': 'BinaryOperation', 'location': '<string>:1:3-6',
                                                        'operator': '+',
                                                        'left': {'type': 'Symbol', 'location': '<string>:1:3-4',
                                                                 'symbol': '1'},
                                                        'right': {'type': 'Symbol', 'location': '<string>:1:5-6',
                                                                  'symbol': '2'}}],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('a(1..2).'),
            [{'type': 'Rule', 'location': '<string>:1:1-9',
              'head': {'type': 'Literal', 'location': '<string>:1:1-8', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-8', 'name': 'a',
                                         'arguments': [{'type': 'Interval', 'location': '<string>:1:3-7',
                                                        'left': {'type': 'Symbol', 'location': '<string>:1:3-4',
                                                                 'symbol': '1'},
                                                        'right': {'type': 'Symbol', 'location': '<string>:1:6-7',
                                                                  'symbol': '2'}}],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('a(1;2).'),
            [{'type': 'Rule', 'location': '<string>:1:1-8',
              'head': {'type': 'Literal', 'location': '<string>:1:1-7', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Pool', 'location': '<string>:1:1-7',
                                         'arguments': [{'type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                                        'arguments': [{'type': 'Symbol', 'location': '<string>:1:3-4',
                                                                       'symbol': '1'}],
                                                        'external': False},
                                                       {'type': 'Function', 'location': '<string>:1:1-7', 'name': 'a',
                                                        'arguments': [{'type': 'Symbol', 'location': '<string>:1:5-6',
                                                                       'symbol': '2'}],
                                                        'external': False}]}}},
              'body': []}])

    def test_encode_literal(self):
        '''
        Tests for converting between python and ast representation of literals.
        '''
        self.assertEqual(
            test_as_dict('a.'),
            [{'type': 'Rule', 'location': '<string>:1:1-3',
              'head': {'type': 'Literal', 'location': '<string>:1:1-2', 'sign': 'NoSign',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:1-2', 'name': 'a', 'arguments': [],
                                         'external': False}}},
              'body': []}])
        self.assertEqual(
            test_as_dict('not a.'),
            [{'type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'type': 'Literal', 'location': '<string>:1:1-6', 'sign': 'Negation',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:5-6', 'name': 'a', 'arguments': [],
                                         'external': False}}}, 'body': []}])
        self.assertEqual(
            test_as_dict('not not a.'),
            [{'type': 'Rule', 'location': '<string>:1:1-11',
              'head': {'type': 'Literal', 'location': '<string>:1:1-10', 'sign': 'DoubleNegation',
                       'atom': {'type': 'SymbolicAtom',
                                'term': {'type': 'Function', 'location': '<string>:1:9-10', 'name': 'a',
                                         'arguments': [], 'external': False}}},
              'body': []}])
        # TODO
        print(repr(test_as_dict('a <= b.')))
        print(repr(test_as_dict('a < b.')))
        print(repr(test_as_dict('a >= b.')))
        print(repr(test_as_dict('a > b.')))
        print(repr(test_as_dict('a = b.')))
        print(repr(test_as_dict('a != b.')))
        self.assertEqual(
            test_as_dict('a : b.'),
            [{'type': 'Rule', 'location': '<string>:1:1-7',
              'head': {'type': 'Disjunction', 'location': '<string>:1:1-6',
                       'elements': [{'type': 'ConditionalLiteral', 'location': '<string>:1:1-6',
                                     'literal': {'type': 'Literal', 'location': '<string>:1:1-2', 'sign': 'NoSign',
                                                 'atom': {'type': 'SymbolicAtom',
                                                          'term': {'type': 'Function', 'location': '<string>:1:1-2',
                                                                   'name': 'a', 'arguments': [], 'external': False}}},
                                     'condition': [{'type': 'Literal', 'location': '<string>:1:5-6', 'sign': 'NoSign',
                                                    'atom': {'type': 'SymbolicAtom',
                                                             'term': {'type': 'Function', 'location': '<string>:1:5-6',
                                                                      'name': 'b', 'arguments': [],
                                                                      'external': False}}}]}]},
              'body': []}])
        self.assertEqual(
            test_as_dict(':- a : b.'),
            [{'type': 'Rule', 'location': '<string>:1:1-10',
              'head': {'type': 'Literal', 'location': '<string>:1:1-10', 'sign': 'NoSign',
                       'atom': {'type': 'BooleanConstant', 'value': False}},
              'body': [{'type': 'ConditionalLiteral', 'location': '<string>:1:4-9',
                        'literal': {'type': 'Literal', 'location': '<string>:1:4-5', 'sign': 'NoSign',
                                    'atom': {'type': 'SymbolicAtom',
                                             'term': {'type': 'Function', 'location': '<string>:1:4-5', 'name': 'a',
                                                      'arguments': [], 'external': False}}},
                        'condition': [{'type': 'Literal', 'location': '<string>:1:8-9', 'sign': 'NoSign',
                                       'atom': {'type': 'SymbolicAtom',
                                                'term': {'type': 'Function', 'location': '<string>:1:8-9', 'name': 'b',
                                                         'arguments': [], 'external': False}}}]}]}])
        self.assertEqual(
            test_as_dict('#sum {1:a:b} <= 2.'),
            [{'type': 'Rule', 'location': '<string>:1:1-19',
              'head': {'type': 'HeadAggregate', 'location': '<string>:1:1-18',
                       'left_guard': {'type': 'AggregateGuard', 'comparison': 'GreaterEqual',
                                      'term': {'type': 'Symbol', 'location': '<string>:1:17-18', 'symbol': '2'}},
                       'function': 'Sum',
                       'elements': [{'type': 'HeadAggregateElement',
                                     'tuple': [{'type': 'Symbol', 'location': '<string>:1:7-8', 'symbol': '1'}],
                                     'condition': {'type': 'ConditionalLiteral', 'location': '<string>:1:9-12',
                                                   'literal': {'type': 'Literal', 'location': '<string>:1:9-10',
                                                               'sign': 'NoSign',
                                                               'atom': {'type': 'SymbolicAtom',
                                                                        'term': {'type': 'Function',
                                                                                 'location': '<string>:1:9-10',
                                                                                 'name': 'a', 'arguments': [],
                                                                                 'external': False}}},
                                                   'condition': [{'type': 'Literal', 'location': '<string>:1:11-12',
                                                                  'sign': 'NoSign',
                                                                  'atom': {'type': 'SymbolicAtom',
                                                                           'term': {'type': 'Function',
                                                                                    'location': '<string>:1:11-12',
                                                                                    'name': 'b', 'arguments': [],
                                                                                    'external': False}}}]}}],
                       'right_guard': None},
              'body': []}])
        self.assertEqual(
            test_as_dict(':- #sum {1:b} <= 2.'),
            [{'type': 'Rule', 'location': '<string>:1:1-20',
              'head': {'type': 'Literal', 'location': '<string>:1:1-20', 'sign': 'NoSign',
                       'atom': {'type': 'BooleanConstant', 'value': False}},
              'body': [{'type': 'Literal', 'location': '<string>:1:4-19', 'sign': 'NoSign',
                        'atom': {'type': 'BodyAggregate', 'location': '<string>:1:4-19',
                                 'left_guard': {'type': 'AggregateGuard', 'comparison': 'GreaterEqual',
                                                'term': {'type': 'Symbol', 'location': '<string>:1:18-19',
                                                         'symbol': '2'}},
                                 'function': 'Sum',
                                 'elements': [{'type': 'BodyAggregateElement',
                                               'tuple': [{'type': 'Symbol', 'location': '<string>:1:10-11',
                                                          'symbol': '1'}],
                                               'condition': [{'type': 'Literal', 'location': '<string>:1:12-13',
                                                              'sign': 'NoSign',
                                                              'atom': {'type': 'SymbolicAtom',
                                                                       'term': {'type': 'Function',
                                                                                'location': '<string>:1:12-13',
                                                                                'name': 'b', 'arguments': [],
                                                                                'external': False}}}]}],
                                 'right_guard': None}}]}])
    def test_encode_theory(self):
        '''
        Tests for converting between python and ast representation of theory
        releated constructs.
        '''
        # theory
        #   definition
        #   term
        #   atom

    def test_encode_statement(self):
        '''
        Tests for converting between python and ast representation of statements.
        '''
        chk = [{'type': 'Rule', 'location': '<string>:1:1-8',
                'head': {'type': 'Literal', 'location': '<string>:1:1-2', 'sign': 'NoSign',
                         'atom': {'type': 'SymbolicAtom',
                                  'term': {'type': 'Function', 'location': '<string>:1:1-2',
                                           'name': 'a', 'arguments': [], 'external': False}}},
                'body': [{'type': 'Literal', 'location': '<string>:1:6-7', 'sign': 'NoSign',
                          'atom': {'type': 'SymbolicAtom',
                                   'term': {'type': 'Function', 'location': '<string>:1:6-7',
                                            'name': 'b', 'arguments': [], 'external': False}}}]}]
        self.assertEqual(test_as_dict('a :- b.'), chk)
        # statements
        #   definition
        #   show
        #   show sig
        #   defined
        #   minimize
        #   script
        #   program
        #   external
        #   edge
        #   heuristic
        #   project
        #   project sig
        # print(repr(test_as_dict('a(1;2).')))
