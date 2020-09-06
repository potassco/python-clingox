"""
Simple tests for ast manipulation.
"""

from unittest import TestCase
from typing import List, Optional, cast

from clingo import parse_program
from clingo.ast import AST, ASTType
from .. import ast
from ..ast import (Visitor, Transformer, TheoryTermParser, TheoryParser, TheoryAtomType,
                   theory_parser_from_definition)


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
    v = TestVisitor()
    v.visit_list(prg)
    return v.result


class TestTransformer(Transformer):
    '''
    Simple transformer renaming variables.
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
        x.name = x.name + suffix
        return x

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
    rule: Optional[AST]

    def __init__(self):
        self.atom = None
        self.rule = None

    def visit_TheoryAtom(self, x: AST):
        '''
        Extract theory atom.
        '''
        self.atom = x

    def visit_Rule(self, x: AST):
        '''
        Extract last rule.
        '''
        self.rule = x
        self.visit_children(x)

def theory_atom(s: str) -> AST:
    """
    Convert string to theory term.
    """
    v = Extractor()
    parse_program(f"{s}.", v)
    return cast(AST, v.atom)

def last_rule(s: str) -> AST:
    """
    Convert string to rule.
    """
    v = Extractor()
    parse_program(s, v)
    return cast(AST, v.rule)

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

def parse_rule(s: str, parser: Optional[TheoryParser] = None) -> str:
    """
    Parse the given theory atom using a simple parse table for testing.
    """
    if parser is None:
        parser = TheoryParser(TERM_TABLE, ATOM_TABLE)

    return str(parser(last_rule(s)))


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

    def test_visit(self):
        '''
        Test the visitor.
        '''
        self.assertEqual(test_visit("a(X) :- p(X)."), "rlavlav")

    def test_transform(self):
        '''
        Test the transformer.
        '''
        self.assertEqual(test_transform("a(X) :- p(X)."), "a(X_x) :- p(X_x).")

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

    def test_parse_atom(self):
        '''
        Test parsing of theory atoms.
        '''
        self.assertEqual(parse_atom("&p {1+2}"), "&p { +(1,2) :  }")
        self.assertEqual(parse_atom("&p {1+2+3}"), "&p { +(+(1,2),3) :  }")
        self.assertEqual(parse_atom("&q(1+2+3) { }"), "&q(((1+2)+3)) {  }")
        self.assertEqual(parse_atom("&r { } < 1+2+3"), "&r {  } < +(+(1,2),3)")

    def test_parse_atom_occ(self):
        """
        Test parsing of different theory atom types.
        """
        self.assertEqual(parse_rule("&p {1+2}."), "&p { +(1,2) :  }.")
        self.assertRaises(RuntimeError, parse_rule, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, parse_rule, "&q(1+2+3) { }.")
        self.assertEqual(parse_rule(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) {  }.")
        self.assertEqual(parse_rule("&r { } < 1+2+3."), "&r {  } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, parse_rule, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, parse_rule, ":- &r { } < 1+2+3.")

    def test_parse_theory(self):
        """
        Test creating parsers from theory definitions.
        """
        parser = parse_theory(TEST_THEORY)
        pa = lambda s: parse_atom(s, parser)
        pr = lambda s: parse_rule(s, parser)

        self.assertEqual(parse_atom("&p {1+2}", pa), "&p { +(1,2) :  }")
        self.assertEqual(parse_atom("&p {1+2+3}", pa), "&p { +(+(1,2),3) :  }")
        self.assertEqual(parse_atom("&q(1+2+3) { }", pa), "&q(((1+2)+3)) {  }")
        self.assertEqual(parse_atom("&r { } < 1+2+3", pa), "&r {  } < +(+(1,2),3)")

        self.assertEqual(pr("&p {1+2}."), "&p { +(1,2) :  }.")
        self.assertRaises(RuntimeError, pr, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, pr, "&q(1+2+3) { }.")
        self.assertEqual(pr(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) {  }.")
        self.assertEqual(pr("&r { } < 1+2+3."), "&r {  } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, pr, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, pr, ":- &r { } < 1+2+3.")