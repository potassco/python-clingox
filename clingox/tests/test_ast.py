"""
Simple tests for ast manipulation.
"""

from unittest import TestCase
from typing import List

from clingo import parse_program
from clingo.ast import AST
from ..ast import Visitor


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


class TestAST(TestCase):
    '''
    Tests for AST manipulation.
    '''

    def test_visit(self):
        '''
        Test the visitor.
        '''
        self.assertEqual(test_visit("a(X) :- p(X)."), "rlavlav")
