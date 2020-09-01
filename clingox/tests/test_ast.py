"""
Simple tests for ast manipulation.
"""

from unittest import TestCase
from typing import List

from clingo import parse_program
from clingo.ast import AST
from ..ast import Visitor, Transformer


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
    # pylint: disable=invalid-name
    result: str

    def __init__(self):
        self.result = ""

    def visit_Program(self, x: AST, suffix: str):
        '''
        Remove program parts.
        '''
        return None

    def visit_Variable(self, x: AST, suffix: str):
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
    ret = v.visit_list(prg, "_x")
    return "\n".join(str(x) for x in ret if x is not None)


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
