"""
This module provides high-level functions to create unit tests for
`clingo.ast.AST`s.
"""

from typing import Any, List, cast
from unittest import TestCase

from clingo.ast import AST, ASTType, parse_string

from clingox.pprint import pformat

__all__ = [
    "ASTTestCase",
    "parse_literal",
    "parse_statement",
    "parse_term",
]


def parse_statement(stm: str) -> AST:
    """
    Parse a statement.
    """
    stms: List[AST] = []
    parse_string(stm, stms.append, logger=lambda code, msg: None, message_limit=1)
    if len(stms) != 2:
        raise RuntimeError(
            f"syntax error: stm must contain exactly one statement, {len(stms)} given"
        )
    return cast(AST, stms[1])


def parse_literal(lit: str) -> AST:
    """
    Parse a literal.
    """
    stm = parse_statement(f":-{lit}.")
    if stm.body[0].ast_type != ASTType.Literal:
        raise RuntimeError("syntax error: lit must be a string representing a literal")
    return stm.body[0]


def parse_term(term: str) -> AST:
    """
    Parse a term.
    """
    lit = parse_literal(f"atom({term})")
    return lit.atom.symbol.arguments[0]


class ASTTestCase(TestCase):
    """
    Class for comparing with `clingo.ast.AST`s.
    """

    def __init__(self, methodName: str = "runTest"):
        """
        Create an instance of the class that will use the named test method
        when executed. Raises a ValueError if the instance does not have a
        method with the specified name.
        """
        super().__init__(methodName)
        self.addTypeEqualityFunc(AST, self.assertASTEqual)

    def assertASTEqual(self, first: AST, second: AST, msg: Any = None):
        """
        Test whether two `clingo.ast.AST`s are equal.
        """
        # pylint: disable=invalid-name
        self.assertIsInstance(first, AST, "First argument is not an AST")
        self.assertIsInstance(second, AST, "Second argument is not an AST")

        self.assertEqual(str(first), str(second), msg)
        first_repr = pformat(first, hide_location=True) + "\n"
        second_repr = pformat(second, hide_location=True) + "\n"
        self.assertEqual(first_repr, second_repr, msg)
        assert first == second
