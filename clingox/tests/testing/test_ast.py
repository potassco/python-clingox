"""
Tests for the `clingox.testing.ast` module.
"""

import textwrap
from unittest import TestCase

from clingo import ast
from clingo.symbol import Function

from ...testing.ast import ASTTestCase, parse_literal, parse_statement, parse_term

LOC = ast.Location(ast.Position("a", 1, 2), ast.Position("a", 1, 2))


def dedent(text):
    """
    Dedenting with special handling for long lines.
    """
    lines = textwrap.dedent(text).splitlines(keepends=True)
    return "".join(line.replace("\\\n", "") for line in lines)


class TestBasicASTParsing(TestCase):
    """
    Tests for basic AST parsing.
    """

    def test_parse_statement(self):
        """
        Test parse_statement.
        """
        rule = ast.Rule(
            LOC,
            ast.Literal(LOC, 0, ast.SymbolicAtom(ast.Function(LOC, "a", [], 0))),
            [ast.Literal(LOC, 0, ast.SymbolicAtom(ast.Function(LOC, "b", [], 0)))],
        )
        self.assertEqual(parse_statement("a :- b."), rule)
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_statement("a")
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_statement("a.b.")
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_statement("")

    def test_parse_literal(self):
        """
        Test parse_literal.
        """
        lit = ast.Literal(LOC, 0, ast.SymbolicAtom(ast.Function(LOC, "a", [], 0)))
        self.assertEqual(parse_literal("a"), lit)
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_literal("+a")
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_literal("a: b")

    def test_parse_term(self):
        """
        Test parse_term.
        """
        lit = ast.SymbolicTerm(LOC, Function("a", []))
        self.assertEqual(parse_term("a"), lit)
        with self.assertRaisesRegex(RuntimeError, "syntax error"):
            parse_term("+a")


class TestASTTestCaseClass(ASTTestCase):
    """
    Test ASTTestCase class.
    """

    # pylint: disable=invalid-name

    def test_assertASTEqual(self):
        """
        Test assertASTEqual.
        """
        self.assertEqual(parse_term("a"), parse_term("a"))
        self.assertEqual(parse_term("a(b(X))"), parse_term("a(b(X))"))
        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_term("a"), parse_term("b"))
        expected_msg = dedent(
            """\
               'a' != 'b'
               - a
               + b
               """
        )
        self.assertEqual(str(ar.exception), expected_msg)

        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_term("a(b)"), parse_term("a(c)"))
        expected_msg = dedent(
            """\
               'a(b)' != 'a(c)'
               - a(b)
               ?   ^
               + a(c)
               ?   ^
               """
        )
        self.assertEqual(str(ar.exception), expected_msg)

        self.maxDiff = None
        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_literal("a"), parse_term("a"))
        expected_msg = dedent(
            """\
            "ast.Literal(location=LOC,\\n            sign=0[271 chars]))\\n" != \\
            "ast.SymbolicTerm(location=LOC,\\n             [33 chars]))\\n"
            - ast.Literal(location=LOC,
            ?     ^ ^  ^^
            + ast.SymbolicTerm(location=LOC,
            ?     ^^^^^^ ^^  ^
            +                  symbol=Function('a', [], True))
            -             sign=0,
            -             atom=ast.SymbolicAtom(symbol=ast.Function(location=LOC,
            -                                                       name='a',
            -                                                       arguments=[],
            -                                                       external=0)))
            """
        )
        self.assertEqual(str(ar.exception), expected_msg)

    def test_assertEqual(self):
        """
        Test assertEqual.
        """
        self.assertEqual(parse_term("a"), parse_term("a"))
        self.assertEqual(parse_term("a(b(X))"), parse_term("a(b(X))"))
        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_term("a"), parse_term("b"))
        expected_msg = dedent(
            """\
            'a' != 'b'
            - a
            + b
            """
        )
        self.assertEqual(str(ar.exception), expected_msg)

        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_term("a(b)"), parse_term("a(c)"))
        expected_msg = dedent(
            """\
            'a(b)' != 'a(c)'
            - a(b)
            ?   ^
            + a(c)
            ?   ^
            """
        )
        self.assertEqual(str(ar.exception), expected_msg)

        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_literal("a"), parse_term("b"))
        expected_msg = dedent(
            """\
            'a' != 'b'
            - a
            + b
            """
        )
        self.assertEqual(str(ar.exception), expected_msg)

        with self.assertRaises(AssertionError) as ar:
            self.assertEqual(parse_literal("a"), parse_term("a"))
        expected_msg = dedent(
            """\
            "ast.Literal(location=LOC,\\n            sign=0[271 chars]))\\n" != \\
            "ast.SymbolicTerm(location=LOC,\\n             [33 chars]))\\n"
            - ast.Literal(location=LOC,
            ?     ^ ^  ^^
            + ast.SymbolicTerm(location=LOC,
            ?     ^^^^^^ ^^  ^
            +                  symbol=Function('a', [], True))
            -             sign=0,
            -             atom=ast.SymbolicAtom(symbol=ast.Function(location=LOC,
            -                                                       name='a',
            -                                                       arguments=[],
            -                                                       external=0)))
            """
        )
        self.assertEqual(str(ar.exception), expected_msg)
