"""
Tests for pretty printing.
"""

from io import StringIO
from sys import version_info
from unittest import TestCase

from clingo.symbol import parse_term as parse_symbol

from .. import pprint as pp
from ..testing.ast import parse_term

REP = """\
ast.Function(location=Location(begin=Position(filename='<string>',
                                              line=1,
                                              column=8),
                               end=Position(filename='<string>',
                                            line=1,
                                            column=12)),
             name='f',
             arguments=[ast.Variable(location=Location(begin=Position(filename='<string>',
                                                                      line=1,
                                                                      column=10),
                                                       end=Position(filename='<string>',
                                                                    line=1,
                                                                    column=11)),
                                     name='X')],
             external=0)\
"""

REP_LOC = """\
ast.Function(location=LOC,
             name='f',
             arguments=[ast.Variable(location=LOC,
                                     name='X')],
             external=0)\
"""

SAFE_REP = """\
ast.Function(Location(begin=Position(filename='<string>', line=1, column=8),\
 end=Position(filename='<string>', line=1, column=12)), 'f',\
 [ast.Variable(Location(begin=Position(filename='<string>', line=1, column=10),\
 end=Position(filename='<string>', line=1, column=11)), 'X')], 0)\
"""

SYM_REP1 = """\
Function('f',
         [Supremum,
          Infimum,
          Function('a', [], True),
          String('b'),
          Function('', [Number(1), Number(2)], True)],
         True)\
"""

SYM_REP2 = """\
Function('f',
         [Function('f',
                   [Function('f',
                             [Function('f',
                                       [Function('f',
                                                 [Function('f',
                                                           [Function('f',
                                                                     [Number(1000000000)],
                                                                     True)],
                                                           True)],
                                                 True)],
                                       True)],
                             True)],
                   True)],
         True)\
"""


class TestPPrint(TestCase):
    """
    Test cases for pretty printing.
    """

    def test_pprint_ast(self):
        """
        Test pprint functions for ASTs.
        """
        self.assertEqual(pp.pformat(parse_term("f(X)")), REP)
        self.assertEqual(pp.pformat(parse_term("f(X)"), hide_location=True), REP_LOC)

    def test_pprint_sym(self):
        """
        Test pprint functions for symbols.
        """
        self.assertEqual(pp.pformat(parse_symbol('f(#sup,#inf,a,"b",(1,2))')), SYM_REP1)
        self.assertEqual(
            pp.pformat(parse_symbol("f(f(f(f(f(f(f(1000000000)))))))")), SYM_REP2
        )

    def test_pprint_module(self):
        """
        Test pprint module functions.
        """

        out = StringIO()
        pp.pprint(parse_term("f(X)"), stream=out)
        self.assertEqual(out.getvalue(), f"{REP}\n")
        self.assertEqual(pp.saferepr(parse_term("f(X)")), SAFE_REP)
        self.assertTrue(pp.isreadable(parse_term("f(X)")))
        self.assertFalse(pp.isrecursive(parse_term("f(X)")))
        if version_info[:2] >= (3, 8):
            out = StringIO()  # nocoverage
            pp.pp(parse_term("f(X)"), stream=out)  # nocoverage
            self.assertEqual(out.getvalue(), f"{REP}\n")  # nocoverage
