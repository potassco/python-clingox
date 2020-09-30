"""
Simple tests for term evaluation.
"""

from unittest import TestCase

from clingo import Control, Function
from ..theory import evaluate, require_number


def eval_term(s: str) -> str:
    """
    Evaluate the given theory term and return its string representation.
    """
    ctl = Control()
    ctl.add("base", [], f"""
#theory test {{
    t {{
    +  : 3, unary;
    -  : 3, unary;
    ?  : 3, unary;
    ?  : 3, binary, left;
    ** : 2, binary, right;
    *  : 1, binary, left;
    /  : 1, binary, left;
    \\ : 1, binary, left;
    +  : 0, binary, left;
    -  : 0, binary, left
    }};
    &a/0 : t, head
}}.
&a {{{s}}}.
""")
    ctl.ground([("base", [])])
    for x in ctl.theory_atoms:
        return str(evaluate(x.elements[0].terms[0]))
    assert False


class TestTheory(TestCase):
    '''
    Tests for theory term evaluation.
    '''

    def test_number(self):
        '''
        Test require_number function.
        '''
        self.assertRaises(TypeError, require_number, Function('a'))

    def test_binary(self):
        '''
        Test evaluation of binary terms.
        '''
        self.assertEqual(eval_term("2+3"), "5")
        self.assertEqual(eval_term("2-3"), "-1")
        self.assertEqual(eval_term("2*3"), "6")
        self.assertEqual(eval_term("7/2"), "3")
        self.assertEqual(eval_term("7\\2"), "1")
        self.assertEqual(eval_term("2**3"), "8")

    def test_unary(self):
        '''
        Test evaluation of unary terms.
        '''
        self.assertEqual(eval_term("-1"), "-1")
        self.assertEqual(eval_term("+1"), "1")
        self.assertEqual(eval_term("-f"), "-f")
        self.assertEqual(eval_term("-f(x)"), "-f(x)")
        self.assertEqual(eval_term("-(-f(x))"), "f(x)")

    def test_nesting(self):
        '''
        Test evaluation of nested terms
        '''
        self.assertEqual(eval_term("f(2+3*4,-g(-1))"), "f(14,-g(-1))")
        self.assertEqual(eval_term("f(2+3*4,-g(-1),0)"), "f(14,-g(-1),0)")

    def test_error(self):
        '''
        Test failed term evaluation.
        '''
        self.assertRaises(TypeError, eval_term, "-(1,2)")
        self.assertRaises(RuntimeError, eval_term, "{1}")
        self.assertRaises(AttributeError, eval_term, "?2")
        self.assertRaises(AttributeError, eval_term, "1?2")
        self.assertRaises(ZeroDivisionError, eval_term, "1\\0")
        self.assertRaises(ZeroDivisionError, eval_term, "1/0")
