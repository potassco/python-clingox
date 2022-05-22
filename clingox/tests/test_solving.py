"""
Simple tests for solving.
"""

from unittest import TestCase
from clingo.control import Control
from ..solving import approximate_cautions_consequences


class TestApproximateCautionsConsequences(TestCase):
    '''
    Tests for approximate_cautions_consequences.
    '''

    def aux_approximate_cautions_consequences(self, prg: str, expected_lower, expected_upper):
        '''
        Auxiliary function to test approximate_cautions_consequences.
        '''
        ctl = Control()
        ctl.add("base", [], prg)
        ctl.ground([("base", [])])
        lower, upper = approximate_cautions_consequences(ctl)
        lower_str = [str(s) for s in lower]
        upper_str = [str(s) for s in upper]
        lower_str.sort()
        upper_str.sort()
        expected_lower.sort()
        expected_upper.sort()
        self.assertListEqual(expected_lower, lower_str)
        self.assertListEqual(expected_upper, upper_str)

    def test_approximate_cautions_consequences(self):
        '''
        Tests for approximate_cautions_consequences.
        '''
        prg = 'a. {b}. c:- not d. d :- not c. e :- not e.'
        lower = ['a']
        upper = ['a', 'b', 'c', 'd']
        self.aux_approximate_cautions_consequences(prg, lower, upper)

        prg = '{a}. :- not a.'
        lower = ['a']
        upper = ['a']
        self.aux_approximate_cautions_consequences(prg, lower, upper)

        prg = '{a}. :- a.'
        lower = []
        upper = []
        self.aux_approximate_cautions_consequences(prg, lower, upper)

        prg = 'a, b.'
        lower = []
        upper = ['a', 'b']
        self.aux_approximate_cautions_consequences(prg, lower, upper)

        prg = 'a, b. :- not a.'
        lower = ['a']
        upper = ['a']
        self.aux_approximate_cautions_consequences(prg, lower, upper)
