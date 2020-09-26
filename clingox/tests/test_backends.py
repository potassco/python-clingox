'''
Test cases for the symbolic backend.
'''
from clingox.backends import SymbolicBackend
from unittest import TestCase

from clingo import Control, Function, TruthValue

from clingox.program_observers import (External, Fact, Minimize, Program, ProgramObserver, Project, Rule, WeightRule)


class TestSymbolicBackend(TestCase):
    '''
    Tests for the ymbolic backend.

    TODO:
    - test cases are missing for some statements
    '''

    def setUp(self):
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)
        self.ctl = Control(message_limit=0)
        self.ctl.register_observer(self.obs)


    def test_add_rule(self):
        with SymbolicBackend(self.ctl.backend()) as backend:
            a = Function("a", [Function("c1")])
            b = Function("b", [Function("c2")])
            c = Function("c", [Function("c3")])
            backend.add_rule([a], [b], [c])
        self.assertEqual(str(self.prg), "a(c1) :- b(c2), not c(c3).")