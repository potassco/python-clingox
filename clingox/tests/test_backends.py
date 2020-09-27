'''
Test cases for the symbolic symbolic_backend.
'''
from unittest import TestCase

from clingo import Control, Function, TruthValue
from clingox.backends import SymbolicBackend
from clingox.program import (External, Fact, Minimize, Program,
                             ProgramObserver, Project, Rule, WeightRule)


class TestSymbolicBackend(TestCase):
    '''
    Tests for the ymbolic symbolic_backend.

    TODO:
    - test cases are missing for some statements
    '''

    def setUp(self):
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)
        self.ctl = Control(message_limit=0)
        self.ctl.register_observer(self.obs)

    def test_add_atom(self):
        with self.ctl.backend() as backend:
            symbolic_backend = SymbolicBackend(self.ctl.backend())
            a = Function("a", [Function("c1")])
            b = Function("b", [Function("c2")])
            atom_a = symbolic_backend.add_atom(a)
            atom_b = symbolic_backend.add_atom(b)
            backend.add_rule([atom_a], [ atom_b])

        self.assertEqual(str(self.prg), "a(c1) :- b(c2).")

    def test_add_rule(self):
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            a = Function("a", [Function("c1")])
            b = Function("b", [Function("c2")])
            c = Function("c", [Function("c3")])
            symbolic_backend.add_rule([a], [b], [c])
        self.assertEqual(str(self.prg), "a(c1) :- b(c2), not c(c3).")

    def test_add_weight_rule(self):
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            a = Function("a", [Function("c1")])
            b = Function("b", [Function("c2")])
            c = Function("c", [Function("c3")])
            symbolic_backend.add_weight_rule([a], 3, [(b,5)], [(c,7)])
        self.assertEqual(str(self.prg), "a(c1) :- 3{5,0: b(c2), 7,1: not c(c3)}.")
