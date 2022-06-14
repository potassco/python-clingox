"""
Test cases for the symbolic symbolic_backend.
"""
from unittest import TestCase

from clingo import Control, Function, HeuristicType, TruthValue
from clingox.backend import SymbolicBackend
from clingox.program import Program, ProgramObserver


class TestSymbolicBackend(TestCase):
    """
    Tests for the ymbolic symbolic_backend.
    """

    def setUp(self):
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)
        self.ctl = Control(message_limit=0)
        self.ctl.register_observer(self.obs)

    def test_add_acyc_edge(self):
        """
        Test edge statement.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_acyc_edge(1, 3, [a], [b, c])
        self.assertEqual(str(self.prg), "#edge (1,3): a(c1), not b(c2), not c(c3).")

    def test_add_assume(self):
        """
        Test assumptions.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_assume([a, b, c])
        self.assertEqual(str(self.prg), "% assumptions: a(c1), b(c2), c(c3)")

    def test_add_external(self):
        """
        Test external statement.
        """
        a = Function("a", [Function("c1")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_external(a, TruthValue.True_)
        self.assertEqual(str(self.prg), "#external a(c1). [True]")

    def test_add_heuristic(self):
        """
        Test heuristic statement.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_heuristic(a, HeuristicType.Level, 2, 3, [b], [c])
        self.assertEqual(
            str(self.prg), "#heuristic a(c1): b(c2), not c(c3). [2@3, Level]"
        )

    def test_add_minimize(self):
        """
        Test minimize statement.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_minimize(1, [(a, 3), (b, 5)], [(c, 7)])
        self.assertEqual(
            str(self.prg), "#minimize{3@1,0: a(c1); 5@1,1: b(c2); 7@1,2: not c(c3)}."
        )

    def test_add_project(self):
        """
        Test project statements.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_project([a, b, c])
        self.assertEqual(
            str(self.prg), "#project a(c1).\n#project b(c2).\n#project c(c3)."
        )

    def test_add_empty_project(self):
        """
        Test project statements.
        """
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_project([])
        self.assertEqual(str(self.prg), "#project x: #false.")

    def test_add_rule(self):
        """
        Test simple rules.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_rule([a], [b], [c])
        self.assertEqual(str(self.prg), "a(c1) :- b(c2), not c(c3).")

    def test_add_choice_rule(self):
        """
        Test choice rules.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_rule([a], [b], [c], choice=True)
        self.assertEqual(str(self.prg), "{a(c1)} :- b(c2), not c(c3).")

    def test_add_weight_rule(self):
        """
        Test weight rules.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_weight_rule([a], 3, [(b, 5)], [(c, 7)])
        self.assertEqual(str(self.prg), "a(c1) :- 3{5,0: b(c2), 7,1: not c(c3)}.")

    def test_add_weight_choice_rule(self):
        """
        Test weight rules that are also choice rules.
        """
        a = Function("a", [Function("c1")])
        b = Function("b", [Function("c2")])
        c = Function("c", [Function("c3")])
        with SymbolicBackend(self.ctl.backend()) as symbolic_backend:
            symbolic_backend.add_weight_rule([a], 3, [(b, 5)], [(c, 7)], choice=True)
        self.assertEqual(str(self.prg), "{a(c1)} :- 3{5,0: b(c2), 7,1: not c(c3)}.")
