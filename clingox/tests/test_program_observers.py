from clingo import Function, TruthValue
import clingo
from clingox.program_observers import External, Minimize, Program, Project, Rule, WeightRule, ProgramObserver
import unittest

class Test(unittest.TestCase):

    def test_add_rule(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.rule(False, [1], [2, -3])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, rules=[Rule(choice=False, head=[1], body=[2, -3])]))
        self.assertEqual(str(prg), "a :- b, not c.")

    def test_add_choice_rule(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.rule(True, [1], [2, -3])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, rules=[Rule(choice=True, head=[1], body=[2, -3])]))
        self.assertEqual(str(prg), "{a} :- b, not c.")

    def test_add_weight_rule(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.weight_rule(True, [1], 10, [(2,7), (-3,5)])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2,7), (-3,5)])]))
        self.assertEqual(str(prg), "{a} :- 10{7:b, 5:not c}.")

    def test_add_weight_choice_rule(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.weight_rule(True, [1], 10, [(2,7), (-3,5)])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2,7), (-3,5)])]))
        self.assertEqual(str(prg), "{a} :- 10{7:b, 5:not c}.")

    def test_add_project(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.project([1,2])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, projects=[Project(atoms=[1,2])]))
        self.assertEqual(str(prg), "#project a,b.")

    def test_add_empty_project(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.project([])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, projects=[Project(atoms=[])]))
        self.assertEqual(str(prg), "#project.")

    def test_add_external(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.external(1, TruthValue.True_)
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, externals=[External(atom=1, value=TruthValue.True_)]))
        self.assertEqual(str(prg), "#external a. % value=True")

    def test_add_minimize(self):
        prg = Program()
        obs = ProgramObserver(prg)
        a = Function("a") # pylint: disable=invalid-name
        b = Function("b") # pylint: disable=invalid-name
        c = Function("c") # pylint: disable=invalid-name
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.minimize(10, [(1,7), (3,5)])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, minimizes=[Minimize(priority=10, literals=[(1,7), (3,5)])]))
        self.assertEqual(str(prg), "#minimize{7@10,a:a; 5@10,c:c}.")

    def test_control(self):
        prg = Program()
        obs = ProgramObserver(prg)
        ctr = clingo.Control()
        ctr.register_observer(obs)
        ctr.add("base", [], """
        b.
        {c}.
        a :- b, not c.
        #minimize{7@10,a:a; 5@10,c:c}.
        #project a.
        #project b.
        #external a. % value=True
        """)
        ctr.ground([("base", [])])
        print(">>>" + str(prg) + ">>>")

        