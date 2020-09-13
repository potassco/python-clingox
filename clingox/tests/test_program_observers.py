'''
Test cases for the ground program and observer.
'''
from unittest import TestCase

from clingo import Control, Function, TruthValue

from clingox.program_observers import (External, Fact, Minimize, Program, ProgramObserver, Project, Rule, WeightRule)


class TestObserver(TestCase):
    '''
    Tests for the program observer.

    TODO:
    - there is too much copy and paste below
    - test cases are missing for other statements
    - a ground program should be parseble by clingo
      - assume statements should be printed as a comment
      - the project statement needs to be printed differently
    '''
    def test_normal_rule(self):
        '''
        Test simple rules.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.rule(False, [1], [2, -3])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            rules=[Rule(choice=False, head=[1], body=[2, -3])]))
        self.assertEqual(str(prg), "a :- b, not c.")

    def test_add_choice_rule(self):
        '''
        Test choice rules.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.rule(True, [1], [2, -3])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            rules=[Rule(choice=True, head=[1], body=[2, -3])]))
        self.assertEqual(str(prg), "{a} :- b, not c.")

    def test_add_weight_rule(self):
        '''
        Test weight rules.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            weight_rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)])]))
        self.assertEqual(str(prg), "{a} :- 10{7,0: b, 5,1: not c}.")

    def test_add_weight_choice_rule(self):
        '''
        Test weight rules that are also choice rules.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            weight_rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)])]))
        self.assertEqual(str(prg), "{a} :- 10{7,0: b, 5,1: not c}.")

    def test_add_project(self):
        '''
        Test project statements.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.project([1, 2])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            projects=[Project(atoms=[1, 2])]))
        self.assertEqual(str(prg), "#project a, b.")

    def test_add_empty_project(self):
        '''
        Test empty projection statement.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.project([])
        self.assertEqual(prg, Program(output_atoms={1: a, 2: b, 3: c}, projects=[Project(atoms=[])]))
        self.assertEqual(str(prg), "#project.")

    def test_add_external(self):
        '''
        Test external statement.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.external(1, TruthValue.True_)
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            externals=[External(atom=1, value=TruthValue.True_)]))
        self.assertEqual(str(prg), "#external a. [True]")

    def test_add_minimize(self):
        '''
        Test minimize statement.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        a, b, c = Function("a"), Function("b"), Function("c")
        a = Function("a")
        b = Function("b")
        c = Function("c")
        obs.output_atom(a, 1)
        obs.output_atom(b, 2)
        obs.output_atom(c, 3)
        obs.minimize(10, [(1, 7), (3, 5)])
        self.assertEqual(prg, Program(
            output_atoms={1: a, 2: b, 3: c},
            minimizes=[Minimize(priority=10, literals=[(1, 7), (3, 5)])]))
        self.assertEqual(str(prg), "#minimize{7@10,0: a; 5@10,1: c}.")

    def test_control(self):
        '''
        Test observer together with a control object.
        '''
        prg = Program()
        obs = ProgramObserver(prg)
        ctl = Control()
        a, b, c = Function("a"), Function("b"), Function("c")
        ctl.register_observer(obs)
        ctl.add("base", [], """\
            b.
            {c}.
            a :- b, not c.
            #minimize{7@10,a:a; 5@10,c:c}.
            #project a.
            #project b.
            #external a. % value=True
            """)
        ctl.ground([("base", [])])
        prg.rules = sorted(prg.rules)
        prg.projects = sorted(prg.projects)
        self.assertEqual(prg, Program(
            output_atoms={3: c, 2: a},
            shows=[],
            facts=[Fact(symbol=b)],
            rules=[Rule(choice=False, head=[1], body=[]),
                   Rule(choice=False, head=[2], body=[-3]),
                   Rule(choice=True, head=[3], body=[])],
            minimizes=[Minimize(priority=10, literals=[(3, 5), (2, 7)])],
            externals=[External(atom=2, value=TruthValue.False_)],
            projects=[Project(atoms=[1]), Project(atoms=[2])]))
