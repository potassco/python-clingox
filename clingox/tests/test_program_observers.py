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
    - test cases are missing for some statements
    '''

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)

    def tearDown(self):
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)

    def add_atoms(self, *atoms: str):
        '''
        Generate an output table for the given atom names.
        '''
        lit = 1
        lits = []
        out = {}
        for atom in atoms:
            sym = Function(atom)
            self.obs.output_atom(sym, lit)
            lits.append(lit)
            out[lit] = sym
            lit += 1
        return out

    def test_normal_rule(self):
        '''
        Test simple rules.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.rule(False, [1], [2, -3])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            rules=[Rule(choice=False, head=[1], body=[2, -3])]))
        self.assertEqual(str(self.prg), "a :- b, not c.")

    def test_add_choice_rule(self):
        '''
        Test choice rules.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.rule(True, [1], [2, -3])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            rules=[Rule(choice=True, head=[1], body=[2, -3])]))
        self.assertEqual(str(self.prg), "{a} :- b, not c.")

    def test_add_weight_rule(self):
        '''
        Test weight rules.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            weight_rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)])]))
        self.assertEqual(str(self.prg), "{a} :- 10{7,0: b, 5,1: not c}.")

    def test_add_weight_choice_rule(self):
        '''
        Test weight rules that are also choice rules.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            weight_rules=[WeightRule(choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)])]))
        self.assertEqual(str(self.prg), "{a} :- 10{7,0: b, 5,1: not c}.")

    def test_add_project(self):
        '''
        Test project statements.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.project([1, 2])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            projects=[Project(atom=1), Project(atom=2)]))
        self.assertEqual(str(self.prg), "#project a.\n#project b.")

    def test_add_empty_project(self):
        '''
        Test empty projection statement.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.project([])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            projects=[]))
        self.assertEqual(str(self.prg), "#project x: #false.")

    def test_add_external(self):
        '''
        Test external statement.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.external(1, TruthValue.True_)
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            externals=[External(atom=1, value=TruthValue.True_)]))
        self.assertEqual(str(self.prg), "#external a. [True]")

    def test_add_minimize(self):
        '''
        Test minimize statement.
        '''
        out = self.add_atoms("a", "b", "c")
        self.obs.minimize(10, [(1, 7), (3, 5)])
        self.assertEqual(self.prg, Program(
            output_atoms=out,
            minimizes=[Minimize(priority=10, literals=[(1, 7), (3, 5)])]))
        self.assertEqual(str(self.prg), "#minimize{7@10,0: a; 5@10,1: c}.")

    def test_control(self):
        '''
        Test observer together with a control object.
        '''
        ctl = Control()
        ctl.register_observer(self.obs)
        ctl.add("base", [], """\
            b.
            {c}.
            a :- b, not c.
            #minimize{7@10,a:a; 5@10,c:c}.
            #project a.
            #project b.
            #external a.
            """)
        ctl.ground([("base", [])])

        a, b, c = (Function(s) for s in ("a", "b", "c"))
        la, lb, lc = (ctl.symbolic_atoms[sym].literal for sym in (a, b, c))

        self.prg.sort()

        self.assertEqual(self.prg, Program(
            output_atoms={la: a, lc: c},
            shows=[],
            facts=[Fact(symbol=b)],
            rules=[Rule(choice=False, head=[lb], body=[]),
                   Rule(choice=False, head=[la], body=[-lc]),
                   Rule(choice=True, head=[lc], body=[])],
            minimizes=[Minimize(priority=10, literals=[(lc, 5), (la, 7)])],
            externals=[External(atom=la, value=TruthValue.False_)],
            projects=[Project(atom=lb), Project(atom=la)]).sort())
