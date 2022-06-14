"""
Test cases for the ground program and observer.
"""
from unittest import TestCase

from typing import cast

from clingo import Control, Function, HeuristicType, TruthValue

from clingox.program import (
    Edge,
    External,
    Fact,
    Heuristic,
    Minimize,
    Program,
    ProgramObserver,
    Project,
    Remapping,
    Rule,
    Show,
    WeightRule,
    remap,
)


def _remap(prg: Program, mapping=None):
    """
    Add the given program to a backend passing it through an observer and then
    return the observer program.

    The resulting program is initialized with the symbols from the orginial
    program.
    """

    ctl, chk = Control(), Program()
    # note that output atoms are not passed to the backend
    if mapping is None:
        chk.output_atoms = prg.output_atoms
        chk.shows = prg.shows
    else:
        chk.output_atoms = {mapping(lit): sym for lit, sym in prg.output_atoms.items()}
        chk.shows = [cast(Show, remap(x, mapping)) for x in prg.shows]
    chk.facts = prg.facts

    ctl.register_observer(ProgramObserver(chk))

    with ctl.backend() as b:
        prg.add_to_backend(b, mapping)

    return chk


def _plus10(atom):
    """
    Simple mapping adding +10 to every atom.
    """
    return atom + 10


class TestProgram(TestCase):
    """
    Tests for the program observer.
    """

    prg: Program
    obs: ProgramObserver

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)

    def tearDown(self):
        self.prg = Program()
        self.obs = ProgramObserver(self.prg)

    def _add_atoms(self, *atoms: str):
        """
        Generate an output table for the given atom names.
        """
        lit = 1
        lits = []
        out, out10 = {}, {}
        for atom in atoms:
            sym = Function(atom)
            self.obs.output_atom(sym, lit)
            lits.append(lit)
            out[lit] = sym
            out10[_plus10(lit)] = sym
            lit += 1
        return out, out10

    def _check(self, prg, prg10, prg_str):
        """
        Check various ways to remap a program.

        1. No remapping.
        2. Identity remapping via Backend and Control.
        3. Remapping via Backend and Control.
        4. Remapping via remap function without Backend and Control.
        5. Remap a program using the Remapping class.
        """
        self.assertEqual(self.prg, prg)
        self.assertEqual(str(self.prg), prg_str)

        r_prg = _remap(self.prg)
        self.assertEqual(self.prg, r_prg)
        self.assertEqual(str(r_prg), prg_str)

        r_prg10 = _remap(self.prg, _plus10)
        self.assertEqual(r_prg10, prg10)
        self.assertEqual(str(r_prg10), prg_str)

        ra_prg10 = self.prg.copy().remap(_plus10)
        self.assertEqual(ra_prg10, prg10)
        self.assertEqual(str(ra_prg10), prg_str)

        # note that the backend below is just used as an atom generator
        ctl = Control()
        with ctl.backend() as b:
            for _ in range(10):
                b.add_atom()
            rm_prg = prg.copy().remap(
                Remapping(b, self.prg.output_atoms, self.prg.facts)
            )
        self.assertEqual(str(rm_prg), prg_str)

    def test_normal_rule(self):
        """
        Test simple rules.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.rule(False, [1], [2, -3])
        self._check(
            Program(
                output_atoms=out, rules=[Rule(choice=False, head=[1], body=[2, -3])]
            ),
            Program(
                output_atoms=out10,
                rules=[Rule(choice=False, head=[11], body=[12, -13])],
            ),
            "a :- b, not c.",
        )

    def test_aux_lit(self):
        """
        Test printing of auxiliary literals.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.rule(False, [4], [1])
        self.assertEqual(
            self.prg,
            Program(output_atoms=out, rules=[Rule(choice=False, head=[4], body=[1])]),
        )
        self.assertEqual(str(self.prg), "__x4 :- a.")

        prg10 = _remap(self.prg, _plus10)
        self.assertEqual(
            prg10,
            Program(
                output_atoms=out10, rules=[Rule(choice=False, head=[14], body=[11])]
            ),
        )
        self.assertEqual(str(prg10), "__x14 :- a.")

        ctl = Control()
        with ctl.backend() as b:
            b.add_atom()
            rm_prg = self.prg.copy().remap(
                Remapping(b, self.prg.output_atoms, self.prg.facts)
            )
        self.assertEqual(str(rm_prg), "__x5 :- a.")

    def test_facts(self):
        """
        Test simple rules.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.output_atom(Function("d"), 0)
        self._check(
            Program(output_atoms=out, facts=[Fact(Function("d"))]),
            Program(output_atoms=out10, facts=[Fact(Function("d"))]),
            "d.",
        )

    def test_add_choice_rule(self):
        """
        Test choice rules.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.rule(True, [1], [2, -3])
        self._check(
            Program(
                output_atoms=out, rules=[Rule(choice=True, head=[1], body=[2, -3])]
            ),
            Program(
                output_atoms=out10, rules=[Rule(choice=True, head=[11], body=[12, -13])]
            ),
            "{a} :- b, not c.",
        )

    def test_add_weight_rule(self):
        """
        Test weight rules.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self._check(
            Program(
                output_atoms=out,
                weight_rules=[
                    WeightRule(
                        choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)]
                    )
                ],
            ),
            Program(
                output_atoms=out10,
                weight_rules=[
                    WeightRule(
                        choice=True, head=[11], lower_bound=10, body=[(12, 7), (-13, 5)]
                    )
                ],
            ),
            "{a} :- 10{7,0: b, 5,1: not c}.",
        )

    def test_add_weight_choice_rule(self):
        """
        Test weight rules that are also choice rules.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.weight_rule(True, [1], 10, [(2, 7), (-3, 5)])
        self._check(
            Program(
                output_atoms=out,
                weight_rules=[
                    WeightRule(
                        choice=True, head=[1], lower_bound=10, body=[(2, 7), (-3, 5)]
                    )
                ],
            ),
            Program(
                output_atoms=out10,
                weight_rules=[
                    WeightRule(
                        choice=True, head=[11], lower_bound=10, body=[(12, 7), (-13, 5)]
                    )
                ],
            ),
            "{a} :- 10{7,0: b, 5,1: not c}.",
        )

    def test_add_project(self):
        """
        Test project statements.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.project([1, 2])
        self._check(
            Program(output_atoms=out, projects=[Project(atom=1), Project(atom=2)]),
            Program(output_atoms=out10, projects=[Project(atom=11), Project(atom=12)]),
            "#project a.\n#project b.",
        )

    def test_add_empty_project(self):
        """
        Test empty projection statement.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.project([])
        self._check(
            Program(output_atoms=out, projects=[]),
            Program(output_atoms=out10, projects=[]),
            "#project x: #false.",
        )

    def test_add_external(self):
        """
        Test external statement.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.external(1, TruthValue.True_)
        self.obs.external(2, TruthValue.Free)
        self.obs.external(3, TruthValue.False_)
        self._check(
            Program(
                output_atoms=out,
                externals=[
                    External(atom=1, value=TruthValue.True_),
                    External(atom=2, value=TruthValue.Free),
                    External(atom=3, value=TruthValue.False_),
                ],
            ),
            Program(
                output_atoms=out10,
                externals=[
                    External(atom=11, value=TruthValue.True_),
                    External(atom=12, value=TruthValue.Free),
                    External(atom=13, value=TruthValue.False_),
                ],
            ),
            "#external a. [True]\n" "#external b. [Free]\n" "#external c. [False]",
        )

    def test_add_minimize(self):
        """
        Test minimize statement.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.minimize(10, [(1, 7), (3, 5)])
        self._check(
            Program(
                output_atoms=out,
                minimizes=[Minimize(priority=10, literals=[(1, 7), (3, 5)])],
            ),
            Program(
                output_atoms=out10,
                minimizes=[Minimize(priority=10, literals=[(11, 7), (13, 5)])],
            ),
            "#minimize{7@10,0: a; 5@10,1: c}.",
        )

    def test_add_edge(self):
        """
        Test edge statement.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.acyc_edge(1, 2, [1])
        self._check(
            Program(output_atoms=out, edges=[Edge(1, 2, [1])]),
            Program(output_atoms=out10, edges=[Edge(1, 2, [11])]),
            "#edge (1,2): a.",
        )

    def test_add_heuristic(self):
        """
        Test heuristic statement.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.heuristic(1, HeuristicType.Level, 2, 3, [2])
        self._check(
            Program(
                output_atoms=out,
                heuristics=[Heuristic(1, HeuristicType.Level, 2, 3, [2])],
            ),
            Program(
                output_atoms=out10,
                heuristics=[Heuristic(11, HeuristicType.Level, 2, 3, [12])],
            ),
            "#heuristic a: b. [2@3, Level]",
        )

    def test_add_assume(self):
        """
        Test assumptions.

        TODO: this test currently fails but probably has to be fixed in clingo
        because assumptions are not observed properly.
        """
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.assume([1, 3])
        self._check(
            Program(output_atoms=out, assumptions=[1, 3]),
            Program(output_atoms=out10, assumptions=[11, 13]),
            "% assumptions: a, c",
        )

    def test_add_show(self):
        """
        Test show statement.
        """
        t = Function("t")
        out, out10 = self._add_atoms("a", "b", "c")
        self.obs.output_term(t, [1])
        self._check(
            Program(output_atoms=out, shows=[Show(t, [1])]),
            Program(output_atoms=out10, shows=[Show(t, [11])]),
            "#show t: a.",
        )

    def test_control(self):
        """
        Test observer together with a control object.
        """
        ctl = Control()
        ctl.register_observer(self.obs)
        ctl.add(
            "base",
            [],
            """\
            b.
            {c}.
            a :- b, not c.
            #minimize{7@10,a:a; 5@10,c:c}.
            #project a.
            #project b.
            #external a.
            """,
        )
        ctl.ground([("base", [])])

        a, b, c = (Function(s) for s in ("a", "b", "c"))
        la, lb, lc = (ctl.symbolic_atoms[sym].literal for sym in (a, b, c))

        self.prg.sort()

        self.assertEqual(
            self.prg,
            Program(
                output_atoms={la: a, lc: c},
                shows=[],
                facts=[Fact(symbol=b)],
                rules=[
                    Rule(choice=False, head=[lb], body=[]),
                    Rule(choice=False, head=[la], body=[-lc]),
                    Rule(choice=True, head=[lc], body=[]),
                ],
                minimizes=[Minimize(priority=10, literals=[(lc, 5), (la, 7)])],
                externals=[External(atom=la, value=TruthValue.False_)],
                projects=[Project(atom=lb), Project(atom=la)],
            ).sort(),
        )
