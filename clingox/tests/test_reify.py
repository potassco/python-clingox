"""
Test cases for the reify module.
"""

import os
from tempfile import NamedTemporaryFile
from multiprocessing import Process
from unittest import TestCase

from typing import Any, Callable, Dict, Set, Union, cast

from clingo.control import Control
from clingo.symbolic_atoms import SymbolicAtom
from clingo.symbol import Function, Number, Symbol
from clingo.application import Application, clingo_main
from clingo.theory_atoms import TheoryTermType

from ..reify import Reifier, ReifiedTheory, ReifiedTheoryTerm, reify_program
from ..theory import evaluate, is_clingo_operator, is_operator

GRAMMAR = """
#theory theory {
    term { +  : 6, binary, left;
           <? : 5, binary, left;
           <  : 4, unary };
    &tel/0 : term, any;
    &tel2/0 : term, {=}, term, head
}.
"""

THEORY = """
#theory theory {
    t { + : 0, binary, left;
        - : 0, unary };
    &a/0 : t, {=}, t, head;
    &b/0 : t, directive
}.
"""


class _Application(Application):
    def __init__(self, main):
        self._main = main

    def main(self, control, files):
        self._main(control)  # nocoverage


class _AppMain:
    def __init__(self, prg: str):
        self._prg = prg

    def __call__(self, ctl: Control):
        ctl.add("base", [], self._prg)  # nocoverage
        ctl.ground([("base", [])])  # nocoverage
        ctl.solve()  # nocoverage


def _reify(prg, calculate_sccs: bool = False, reify_steps: bool = False):
    if isinstance(prg, str):
        symbols = reify_program(prg, calculate_sccs, reify_steps)
    else:
        ctl = Control()
        symbols = []
        reifier = Reifier(symbols.append, calculate_sccs, reify_steps)
        ctl.register_observer(reifier)
        prg(ctl)

    return [str(sym) for sym in symbols]


def _reify_check(
    prg: Union[str, Callable[[Control], None]],
    calculate_sccs: bool = False,
    reify_steps: bool = False,
):
    with NamedTemporaryFile(delete=False) as temp_out:
        name_out = temp_out.name

    try:
        fd_stdout = os.dup(1)
        fd_out = os.open(name_out, os.O_WRONLY)
        os.dup2(fd_out, 1)
        os.close(fd_out)

        args = ["--output=reify", "-Wnone"]
        if calculate_sccs:
            args.append("--reify-sccs")
        if reify_steps:
            args.append("--reify-steps")

        if isinstance(prg, str):
            app_main = _AppMain(prg)
        else:
            app_main = cast(Any, prg)

        proc = Process(target=clingo_main, args=(_Application(app_main), args))
        proc.start()
        proc.join()

        os.fsync(1)
        os.dup2(fd_stdout, 1)
        os.close(fd_stdout)

        with open(name_out, encoding="utf8") as file_out:
            return [s.rstrip(".\n") for s in file_out]

    finally:
        os.unlink(name_out)


def term_symbols(term: ReifiedTheoryTerm, ret: Dict[int, Symbol]) -> None:
    """
    Represent arguments to theory operators using clingo's `clingo.Symbol`
    class.

    Theory terms are evaluated using `clingox.theory.evaluate_unary` and added
    to the given dictionary using the index of the theory term as key.
    """
    if (
        term.type == TheoryTermType.Function
        and is_operator(term.name)
        and not is_clingo_operator(term.name)
    ):
        term_symbols(term.arguments[0], ret)
        term_symbols(term.arguments[1], ret)
    elif term.index not in ret:
        ret[term.index] = evaluate(term)


def visit_terms(thy: ReifiedTheory, cb: Callable[[ReifiedTheoryTerm], None]):
    """
    Visit the terms occuring in the theory atoms of the given theory.

    This function does not recurse into terms.
    """
    for atm in thy:
        for elem in atm.elements:
            for term in elem.terms:
                cb(term)
        cb(atm.term)
        guard = atm.guard
        if guard:
            cb(guard[1])


def _assume(ctl: Control):
    ctl.add("base", [], "{a;b}.")
    ctl.ground([("base", [])])

    lit_a = cast(SymbolicAtom, ctl.symbolic_atoms[Function("a")]).literal
    lit_b = cast(SymbolicAtom, ctl.symbolic_atoms[Function("b")]).literal
    ctl.solve(assumptions=[lit_a, lit_b])
    ctl.solve(assumptions=[-lit_a, -lit_b])


def _incremental(ctl: Control):
    ctl.add("step0", [], "a :- b. b :- a. {a;b}.")
    ctl.ground([("step0", [])])
    ctl.solve()
    ctl.add("step1", [], "c :- d. d :- c. {c;d}.")
    ctl.ground([("step1", [])])
    ctl.solve()


class TestReifier(TestCase):
    """
    Tests for the Reifier.
    """

    def test_incremental(self):
        """
        Test incremental reification.
        """

        # Note: we use sets here because the reification of sccs does not
        # exactly follow what clingo does. In priniciple, it would be possible
        # to implement this in the same fashion clingo does.
        self.assertSetEqual(
            set(_reify(_incremental, True, True)),
            set(_reify_check(_incremental, True, True)),
        )

    def test_reify(self):
        """
        Test reification of different language elements.
        """

        prgs = [
            _assume,
            GRAMMAR + "&tel { a <? b: x}. { x }.",
            GRAMMAR + '&tel { a("s") <? b({2,3}) }.',
            GRAMMAR + "&tel { a <? b([2,c(1)]) }.",
            GRAMMAR + "&tel { a(s) <? b((2,3)) }.",
            GRAMMAR + "&tel2 { a <? b } = c.",
            "a :- b. b :- a. c :- d. {a; d}.",
            "{ a(1); a(2) } 2. :- a(1..2).",
            ":- not b. {b}.",
            "{ a(1..4) }. :- #count{ X: a(X) } > 2.",
            "a(1..2). #show b(X): a(X).",
            "1{ a(1..2) }. #minimize { X@2: a(X) }.",
            "{ a(1..2)}. #show c: a(_). #show.",
            "#external a. [true]",
            "#external a. [false]",
            "#external a. [free]",
            "#heuristic a. [1,true] {a}.",
            "#project c: a. { a; b; c }. #project b: a.",
            "#edge (a,b): c. {c}.",
        ]
        for prg in prgs:
            self.assertListEqual(_reify(prg), _reify_check(prg))
            self.assertListEqual(
                _reify(prg, reify_steps=True), _reify_check(prg, reify_steps=True)
            )
            self.assertListEqual(
                _reify(prg, calculate_sccs=True), _reify_check(prg, calculate_sccs=True)
            )

    def test_theory(self):
        """
        Test the reified theory class.
        """

        def get_theory(prg):
            symbols = reify_program(prg)
            thy = ReifiedTheory(symbols)
            return list(thy)

        atm1 = get_theory(THEORY + "&a { f(1+ -2): x } = z. { x }.")[0]
        atm2 = get_theory(THEORY + "&a { f((1,2)): x }. { x }.")[0]
        atm3 = get_theory(THEORY + "&a { f([1,2]): x }. { x }.")[0]
        atm4 = get_theory(THEORY + "&a { f({1,2}): x }. { x }.")[0]
        atm5 = get_theory(THEORY + "&a. { x }.")[0]
        self.assertEqual(str(atm1), "&a { f((1)+(-(2))): literal_tuple(1) } = z")
        self.assertEqual(str(atm2), "&a { f((1,2)): literal_tuple(1) }")
        self.assertEqual(str(atm3), "&a { f([1,2]): literal_tuple(1) }")
        self.assertEqual(str(atm4), "&a { f({1,2}): literal_tuple(1) }")
        self.assertEqual(str(atm5), "&a")

        self.assertEqual(
            evaluate(atm1.elements[0].terms[0]), Function("f", [Number(-1)])
        )
        self.assertGreaterEqual(atm1.literal, 1)

        dir1 = get_theory(THEORY + "&b.")[0]
        self.assertEqual(dir1.literal, 0)

        atms = get_theory(THEORY + "&a { 1 }. &a { 2 }. &a { 3 }.")
        self.assertEqual(len(set(atms)), 3)
        self.assertNotEqual(atms[0], atms[1])
        self.assertNotEqual(atms[0] < atms[1], atms[0] > atms[1])

        aele = get_theory(THEORY + "&a { 1; 2; 3 }.")[0]
        self.assertEqual(len(set(aele.elements)), 3)
        self.assertNotEqual(aele.elements[0], aele.elements[1])
        self.assertNotEqual(
            aele.elements[0] < aele.elements[1], aele.elements[0] > aele.elements[1]
        )

        atup = get_theory(THEORY + "&a { 1,2,3 }.")[0]
        self.assertEqual(len(set(atup.elements[0].terms)), 3)
        self.assertNotEqual(atup.elements[0].terms[0], atup.elements[0].terms[1])
        self.assertNotEqual(
            atup.elements[0].terms[0] < atup.elements[0].terms[1],
            atup.elements[0].terms[0] > atup.elements[0].terms[1],
        )

    def test_theory_symbols(self):
        """
        Test function to get symbols in a theory.
        """

        def theory_symbols(prg: str) -> Set[str]:
            ret: Dict[int, Symbol] = {}
            visit_terms(
                ReifiedTheory(reify_program(prg)), lambda term: term_symbols(term, ret)
            )
            return set(str(x) for x in ret.values())

        prg = GRAMMAR + "&tel { a(s) <? b((2,3)) }."
        self.assertSetEqual(theory_symbols(prg), set(["a(s)", "b((2,3))", "tel"]))

        prg = GRAMMAR + '&tel2 { (a("s") <? 2+3) <? b((2,3)) } = z.'
        self.assertSetEqual(
            theory_symbols(prg), set(["5", 'a("s")', "z", "tel2", "b((2,3))"])
        )

        prg = GRAMMAR + "&tel{ a({b,c}) <? c}."
        self.assertRaises(RuntimeError, theory_symbols, prg)
