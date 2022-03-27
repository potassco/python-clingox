'''
Test cases for the reify module.
'''

import os
from tempfile import NamedTemporaryFile
from multiprocessing import Process
from unittest import TestCase

from typing import Any, Callable, Union, cast

from clingo.control import Control
from clingo.symbolic_atoms import SymbolicAtom
from clingo.symbol import Function, Number
from clingo.application import Application, clingo_main

from ..reify import Reifier, ReifiedTheory, reify_program, theory_symbols
from ..theory import evaluate


class _Application(Application):
    def __init__(self, main):
        self._main = main

    def main(self, control, files):
        self._main(control)  # nocoverage


class _AppMain:
    def __init__(self, prg: str):
        self._prg = prg

    def __call__(self, ctl: Control):
        ctl.add('base', [], self._prg)  # nocoverage
        ctl.ground([('base', [])])  # nocoverage
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


def _reify_check(prg: Union[str, Callable[[Control], None]], calculate_sccs: bool = False, reify_steps: bool = False):
    with NamedTemporaryFile(delete=False) as temp_out:
        name_out = temp_out.name

    try:
        fd_stdout = os.dup(1)
        fd_out = os.open(name_out, os.O_WRONLY)
        os.dup2(fd_out, 1)
        os.close(fd_out)

        args = ["--output=reify", "-Wnone"]
        if calculate_sccs:
            args.append('--reify-sccs')
        if reify_steps:
            args.append('--reify-steps')

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
            return [s.rstrip('.\n') for s in file_out]

    finally:
        os.unlink(name_out)


GRAMMAR = """
#theory theory {
    term { <? : 4, binary, left;
           <  : 5, unary };
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


def _assume(ctl: Control):
    ctl.add("base", [], '{a;b}.')
    ctl.ground([('base', [])])

    lit_a = cast(SymbolicAtom, ctl.symbolic_atoms[Function("a")]).literal
    lit_b = cast(SymbolicAtom, ctl.symbolic_atoms[Function("b")]).literal
    ctl.solve(assumptions=[lit_a, lit_b])
    ctl.solve(assumptions=[-lit_a, -lit_b])


def _incremental(ctl: Control):
    ctl.add('step0', [], 'a :- b. b :- a. {a;b}.')
    ctl.ground([('step0', [])])
    ctl.solve()
    ctl.add('step1', [], 'c :- d. d :- c. {c;d}.')
    ctl.ground([('step1', [])])
    ctl.solve()


class TestReifier(TestCase):
    '''
    Tests for the Reifier.
    '''

    def test_incremental(self):
        '''
        Test incremental reification.
        '''

        # Note: we use sets here because the reification of sccs does not
        # exactly follow what clingo does. In priniciple, it would be possible
        # to implement this in the same fashion clingo does.
        self.assertSetEqual(set(_reify(_incremental, True, True)),
                            set(_reify_check(_incremental, True, True)))

    def test_reify(self):
        '''
        Test reification of different language elements.
        '''

        prgs = [
            _assume,
            GRAMMAR + '&tel { a <? b: x}. { x }.',
            GRAMMAR + '&tel { a("s") <? b({2,3}) }.',
            GRAMMAR + '&tel { a <? b([2,c(1)]) }.',
            GRAMMAR + '&tel { a(s) <? b((2,3)) }.',
            GRAMMAR + '&tel2 { a <? b } = c.',
            'a :- b. b :- a. c :- d. {a; d}.',
            '$x$=1.',
            '{ a(1); a(2) } 2. :- a(1..2).',
            ':- not b. {b}.',
            '{ a(1..4) }. :- #count{ X: a(X) } > 2.',
            'a(1..2). #show b(X): a(X).',
            '1{ a(1..2) }. #minimize { X@2: a(X) }.',
            '{ a(1..2)}. #show c: a(_). #show.',
            '#external a. [true]',
            '#external a. [false]',
            '#external a. [free]',
            '#heuristic a. [1,true] {a}.',
            '#project c: a. { a; b; c }. #project b: a.',
            '#edge (a,b): c. {c}.'
        ]
        for prg in prgs:
            self.assertListEqual(_reify(prg), _reify_check(prg))
            self.assertListEqual(_reify(prg, reify_steps=True), _reify_check(prg, reify_steps=True))
            self.assertListEqual(_reify(prg, calculate_sccs=True), _reify_check(prg, calculate_sccs=True))

    def test_theory(self):
        '''
        Test the reified theory class.
        '''
        def get_theory(prg):
            symbols = reify_program(prg)
            thy = ReifiedTheory(symbols)
            return list(thy)

        atm1 = get_theory(THEORY + '&a { f(1+ -2): x } = z. { x }.')[0]
        atm2 = get_theory(THEORY + '&a { f((1,2)): x }. { x }.')[0]
        atm3 = get_theory(THEORY + '&a { f([1,2]): x }. { x }.')[0]
        atm4 = get_theory(THEORY + '&a { f({1,2}): x }. { x }.')[0]
        atm5 = get_theory(THEORY + '&a. { x }.')[0]
        self.assertEqual(str(atm1), '&a { f((1)+(-(2))): literal_tuple(1) } = z')
        self.assertEqual(str(atm2), '&a { f((1,2)): literal_tuple(1) }')
        self.assertEqual(str(atm3), '&a { f([1,2]): literal_tuple(1) }')
        self.assertEqual(str(atm4), '&a { f({1,2}): literal_tuple(1) }')
        self.assertEqual(str(atm5), '&a')

        self.assertEqual(evaluate(atm1.elements[0].terms[0]), Function('f', [Number(-1)]))
        self.assertGreaterEqual(atm1.literal, 1)

        dir1 = get_theory(THEORY + '&b.')[0]
        self.assertEqual(dir1.literal, 0)

        atms = get_theory(THEORY + '&a { 1 }. &a { 2 }. &a { 3 }.')
        self.assertEqual(len(set(atms)), 3)
        self.assertNotEqual(atms[0], atms[1])
        self.assertNotEqual(atms[0] < atms[1],
                            atms[0] > atms[1])

        aele = get_theory(THEORY + '&a { 1; 2; 3 }.')[0]
        self.assertEqual(len(set(aele.elements)), 3)
        self.assertNotEqual(aele.elements[0], aele.elements[1])
        self.assertNotEqual(aele.elements[0] < aele.elements[1],
                            aele.elements[0] > aele.elements[1])

        atup = get_theory(THEORY + '&a { 1,2,3 }.')[0]
        self.assertEqual(len(set(atup.elements[0].terms)), 3)
        self.assertNotEqual(atup.elements[0].terms[0], atup.elements[0].terms[1])
        self.assertNotEqual(atup.elements[0].terms[0] < atup.elements[0].terms[1],
                            atup.elements[0].terms[0] > atup.elements[0].terms[1])

    def test_theory_symbols(self):
        """
        Test function to get symbols in a theory.
        """
        prg = GRAMMAR + '&tel { a(s) <? b((2,3)) }.'
        ret = theory_symbols(ReifiedTheory(reify_program(prg)))
        self.assertListEqual([str(sym) for sym in ret.values()],
                             ['a(s)', 'b((2,3))', 'tel'])

        prg = GRAMMAR + '&tel{ (a("s") <? c) <? b((2,3)) }.'
        ret = theory_symbols(ReifiedTheory(reify_program(prg)))
        self.assertListEqual([str(sym) for sym in ret.values()],
                             ['a("s")', 'c', 'b((2,3))', 'tel'])

        prg = GRAMMAR + '&tel{ a({b,c}) <? c}.'
        self.assertRaises(RuntimeError, theory_symbols, ReifiedTheory(reify_program(prg)))
