'''
Test cases for the ground program and observer.
'''

import os
from tempfile import NamedTemporaryFile
from multiprocessing import Process
from unittest import TestCase

from typing import Any, Callable, Union, cast

from clingo.control import Control
from clingo.symbolic_atoms import SymbolicAtom
from clingo.symbol import Function
from clingo.application import Application, clingo_main

from ..reify import Reifier, reify_program, theory_symbols


class _Application(Application):
    def __init__(self, main):
        self._main = main

    def main(self, control, files):
        self._main(control)  # nocoverage


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
            def app_main(ctl: Control):
                ctl.add('base', [], cast(str, prg))  # nocoverage
                ctl.ground([('base', [])])  # nocoverage
                ctl.solve()  # nocoverage
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
#theory theory{
    term { <? : 4, binary, left;
           <  : 5, unary };
    &tel/0 : term, any;
    &tel2/0 : term, {=}, term, head }.
"""


class TestReifier(TestCase):
    '''
    Tests for the Reifier.
    '''

    def test_incremental(self):
        '''
        Test incremental reification.
        '''

        def test_main(ctl: Control):
            ctl.add('step0', [], 'a :- b. b :- a. {a;b}.')
            ctl.ground([('step0', [])])
            ctl.solve()
            ctl.add('step1', [], 'c :- d. d :- c. {c;d}.')
            ctl.ground([('step1', [])])
            ctl.solve()

        # Note: we use sets here because the reification of sccs does not
        # exactly follow what clingo does. In priniciple, it would be possible
        # to implement this in the same fashion clingo does.
        self.assertSetEqual(set(_reify(test_main, True, True)),
                            set(_reify_check(test_main, True, True)))

    def test_assume(self):
        '''
        Test reification of assumptions.
        '''
        symbols = []
        reifier = Reifier(lambda sym: symbols.append(str(sym)), reify_steps=True)

        ctl = Control()
        ctl.register_observer(reifier)

        ctl.add("base", [], '{a;b}.')
        ctl.ground([('base', [])])

        lit_a = ctl.symbolic_atoms[Function("a")].literal
        lit_b = ctl.symbolic_atoms[Function("b")].literal
        ctl.solve(assumptions=[lit_a, lit_b])
        ctl.solve(assumptions=[-lit_a, -lit_b])

        expected = [f'assume({lit_a},0)',
                    f'assume({lit_b},0)',
                    f'assume({-lit_a},1)',
                    f'assume({-lit_b},1)']
        for sym in expected:
            self.assertIn(sym, symbols)

    def test_reify(self):
        '''
        Test reification of different language elements.
        '''

        def assume(ctl: Control):
            ctl.add("base", [], '{a;b}.')
            ctl.ground([('base', [])])

            lit_a = cast(SymbolicAtom, ctl.symbolic_atoms[Function("a")]).literal
            lit_b = cast(SymbolicAtom, ctl.symbolic_atoms[Function("b")]).literal
            ctl.solve(assumptions=[lit_a, lit_b])
            ctl.solve(assumptions=[-lit_a, -lit_b])

        prgs = [
            assume,
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

    def test_theory_symbols(self):
        """
        Test function to get symbols in a theory.
        """
        prg = GRAMMAR + '&tel { a(s) <? b((2,3)) }.'
        x = reify_program(prg)
        t_s = theory_symbols(x)
        expected_new_symbols = [
            "theory_symbol(0,tel)",
            "theory_symbol(3,s)",
            "theory_symbol(2,a)",
            "theory_symbol(4,a(s))",
            "theory_symbol(8,(2,3))",
            "theory_symbol(5,b)",
            "theory_symbol(9,b((2,3)))",
        ]

        out_str = [str(s) for s in t_s]
        self.assertListEqual(out_str, expected_new_symbols)

        # Complex formula with string
        prg = GRAMMAR + '&tel{ (a("s") <? c) <? b((2,3)) }.'
        x = reify_program(prg)

        t_s = theory_symbols(x)
        expected_new_symbols = [
            'theory_symbol(0,tel)',
            'theory_symbol(3,"s")',
            'theory_symbol(2,a)',
            'theory_symbol(4,a("s"))',
            'theory_symbol(5,c)',
            'theory_symbol(10,(2,3))',
            'theory_symbol(7,b)',
            'theory_symbol(11,b((2,3)))',
        ]
        out_str = [str(s) for s in t_s]
        self.assertListEqual(out_str, expected_new_symbols)

        # Error
        prg = GRAMMAR + '&tel{ a({b,c}) <? c}.'
        x = reify_program(prg)
        self.assertRaises(RuntimeError, theory_symbols, x)
