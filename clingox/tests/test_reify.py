'''
Test cases for the ground program and observer.
'''

import os
from tempfile import NamedTemporaryFile
from multiprocessing import Process
from unittest import TestCase

from typing import List, Optional

from clingo.control import Control
from clingo.symbol import Function, Number, Symbol
from clingo.__main__ import PyClingoApplication
from clingo.application import clingo_main

from ..reify import Reifier, theory_symbols, reify_program


def _reify(prg, calculate_sccs: bool = False, reify_steps: bool = False):
    return [str(sym) for sym in reify_program(prg, calculate_sccs, reify_steps)]


def _reify_check(prg, calculate_sccs: bool = False, reify_steps: bool = False):
    with NamedTemporaryFile(delete=False) as temp_in, NamedTemporaryFile(delete=False) as temp_out:
        temp_in.write(prg.encode())
        name_in = temp_in.name
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
        args.append(name_in)

        proc = Process(target=clingo_main, args=(PyClingoApplication(), args))
        proc.start()
        proc.join()

        os.fsync(1)
        os.dup2(fd_stdout, 1)
        os.close(fd_stdout)

        with open(name_out, encoding="utf8") as file_out:
            return [s.rstrip('.\n') for s in file_out]

    finally:
        os.unlink(name_in)
        os.unlink(name_out)


def _out(name, args, step: Optional[int]):
    return Function(name, args if step is None else args + [Number(step)])


def _at(idx: int, atoms: List[int], step: Optional[int] = None):
    ret = set()
    ret.add(_out('atom_tuple', [Number(idx)], step))
    for atm in atoms:
        ret.add(_out('atom_tuple', [Number(idx), Number(atm)], step))
    return ret


def _lt(idx: int, lits: List[int], step: Optional[int] = None):
    ret = set()
    ret.add(_out('literal_tuple', [Number(idx)], step))
    for lit in lits:
        ret.add(_out('literal_tuple', [Number(idx), Number(lit)], step))
    return ret


def _tag(inc: bool):
    return {_out('tag', [Function('incremental')], None)} if inc else set()


def _scc(idx: int, scc: List[int], step: Optional[int] = None):
    ret = set()
    for atm in scc:
        ret.add(_out('scc', [Number(idx), Number(atm)], step))
    return ret


def _rule(hd, bd, choice=False, step: Optional[int] = None):
    t = 'choice' if choice else 'disjunction'
    return {_out('rule', [Function(t, [Number(hd)]),
                          Function('normal', [Number(bd)])], step)}


def _output(sym: Symbol, lt: int, step: Optional[int] = None):
    return {_out('output', [sym, Number(lt)], step)}


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
        Test `#step 0. a :- b. b :- a. #step 1. c :- d. d :- c. `.

        TODO: By passing a custom main function to reify_check, we can simplify
        this and get rid of all the helper functions.
        '''
        ctl = Control()
        x = set()
        reifier = Reifier(x.add, True, True)
        ctl.register_observer(reifier)

        with ctl.backend() as bck:
            a = bck.add_atom(Function('a'))
            b = bck.add_atom(Function('b'))
            bck.add_rule([a], [b])
            bck.add_rule([b], [a])
        ctl.solve()
        with ctl.backend() as bck:
            c = bck.add_atom(Function('c'))
            d = bck.add_atom(Function('d'))
            bck.add_rule([c], [d])
            bck.add_rule([d], [c])
        ctl.solve()
        self.assertSetEqual(
            _tag(True) |
            _at(0, [1], 0) | _at(1, [2], 0) |
            _lt(0, [2], 0) | _lt(1, [1], 0) |
            _rule(0, 0, step=0) |
            _rule(1, 1, step=0) |
            _output(Function('a'), 1, 0) |
            _output(Function('b'), 0, 0) |
            _scc(0, [1, 2], 0) |
            _at(0, [3], 1) | _at(1, [4], 1) |
            _lt(0, [4], 1) | _lt(1, [3], 1) |
            _rule(0, 0, step=1) |
            _rule(1, 1, step=1) |
            _output(Function('c'), 1, 1) |
            _output(Function('d'), 0, 1) |
            _scc(0, [3, 4], 1),
            x)

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

    def test_fail(self):
        '''
        Test reification of different language elements.
        '''
        prg = GRAMMAR + '&tel { a <? b: x}. { x }.'
        self.assertListEqual(_reify(prg), _reify_check(prg), prg)

    def test_reify(self):
        '''
        Test reification of different language elements.
        '''
        prgs = [
            # GRAMMAR + '&tel { a <? b: x}. { x }.',
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
