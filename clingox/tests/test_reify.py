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

from ..reify import Reifier, theory_symbols, reify


def _get_command_line_reification(prg):
    with NamedTemporaryFile(delete=False) as temp_in, NamedTemporaryFile(delete=False) as temp_out:
        temp_in.write(prg.encode())
        name_in = temp_in.name
        name_out = temp_out.name

    try:
        fd_stdout = os.dup(1)
        fd_out = os.open(name_out, os.O_WRONLY)
        os.dup2(fd_out, 1)
        os.close(fd_out)

        proc = Process(target=clingo_main,
                       args=(PyClingoApplication(),
                             ["--output=reify", "-Wnone", name_in]))
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
    term { <?  : 4, binary, left;
            <  : 5, unary   };
    &tel/0     : term, any;
    &tel2/0 : term, {=}, term, head }.
"""


class TestReifier(TestCase):
    '''
    Tests for the Reifier.
    '''

    def test_simple_with_step(self):
        '''
        Test `a :- b. b :- a. c:-d.`.
        '''
        ctl = Control()
        x = set()
        reifier = Reifier(x.add, True)
        ctl.register_observer(reifier)
        with ctl.backend() as bck:
            a = bck.add_atom(Function('a'))
            b = bck.add_atom(Function('b'))
            c = bck.add_atom(Function('c'))
            d = bck.add_atom(Function('d'))
            bck.add_rule([a], [b])
            bck.add_rule([b], [a])
            bck.add_rule([c], [d])
        reifier.calculate_sccs()
        self.assertSetEqual(
            _tag(True) |
            _at(0, [1]) | _at(1, [2]) |
            _lt(0, [2]) | _lt(1, [1]) |
            _rule(0, 0) |
            _rule(1, 1) |
            _output(Function('a'), 1) |
            _output(Function('b'), 0) |
            _at(2, [3]) |
            _lt(2, [4]) | _lt(3, [3]) |
            _rule(2, 2) |
            _output(Function('c'), 3) |
            _output(Function('d'), 2) |
            _scc(0, [1, 2]),
            x)

    def test_incremental(self):
        '''
        Test `#step 0. a :- b. b :- a. #step 1. c :- d. d :- c. `.

        TODO: verify that this is relly what clingo is doing!!!!
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

    def test_external(self):
        '''
        Test external directive
        '''
        prg = '#external a.'
        ctl = Control()
        x = []
        reifier = Reifier(x.append, reify_steps=True)
        ctl.register_observer(reifier)
        ctl.add("base", [], prg)
        ctl.ground([('base', [])])
        ctl.solve()
        ctl.assign_external(Function("a"), True)
        ctl.solve()
        ctl.assign_external(Function("a"), False)
        ctl.solve()
        ctl.release_external(Function("a"))
        out_str = [str(s) for s in x]
        expected = ['literal_tuple(0,0)', 'literal_tuple(0,1,0)', 'external(0,false,0)',
                    'external(0,true,1)', 'external(0,false,2)', 'external(0,false,3)']
        self.assertTrue(all(x in out_str for x in expected))

    def test_assume(self):
        '''
        Test assumed
        '''
        prg = 'b:-a.'
        ctl = Control()
        x = []
        reifier = Reifier(x.append, reify_steps=True)
        ctl.register_observer(reifier)
        ctl.add("base", [], prg)
        ctl.ground([('base', [])])
        ctl.solve(assumptions=[(Function("a"), True)])
        ctl.solve(assumptions=[(Function("a"), False)])
        out_str = [str(s) for s in x]
        expected = ['literal_tuple(0,0)',
                    'literal_tuple(0,-1,0)',
                    'assume(0,0)',
                    'literal_tuple(0,1)',
                    'literal_tuple(0,1,1)',
                    'assume(0,1)']
        self.assertTrue(all(x in out_str for x in expected))

    def test_csp(self):
        '''
        Test csp output
        '''
        prg = '$x$=1.'
        out_str = [str(s) for s in reify(prg, False)]
        r = _get_command_line_reification(prg)
        self.assertListEqual(out_str, r)

    def test_simple(self):
        '''
        Test simple programs without adding the step
        '''
        prgs = [
            '{a(1);a(2)}2. :-a(1..2).',
            ':- not b. {b}.',
            '{a(1..4)}.:- #count{X:a(X)} >2.',
            'a(1..2). #show b(X):a(X).',
            '1{a(1..2)}. #minimize{X@2:a(X)}.',
            '{a(1..2)}. #show c:a(_). #show.',
            '#heuristic a. [1,true] {a}.',
            '#project c : a. {a;c;b}. #project b: a.',
            '#edge (a,b) : c. {c}.'
        ]
        for prg in prgs:
            out_str = [str(s) for s in reify(prg, False)]
            r = _get_command_line_reification(prg)
            self.assertListEqual(out_str, r)

    def test_theory_element(self):
        '''
        Test theory element order
        '''
        # Could be just added to test_theory once libreify is fixed for the order
        prg = GRAMMAR + '&tel{ a <? b: x}. {x}.'
        reify(prg, False)

    def test_theory(self):
        """
        Test programs using theory
        """
        prgs = [
            '&tel{ a("s") <? b({2,3}) }.',
            '&tel{ a <? b([2,c(1)]) }.',
            '&tel{ a(s) <? b((2,3)) }.',
            '&tel2{ a <? b } = c.'
        ]
        for p in prgs:
            prg = GRAMMAR + p
            out_str = [str(s) for s in reify(prg, False)]
            r = _get_command_line_reification(prg)
            self.assertListEqual(out_str, r)

    def test_theory_symbols(self):
        """
        Test function to get the theory_symbols
        """
        prg = GRAMMAR + '&tel{ a(s) <? b((2,3)) }.'
        x = reify(prg, False)
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
        x = reify(prg, False)

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
        x = reify(prg, False)
        self.assertRaises(RuntimeError, theory_symbols, x)
