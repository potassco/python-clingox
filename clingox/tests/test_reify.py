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

from ..reify import Reifier, get_theory_symbols

def _get_command_line_reification(prg):
    with NamedTemporaryFile() as inf, NamedTemporaryFile() as outf:
        inf.write(prg.encode())
        inf.flush()

        old_stdout = os.dup(1)
        devnull = os.open(outf.name, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)

        proc = Process(target=clingo_main,
                       args=(PyClingoApplication(),
                             ["--output=reify", "-Wnone", inf.name]))
        proc.start()
        proc.join()

        os.fsync(1)
        os.dup2(old_stdout, 1)
        os.close(old_stdout)

        return [s.strip('.') for s in outf.read().decode().split('\n') if s != ""]

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
            >?  : 4, binary, left };
    &tel/0     : term, any;
    &tel2/0 : term, {=}, term, head }.
"""
class TestReifier(TestCase):
    '''
    Tests for the Reifier.

    TODO: Add more tests. To make it a bit easier, tests we can also use a meta
    encoding to ensure that the tested programs are equal.
    '''

    def test_simple_with_step(self):
        '''
        Test `a :- b. b :- a.`.
        '''
        ctl = Control()
        x = set()
        reifier = Reifier(x.add, True)
        ctl.register_observer(reifier)
        with ctl.backend() as bck:
            a = bck.add_atom(Function('a'))
            b = bck.add_atom(Function('b'))
            bck.add_rule([a], [b])
            bck.add_rule([b], [a])
        reifier.calculate_sccs()

        self.assertSetEqual(
            _tag(True) |
            _at(0, [1]) | _at(1, [2]) |
            _lt(0, [2]) | _lt(1, [1]) |
            _rule(0, 0) |
            _rule(1, 1) |
            _output(Function('a'), 1) |
            _output(Function('b'), 0) |
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
            '#project c : a. {a;c;b}. #project b: a.'
        ]
        for prg in prgs:
            # print("---------")
            # print(prg)
            ctl = Control()
            x = []
            reifier = Reifier(x.append, False)
            ctl.register_observer(reifier)
            ctl.add("base", [], prg)
            ctl.ground([('base', [])])
            out_str = [str(s) for s in x]
            r = _get_command_line_reification(prg)

            # print('\nFROM COMMAND LINE:')
            # print("\n".join(r))
            # print('\nFROM CLINGOX: ')
            # print("\n".join(out_str))
            self.assertListEqual(out_str, r)

    def test_theory(self):
        """
        Test programs using theory
        """
        prgs = [
            '&tel{ a("s") <? b({2,3}) }.',
            '&tel{ a <? b([2,c(1)]) }.',
            '&tel{ a(s) <? b((2,3)) }.',
            '&tel2{ a <? b } = c.',
            '&tel{ a <? b: x}. {x}.'
        ]
        for p in prgs:
            ctl = Control()
            prg = GRAMMAR + p
            x = []
            reifier = Reifier(x.append, False)
            ctl.register_observer(reifier)
            ctl.add("base", [], prg)
            ctl.ground([('base', [])])
            out_str = [str(s) for s in x]
            r = _get_command_line_reification(prg)

            self.assertListEqual(out_str, r)

    def test_theory_symbols(self):
        """
        Test function to get the theory_symbols
        """
        ctl = Control()
        prg = GRAMMAR + '&tel{ a(s) <? b((2,3)) }.'
        x = []
        reifier = Reifier(x.append, False)
        ctl.register_observer(reifier)
        ctl.add("base", [], prg)
        ctl.ground([('base', [])])
        theory_symbols = get_theory_symbols(x)
        expected_new_symbols = [
            "theory_symbol(0,tel)",
            "theory_symbol(3,s)",
            "theory_symbol(2,a)",
            "theory_symbol(4,a(s))",
            "theory_symbol(8,(2,3))",
            "theory_symbol(5,b)",
            "theory_symbol(9,b((2,3)))",
        ]
        out_str = [str(s) for s in theory_symbols]
        self.assertListEqual(out_str, expected_new_symbols)

        ctl = Control()
        prg = GRAMMAR + '&tel{ (a("s") <? c) <? b((2,3)) }.'
        x = []
        reifier = Reifier(x.append, False)
        ctl.register_observer(reifier)
        ctl.add("base", [], prg)
        ctl.ground([('base', [])])
        theory_symbols = get_theory_symbols(x)
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
        out_str = [str(s) for s in theory_symbols]
        self.assertListEqual(out_str, expected_new_symbols)

        ctl = Control()
        prg = GRAMMAR + '&tel{ a({b,c}) <? c}.'
        x = []
        reifier = Reifier(x.append, False)
        ctl.register_observer(reifier)
        ctl.add("base", [], prg)
        ctl.ground([('base', [])])
        self.assertRaises(RuntimeError, get_theory_symbols, x)
