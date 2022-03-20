'''
Test cases for the ground program and observer.
'''
from unittest import TestCase

from typing import List, Optional

from clingo.control import Control
from clingo.symbol import Function, Number, Symbol

from ..reify import Reifier

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

class TestReifier(TestCase):
    '''
    Tests for the Reifier.

    TODO: Add more tests. To make it a bit easier, tests we can also use a meta
    encoding to ensure that the tested programs are equal.
    '''

    def test_simple(self):
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
