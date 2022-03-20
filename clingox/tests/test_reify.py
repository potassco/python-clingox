'''
Test cases for the ground program and observer.
'''
from unittest import TestCase

from typing import List

from clingo.control import Control
from clingo.symbol import Function, Number, Symbol

from ..reify import Reifier

def _at(idx: int, atoms: List[int]):
    ret = set()
    ret.add(Function('atom_tuple', [Number(idx)]))
    for atm in atoms:
        ret.add(Function('atom_tuple', [Number(idx), Number(atm)]))
    return ret

def _lt(idx: int, lits: List[int]):
    ret = set()
    ret.add(Function('literal_tuple', [Number(idx)]))
    for lit in lits:
        ret.add(Function('literal_tuple', [Number(idx), Number(lit)]))
    return ret

def _tag(inc: bool):
    return {Function('tag', [Function('incremental')])} if inc else set()

def _scc(idx: int, scc: List[int]):
    ret = set()
    for atm in scc:
        ret.add(Function('scc', [Number(idx), Number(atm)]))
    return ret

def _rule(hd, bd, choice=False):
    t = 'choice' if choice else 'disjunction'
    return {Function('rule', [Function(t, [Number(hd)]),
                              Function('normal', [Number(bd)])])}

def _output(sym: Symbol, lt: int):
    return {Function('output', [sym, Number(lt)])}

class TestReifier(TestCase):
    '''
    Tests for the Reifier.
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
